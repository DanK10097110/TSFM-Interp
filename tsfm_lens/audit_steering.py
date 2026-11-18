"""Independent arithmetic and full-model replay audit of a control-study run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.steering.native import BoltHead
from tsfm_lens.steering.timesfm import TimesFMHead


def audit(source, *, replay=False):
    """Recompute metrics from saved arrays and optionally replay EVERY internal edit."""
    source=Path(source)
    result=json.loads((source/'results.json').read_text())
    p=result['protocol']
    arrays=np.load(source/'interventions.npz')
    baseline=arrays['baseline']
    discrepancies=[]
    for row in result['rows']:
        index=row['seed']-p['seed']
        key=f'{index}_{row["task"]}_{row["dose"]:g}_{row["radius"]:g}_{row["steps"]}'
        actual=arrays[key+'_forecast']-baseline[index]
        target=arrays[key+'_target']
        reference=float(np.sqrt(np.sum((actual.astype(float)-target)**2)/np.sum(target**2)))
        discrepancies.append(abs(reference-row['relative_error']))
    maximum=max(discrepancies)
    if maximum>1e-10:
        raise AssertionError(f'saved error does not recompute: {maximum}')
    for key,entry in result['summary'].items():
        task,dose,radius,steps=key.split('/')
        subset=[r for r in result['rows'] if (r['task'],r['dose'],r['radius'],r['steps'])==
                (task,float(dose),float(radius),int(steps))]
        if len(subset)!=entry['n'] or abs(np.mean([r['relative_error'] for r in subset])-entry['error'])>1e-12:
            raise AssertionError('summary arithmetic mismatch')
        if np.mean([r['relative_error']<.1 for r in subset])!=entry['success']:
            raise AssertionError('summary success count mismatch')
    replay_error=norm_violation=None
    if replay:
        torch.set_num_threads(2)
        head=(BoltHead if p['model']=='bolt' else TimesFMHead)()
        cache=head.capture(arrays['context'])
        replay_error=0.
        norm_violation=0.
        for key in result['summary']:
            task,dose,radius,steps=key.split('/')
            stems=[f'{i}_{task}_{float(dose):g}_{float(radius):g}_{int(steps)}' for i in range(len(baseline))]
            delta=np.stack([arrays[stem+'_delta'] for stem in stems])
            expected=np.stack([arrays[stem+'_forecast'] for stem in stems])
            violation=np.max(np.linalg.norm(delta,axis=1)-float(radius)*np.linalg.norm(cache['hidden'],axis=1))
            norm_violation=max(norm_violation,float(violation))
            for start in range(0,len(delta),16):
                actual=head.predict_edited(arrays['context'][start:start+16],delta[start:start+16])
                normed=(actual-cache['location'][start:start+16,None])/cache['scale'][start:start+16,None]
                error=float(np.max(abs(normed-expected[start:start+16])))
                replay_error=max(replay_error,error)
                if not np.allclose(normed,expected[start:start+16],atol=2e-5,rtol=2e-5):
                    raise AssertionError(f'full model replay mismatch: {error}')
        if norm_violation>2e-5:
            raise AssertionError(f'norm budget exceeded: {norm_violation}')
        if len(head.head._forward_pre_hooks):
            raise AssertionError('intervention hook leaked after prediction')
    record=dict(status='passed',n_interventions=len(result['rows']),metric_max_abs_error=maximum,
                full_model_replay=replay,normalized_replay_max_abs=replay_error,
                norm_budget_violation=norm_violation,
                results_sha256=hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),
                arrays_sha256=hashlib.sha256((source/'interventions.npz').read_bytes()).hexdigest())
    (source/'independent_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    audit(args.source,replay=args.replay)
