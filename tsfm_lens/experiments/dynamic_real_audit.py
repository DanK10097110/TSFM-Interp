"""Transfer saved synthetic profiles to reused real-source contexts, without refit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from dynamic_steering_study import register
from tsfm_lens.steering import DynamicSteerer,SteeringRequest


def run(args):
    torch.set_num_threads(2);started=time.monotonic()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source);meta=json.loads((source/'metadata.json').read_text())
    series=np.load(source/'corpus.npz')['series'];selected=[i for i,m in enumerate(meta) if m['split']=='evaluation']
    x=np.asarray([series[i,128:256]*meta[i]['scale']+meta[i]['center'] for i in selected],dtype=np.float32)
    protocol=dict(arguments=vars(args),selected_indices=selected,context_slice=[128,256],
        corpus_sha256=hashlib.sha256((source/'corpus.npz').read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256((Path(args.profiles)/'profiles.npz').read_bytes()).hexdigest(),
        fit='none: restore frozen synthetic profiles; normal per-input refinement remains enabled',
        scope='external-validity check on reused public sources; not sealed confirmation; pretraining overlap unknown')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        model=register(args.model,64);controller=DynamicSteerer.load(args.profiles,model)
    except Exception as error:
        (out/'findings.md').write_text(f'# Incomplete real-context audit\n\n{type(error).__name__}: {error}\n\nFailed before evaluation. Protocol retained.\n')
        raise
    rs=[]
    for profile in controller.profiles.values():
        r=profile.request
        if ((r.kind=='level' and r.forecast_window in ('short','long')) or
            (r.kind=='uncertainty' and (r.forecast_window=='all' or r.strength>0)) or
            r.kind=='seasonality'):rs.append(r)
    rs += [SteeringRequest('context_ignore',1.,context_window='recent'),
           SteeringRequest('context_focus',context_window='old',outside_weight=0.)]
    rows=[];arrays={}
    try:
        for k,r in enumerate(rs):
            result=controller.steer(x,r,prediction_length=64,auto_discover=False)
            replay=controller.replay(x,result)
            for j,d in enumerate(result.diagnostics):
                m=meta[selected[j]]
                rows.append(dict(request_index=k,context_index=j,kind=r.kind,strength=r.strength,window=r.forecast_window,
                    dataset=m['dataset'],source_id=m['id'],
                    replay_max_abs=float(np.max(abs(replay[j]-result.forecast[j]))),**d))
            arrays[f'{k}_baseline']=result.baseline;arrays[f'{k}_forecast']=result.forecast;arrays[f'{k}_target']=result.requested_change
            (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(kind=r.kind,window=r.forecast_window,successes=sum(d['achieved'] for d in result.diagnostics),
                total=len(x),mean_error=float(np.mean([d['relative_error'] for d in result.diagnostics])),seconds=time.monotonic()-started)),flush=True)
        learned=[r for r in rows if r['site']['source']!='context_policy']
        summary=dict(n=len(rows),learned_n=len(learned),learned_successes=sum(r['achieved'] for r in learned),
            learned_mean_error=float(np.mean([r['relative_error'] for r in learned])),
            input_policy_n=len(rows)-len(learned),replay_max_abs=max(r['replay_max_abs'] for r in rows),seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez_compressed(out/'forecasts.npz',contexts=x,**arrays)
        (out/'findings.md').write_text('# Frozen profile transfer to real contexts\n\n'+json.dumps(summary,indent=2)+
            '\n\nNo source-specific refit or setting changes. Profiles were selected on synthetic validation contexts. '
            'Normal per-input response refinement remains active; this is not a fixed steering vector. '
            'Public sources are reused from earlier repository research, with unknown pretraining overlap. '
            'Only context values are used; this is a control audit, not an accuracy or calibration study.\n')
    except Exception as error:
        (out/'findings.md').write_text(f'# Incomplete real-context audit\n\n{type(error).__name__}: {error}\n\nPartial rows retained.\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['bolt','timesfm'],required=True)
    p.add_argument('--profiles',required=True);p.add_argument('--out',required=True)
    p.add_argument('--source',default='runs/real_background_poc492306')
    run(p.parse_args())
