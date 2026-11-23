"""Validate item splits, reference equivalence and artifact fingerprints in a finished run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from research.activation_transfer.boundary.capture import digest


def read(path):
    return json.loads(path.read_text())


def validate(results,artifacts):
    members=read(results/'artifacts.json')
    for name,meta in members.items():
        f=artifacts/name
        if f.stat().st_size!=meta['bytes'] or digest(f)!=meta['sha256']:
            raise ValueError('incorrect artifact '+name)
    daily=read(results/'m4.json')['ids']
    all_ids=sum(daily.values(),[])
    assert len(all_ids)==len(set(all_ids))==256
    reserve=read(results/'fixed.json')['datasets']['reserve']['ids']
    assert len(set(reserve))==80 and not set(reserve)&set(all_ids)
    assert not {int(i) for i in reserve}&set(read(results/'reserve_excluded.json'))
    out={'verified_artifact_count':len(members),'m4_count':len(all_ids),'disjoint_reserve_count':len(reserve),
         'sae_seed_controls':{},'paired_forecast_replays':{},'third_recipient':{}}
    for ds in ['m4','weekly','fast']:
        with np.load(artifacts/f'{ds}_smooth_predictions.npz') as archive:
            for seed in [1,2]:
                with np.load(artifacts/f'{ds}_seed{seed}_predictions.npz') as replica:
                    np.testing.assert_array_equal(archive['ids'],replica['ids'])
                    diffs=[]
                    for m in ['timesfm','bolt']:
                        for ctrl in ['cross_dense','cross_input','native_dense']:
                            k=m+'_'+ctrl
                            diffs.append(float(np.max(np.abs(archive[k]-replica[k]))))
                    assert max(diffs)<2e-4
                    out['sae_seed_controls'][f'{ds}/seed{seed}']=max(diffs)
    for ds in ['weekly','fast']:
        for m in ['timesfm','bolt']:
            report=read(results/f'{ds}_smooth_{m}_forecasts.json')
            y=np.array(report['forecast_changes']['native_dense'])
            for seed in [1,2]:
                rr=read(results/f'{ds}_seed{seed}_{m}_forecasts.json')
                assert report['ids']==rr['ids']
                difference=float(np.max(np.abs(y-np.array(rr['forecast_changes']['native_dense']))))
                assert difference<1e-5 and rr['self_patch_max']<1e-5
                out['paired_forecast_replays'][f'{ds}/{m}/seed{seed}']=difference
    a=read(results/'weekly_third_timesfm_to_chronos2_forecasts.json')
    b=read(results/'weekly_third_bolt_to_chronos2_forecasts.json')
    assert a['ids']==b['ids']
    out['third_recipient']['native_forecast_max_difference']=float(np.max(np.abs(
        np.array(a['forecast_changes']['native_dense'])-np.array(b['forecast_changes']['native_dense']))))
    assert out['third_recipient']['native_forecast_max_difference']<1e-5
    tr=read(results/'replica_training.json')
    for m in ['timesfm','bolt']:
        a,b=tr['seed1_'+m],tr['seed2_'+m]
        assert a['train_state_sha256']==b['train_state_sha256']
        assert a['sha256']!=b['sha256']
        assert a['training']['n_optimizer_steps']==b['training']['n_optimizer_steps']==720
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.write_text(json.dumps(validate(a.results,a.artifacts),indent=2)+'\n')
