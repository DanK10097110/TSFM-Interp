"""Replay a fixed map and four shams with doses expressed in native recipient units."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import (
    ALIGN, CAL, CUES, DATA_ROOT, FIT, N_ALIGN, TEST, adapter_for, data, desired,
    learn_directions, live_effect, scale_factors, sites_and_states,
)
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite

HERE=Path(__file__).resolve().parent
ARCHIVES=HERE.parent/'results'


def normalize_dose(mapped, shams, native):
    """Keep the mapped and shuffled write axes, assigning a common native norm per request."""
    directions=np.concatenate([mapped[:,None],shams],axis=1)
    original=np.linalg.norm(directions,axis=-1)
    wanted=np.linalg.norm(native,axis=1)
    if np.any(~np.isfinite(original)) or np.any(~np.isfinite(wanted)) or np.min(original)<=1e-12 or np.min(wanted)<=1e-12:
        raise ValueError('Native, transported and sham writes must all have nonzero norm')
    np.testing.assert_allclose(original,np.repeat(original[:,:1],original.shape[1],axis=1),rtol=1e-7)
    ratio=wanted[:,None]/original
    adjusted=directions*ratio[:,:,None]
    np.testing.assert_allclose(np.linalg.norm(adjusted,axis=2),np.broadcast_to(wanted[:,None],original.shape),rtol=1e-7)
    return adjusted,ratio


def run(target,cache,batch):
    source='timesfm' if target=='bolt' else 'bolt'
    ids,x=data(DATA_ROOT)
    if len(set(ids))!=len(ids):
        raise ValueError('Overlapping study item IDs')
    records={}; states={}; native={}
    for name in [source,target]:
        with np.load(cache/(name+'_fit.npz')) as f:
            np.testing.assert_array_equal(ids,f['ids'])
            records[name]=json.loads(str(f['metadata']))
            states[name]=f['s0']; native[name]=f['d0']
        np.testing.assert_allclose(records[name]['fit'][0]['coef_norm'],np.linalg.norm(native[name],axis=1),rtol=1e-8)
    meta=json.loads((ARCHIVES/(target+'.json')).read_text())
    for name in records:
        if meta['source_fits'][name]!=records[name]:
            raise ValueError('Archived fit metadata differs')
    for s,k in [(ALIGN,'clean_fit_ids'),(FIT,'source_fit_ids'),(CAL,'calibration_ids'),(TEST,'test_ids')]:
        np.testing.assert_array_equal(ids[s],meta[k])
    if meta['depths'][0]!=.5 or meta['requests']!=CUES:
        raise ValueError('Unexpected study site or goal labels')
    perm_rng=np.random.default_rng(202611207)
    permutations=[perm_rng.permutation(N_ALIGN) for _ in range(4)]
    mapped,shams=learn_directions(states[source][ALIGN],states[target][ALIGN],native[source],permutations)
    units,ratio=normalize_dose(mapped,shams,native[target])
    key='0_'+source
    with np.load(ARCHIVES/(target+'.npz')) as f:
        np.testing.assert_array_equal(ids[TEST],f['ids']); np.testing.assert_array_equal(desired(),f['desired'])
        native_ref=f['0_native']; direct_ref=f[key]
    with np.load(ARCHIVES/('dose_shams_'+target+'.npz')) as f:
        np.testing.assert_array_equal(ids[TEST],f['ids'])
        raw_scale=np.column_stack([meta['applied_dose'][key],f[key+'_scales']])
    model=SteeringModel.from_adapter(adapter_for(target),quantiles=[.5])
    site=InterventionSite(**records[target]['site'][0]); norms=np.linalg.norm(states[target],axis=1)
    if np.min(norms)<=1e-6:
        raise ValueError('Clean-state normalization floor active')
    sites,fresh=sites_and_states(model,[site.module],x[CAL][:4],4)
    if asdict(sites[site.module])!=asdict(site):
        raise ValueError('Live intervention site changed')
    np.testing.assert_allclose(fresh[site.module],states[target][CAL][:4],rtol=1e-4,atol=4e-5)
    def effect(rows,directions,batch_size):
        return live_effect(model,site,x[rows],norms[rows],directions.reshape(-1,directions.shape[-1]),batch_size)
    check=np.arange(TEST.start,TEST.start+2)
    replay_native=effect(check,native[target],1)
    replay_raw=effect(check,mapped,1)
    np.testing.assert_allclose(replay_native,native_ref[:2],atol=3e-4,rtol=2e-3)
    np.testing.assert_allclose(replay_raw,direct_ref[:2,:,0],atol=3e-4,rtol=2e-3)
    k=len(CUES); a=1+len(permutations)
    probe=effect(CAL,units,batch).reshape(len(x[CAL]),k,a,-1)
    energy=(probe.astype(float)**2).sum(axis=(0,3))
    if np.min(energy)<=1e-15 or not np.isfinite(probe).all():
        raise ValueError('Inactive or nonfinite calibration direction')
    dosed=np.stack([scale_factors(probe[:,:,i],desired()) for i in range(a)],axis=1)
    uncapped=np.einsum('nhat,ht->ha',probe.astype(float),desired())/energy
    np.testing.assert_allclose(dosed,np.clip(uncapped,0,8),rtol=2e-6,atol=1e-7)
    effect_test=effect(TEST,units*dosed[:,:,None],batch).reshape(len(x[TEST]),k,a,-1)
    check_test=effect(check[:1],units*dosed[:,:,None],1).reshape(1,k,a,-1)
    np.testing.assert_allclose(check_test,effect_test[:1],atol=3e-4,rtol=2e-3)
    if not np.isfinite(effect_test).all():
        raise ValueError('Nonfinite held-out forecast')
    np.savez_compressed(HERE/(target+'_effects.npz'),ids=ids[TEST],calibration_ids=ids[CAL],
                        unit_calibration=probe, effects=effect_test, scales=dosed, unconstrained=uncapped,
                        ratio=ratio, original_scales=raw_scale, desired=desired())
    report=dict(source=source,target=target,site=asdict(site),requests=CUES,
                input_meta=records[target]['checkpoint'],source_norm=np.linalg.norm(native[source],axis=1).tolist(),
                native_norm=np.linalg.norm(native[target],axis=1).tolist(),map_norm=np.linalg.norm(mapped,axis=1).tolist(),
                unit_over_original=ratio[:,0].tolist(), original_scales=raw_scale.tolist(), recipient_scales=dosed.tolist(),
                recipient_scale_in_raw_units=(ratio*dosed).tolist(),unit_uncapped=uncapped.tolist(),
                n_unit_probe_rms_above_1e_6=(np.sqrt(np.mean(probe.astype(float)**2,axis=-1))>1e-6).sum(0).tolist(),
                max_clean_state_replay=float(np.max(abs(fresh[site.module]-states[target][CAL][:4]))),
                max_archived_native_replay=float(np.max(abs(replay_native-native_ref[:2]))),
                max_archived_direct_replay=float(np.max(abs(replay_raw-direct_ref[:2,:,0]))),
                max_single_vs_batch=float(np.max(abs(check_test-effect_test[:1]))))
    (HERE/(target+'_doses.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('target',choices=['bolt','timesfm'])
    p.add_argument('--cache',type=Path,default=Path('/tmp/tsfm_forecast_requests_20261120'))
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args(); torch.set_num_threads(3); run(args.target,args.cache,args.batch_size)
