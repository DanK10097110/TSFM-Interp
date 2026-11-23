"""Score actual recipient patches and audit directional probes against held-aside donor derivatives."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.forward_only.probe import HERE
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL_SPACE


ARMS = ('pc_current','random_current','pc_mean','random_mean','full_cal_current','full_cal_mean','source_forward')


def norm_error(measured,reference):
    return float(np.linalg.norm(measured-reference)/np.linalg.norm(reference))


def analyze(source,out):
    with np.load(out/(source+'_effects.npz')) as e, np.load(out/(source+'_probes.npz')) as p, np.load(
            REPLICATION/'results'/(source+'_seed0.npz')) as a, np.load(FULL_SPACE/'results'/(source+'.npz')) as b, np.load(
                FULL_SPACE/'results'/(source+'_access.npz')) as c:
        held = a['split'] != 'cal'
        for f in (e,p,b,c):
            np.testing.assert_array_equal(a['ids'][held],f['ids'])
            np.testing.assert_array_equal(a['atoms'],f['atoms'])
        active = a['active'][held]
        for f in (e,b,c):
            np.testing.assert_array_equal(active,f['active'])
        truth,atoms,labels = a['native'][held],e['atoms'],e['split']
        oracle = b['source_linear'].astype(float)
        estimate = e['source_linear'].astype(float)
        arrays = {arm:e[arm] for arm in ARMS[:4]}
        arrays.update(full_cal_current=c['current_source_fixed_target'],full_cal_mean=c['mean_source_fixed_target'],source_forward=estimate)
        caps = {arm:e[arm+'_clipped'] for arm in ARMS[:4]}
        caps.update(full_cal_current=c['current_source_fixed_target_clipped'],full_cal_mean=c['mean_source_fixed_target_clipped'])
        sizes = p['test_steps'][active]
        actual_norm = (p['codes']*p['decoder_norm'][None])[active]
        ratio = sizes/actual_norm
        assert np.all(ratio <= .2500001) and np.all(sizes>0)
        absolute = p['half_difference'][active]
        details = dict(cases=int(active.sum()), donor_test_perturbed_forward_rows=2*int(active.sum()),
            calibration_perturbed_rows_per_target_span=96*24*2, donor_mean_control_perturbed_cal_rows=96*16*2,
            calibration_rows_for_source_shape_weight=96*24*2, cal_seconds=float(p['cal_seconds']),
            test_probe_seconds=float(p['test_seconds']), unperturbed_radius=float(p['radius']),
            probe_step_min=float(sizes.min()),probe_to_removal_median=float(np.median(ratio)),probe_to_removal_max=float(ratio.max()),
            half_output_rms_quantiles=np.quantile(np.sqrt(np.mean(absolute.astype(float)**2,axis=1)),[0,.01,.5,.99,1]).tolist(),
            all_zero_half_difference=int(np.sum(np.max(np.abs(absolute),axis=1)==0)),
            derivative_vs_autograd_relative_error=norm_error(estimate[active],oracle[active]),
            tiny_forward_estimate_vs_actual_removal_relative_error=norm_error(estimate[active],truth[active]))
    result = dict(probe_audit=details, domains={})
    for domain in ('daily','machines'):
        sub = labels==domain
        counts = active[sub].sum(axis=0)
        frequent = counts>=8
        firing = active[sub][:,frequent]
        boot = resampling_counts(len(firing))
        result['domains'][domain] = cell = dict(n=len(firing),n_frequent=int(frequent.sum()),n_active=int(firing.sum()),
            atoms=atoms[frequent].tolist(),all_firing_counts=counts.tolist(),
            capped={arm:int(clips[sub][:,frequent].sum()) for arm,clips in caps.items()})
        for name in ('full','centered'):
            pts,draws = {},{}
            ref = truth[sub][:,frequent].astype(float)
            if name=='centered':
                ref -= ref.mean(axis=-1,keepdims=True)
            for arm,predictions in arrays.items():
                pred = predictions[sub][:,frequent].astype(float)
                if name=='centered':
                    pred -= pred.mean(axis=-1,keepdims=True)
                per,point,draws[arm] = aggregate(firing,pred,ref,boot)
                pts[arm] = {k:ci(point[k],draws[arm][k]) for k in ('cosine','skill','gain')}
                pts[arm]['atom_skill'] = per['skill'].tolist()
            for arm in ('random_current','pc_mean','full_cal_current'):
                pts['pc_current_minus_'+arm] = ci(pts['pc_current']['skill']['value']-pts[arm]['skill']['value'],
                    draws['pc_current']['skill']-draws[arm]['skill'])
            cell[name] = pts
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args()
    report={s:analyze(s,a.out) for s in ('bolt','timesfm')}
    (a.out/'scores.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
