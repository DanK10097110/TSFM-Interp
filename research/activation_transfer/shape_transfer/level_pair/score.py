"""Score signed changes in the selected SAE responses and actual forecasts for all isolated edits."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.level_pair.inputs import CUES,HERE
from research.activation_transfer.shape_transfer.level_pair.run import ARMS


def half_difference(curves):
    return .5*(curves[:,1].astype(float)-curves[:,0].astype(float))


def interval(value,samples):
    good=np.isfinite(samples)
    return dict(value=float(value),ci95=np.quantile(samples[good],[.025,.975]).tolist()) if good.any() else dict(value=None,ci95=None)


def mean_stat(v,draws):
    return interval(np.mean(v),draws@v/len(v))


def ratio_stat(num,den,draws,threshold=1e-12):
    total=float(np.sum(den));snum=draws@num;sden=draws@den
    if abs(total)<=threshold:return dict(value=None,ci95=None)
    samples=np.divide(snum,sden,out=np.full(len(draws),np.nan),where=np.abs(sden)>threshold)
    return interval(np.sum(num)/total,samples)


def compare(reference,clamped,draws):
    """Estimate signed mean lift removed and projection of the lost centered waveform."""
    lost=reference-clamped
    original_level=reference.mean(axis=1);remaining_level=clamped.mean(axis=1);lost_level=lost.mean(axis=1)
    centered=reference-original_level[:,None];removed_centered=lost-lost_level[:,None]
    rc2=np.mean(centered*centered,axis=1)
    low,high=np.quantile(draws@original_level/len(original_level),[.025,.975])
    return dict(lift_denominator_ci_excludes_zero=bool(low>0 or high<0),removed_level=mean_stat(lost_level,draws),remaining_level=mean_stat(remaining_level,draws),
        fraction_mean_lift_removed=ratio_stat(lost_level,original_level,draws),
        fraction_shape_projected_away=ratio_stat(np.mean(removed_centered*centered,axis=1),rc2,draws,threshold=1e-10*len(rc2)),
        removed_shape_rms=float(np.sqrt(np.mean(removed_centered**2))),
        remaining_shape_rms=float(np.sqrt(np.mean((clamped-remaining_level[:,None])**2))))


def summarize(out):
    records={}
    with np.load(out/'inputs.npz') as f:
        x=dict(f)
    all_effects={model:dict(np.load(out/(model+'_effects.npz'))) for model in ('bolt','timesfm')}
    for f in all_effects.values():np.testing.assert_array_equal(f['ids'],x['ids'])
    for model,f in all_effects.items():
        records[model]=entry={};other=all_effects['timesfm' if model=='bolt' else 'bolt']
        for domain in np.unique(f['split']):
            here=f['split']==domain;entry[str(domain)]=cohort={}
            for j,cue in enumerate(CUES):
                z=f['edited_z'][:,j];z0=f['clean_z'];either=(z>0).any(1)
                masks={'all':here,'clean_active':here&(z0>0),'either_sign_active':here&either,
                    'joint_clean_active':here&(z0>0)&(other['clean_z']>0)}
                cohort[cue]=summary={}
                for name,keep in masks.items():
                    n=int(keep.sum());summary[name]=cell={'n':n}
                    if not n:continue
                    draws=resampling_counts(n);zc=z0[keep];ze=z[keep];zd=half_difference(ze)
                    n_neutral=1e-5*max(1.,float(z0.max()))
                    cell['codes']=dict(clean_active=int((zc>0).sum()),minus_active=int((ze[:,0]>0).sum()),plus_active=int((ze[:,1]>0).sum()),
                        plus_only=int(((ze[:,1]>0)&(ze[:,0]==0)).sum()),minus_only=int(((ze[:,0]>0)&(ze[:,1]==0)).sum()),
                        half_change=mean_stat(zd,draws),half_pre_change=mean_stat(half_difference(f['edited_pre'][keep,j]),draws),
                        rising=int((zd>n_neutral).sum()),falling=int((zd< -n_neutral).sum()),within_tolerance=int((abs(zd)<=n_neutral).sum()),
                        direction_tolerance=n_neutral,plus_minus_median=float(np.median(ze[:,1]-ze[:,0])),
                        half_after_named_lock=mean_stat(half_difference(f['named_lock_z'][keep,j]),draws),
                        half_after_all_lock=mean_stat(half_difference(f['all_lock_z'][keep,j]),draws),
                        mean_abs_named_to_clean=float(np.mean(abs(f['named_lock_z'][keep,j]-zc[:,None]))),
                        mean_abs_intact_to_clean=float(np.mean(abs(ze-zc[:,None]))))
                    ref=half_difference(f['intact'][keep,j]);level=ref.mean(axis=1)
                    cell.update(intact_lift=mean_stat(level,draws),
                        plus_from_clean=mean_stat((f['intact'][keep,j,1]-f['clean_forecast'][keep]).mean(axis=1),draws),
                        intact_shape_rms=float(np.sqrt(np.mean((ref-level[:,None])**2))),
                        effective_input_half_dose=float(np.mean((x['doses']*x['scales'])[keep,j])),
                        reduced_doses=int((x['scales'][keep,j]<1-1e-8).sum()))
                    arms={a:half_difference(f[a][keep,j]) for a in ARMS[1:]}
                    cell['arms']={a:compare(ref,v,draws) for a,v in arms.items()}
                    # Average three distinct control readouts; no hidden control vectors were averaged.
                    control_mean=np.mean([arms['control'+str(k)] for k in range(3)],axis=0)
                    cell['named_minus_control_removed_level']=mean_stat((control_mean-arms['named']).mean(axis=1),draws)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    (a.out/'scores.json').write_text(json.dumps(summarize(a.out),indent=2,allow_nan=False)+'\n')
