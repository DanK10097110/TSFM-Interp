"""Macro-feature skill with paired history and calibration-draw uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.fresh_cohort.data import OUT as FRESH_RESULTS
from research.activation_transfer.shape_transfer.calibration_budget.study import DESIGN, FRESH_ARMS, HERE, SEEDS

COST={f'{n}x{k}':2*n*k for n,k in DESIGN} | {'pilot':768,'wrong':768,'baseline':4608}


def two_way(draws,seed_counts):
    return np.sum(draws.T*seed_counts,axis=1)/draws.shape[0]


def score_cohort(source,cohort,out):
    with np.load(out/(source+'_'+cohort+'.npz')) as f:
        arrays=dict(f)
    path=(REPLICATION/'results'/(source+'_seed0.npz') if cohort=='original' else FRESH_RESULTS/(source+'_native.npz'))
    with np.load(path) as f:
        keep=(f['split']!='cal') if cohort=='original' else np.ones(len(f['ids']),bool)
        for k in ('ids','split'):
            np.testing.assert_array_equal(f[k][keep],arrays[k])
        np.testing.assert_array_equal(f['atoms'],arrays['atoms'])
        np.testing.assert_array_equal(f['source_codes' if cohort=='original' else 'codes'][keep],arrays['codes'])
        truth=f['native'][keep]
    arms=([f'{n}x{k}' for n,k in DESIGN]+['pilot','wrong']) if cohort=='original' else list(FRESH_ARMS)
    arms=['baseline',*arms,'source_est']
    reports={}
    for domain in np.unique(arrays['split']):
        rows=arrays['split']==domain
        active=arrays['codes'][rows]>0
        freq=active.sum(axis=0)>=8
        mask=active[:,freq]
        draws=resampling_counts(len(mask))
        rng=np.random.default_rng(2026112121)
        scount=rng.multinomial(len(SEEDS),np.full(len(SEEDS),1/len(SEEDS)),size=len(draws))
        counts={}
        for arm in arms[:-1]:
            clips=arrays[arm+'_clip']
            if clips.ndim==2: clips=clips[None]
            counts[arm]=clips[:,rows][:,:,freq].sum(axis=(1,2)).tolist()
        reports[str(domain)]=record=dict(n=int(rows.sum()),atoms=arrays['atoms'][freq].tolist(),firings=int(mask.sum()),
            firing_counts=active.sum(axis=0).tolist(),capped_by_seed=counts)
        for centered in (False,True):
            ref=truth[rows][:,freq].astype(float)
            if centered: ref-=ref.mean(axis=-1,keepdims=True)
            metrics={};fit_boot={};fit_point={}
            for arm in arms:
                predictions=arrays[arm]
                if predictions.ndim==3:predictions=predictions[None]
                points=[];boots=[];cos=[];cos_boot=[]
                for candidate in predictions:
                    pred=candidate[rows][:,freq].astype(float)
                    if centered:pred-=pred.mean(axis=-1,keepdims=True)
                    _,p,b=aggregate(mask,pred,ref,draws)
                    points.append(p['skill']);boots.append(b['skill']);cos.append(p['cosine']);cos_boot.append(b['cosine'])
                fit_point[arm]=float(np.mean(points));fit_boot[arm]=np.broadcast_to(np.stack(boots),(len(SEEDS),len(draws)))
                metrics[arm]=dict(skill=ci(fit_point[arm],fit_boot[arm].mean(axis=0)),
                    cosine=ci(np.mean(cos),np.mean(cos_boot,axis=0)),skill_by_seed=points,cosine_by_seed=cos,
                    skill_two_way=ci(fit_point[arm],two_way(fit_boot[arm],scount)))
            contrasts={}
            pairs=[(a,'baseline') for a in arms if a not in ('baseline','source_est')]
            pairs.extend([('4x24','12x8'),('16x24','48x8'),('pilot','16x24'),('pilot','48x8'),('16x24','wrong')])
            for a,b in pairs:
                difference=fit_point[a]-fit_point[b];boot=fit_boot[a]-fit_boot[b]
                contrasts[a+' minus '+b]=dict(history=ci(difference,boot.mean(axis=0)),two_way=ci(difference,two_way(boot,scount)))
            record['centered' if centered else 'full']=dict(arms=metrics,contrasts=contrasts)
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    result={s:{c:score_cohort(s,c,a.out) for c in ('original','fresh')} for s in ('bolt','timesfm')}
    (a.out/'scores.json').write_text(json.dumps(dict(cost=COST,cohorts=result),indent=2,allow_nan=False)+'\n')
