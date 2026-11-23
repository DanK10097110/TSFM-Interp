"""Score physical forecast effects under common history and feature resampling."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from research.activation_transfer.shape_transfer.support_confirmation.data import OUT
from research.activation_transfer.shape_transfer.support_confirmation.supports import ARM_NAMES,SEEDS


def reduce_draws(draws):
    return tuple(np.mean(np.stack(values),axis=0) for values in zip(*draws))


def difference(first,second):
    return entry(tuple(a-b for a,b in zip(first,second)))


def score_archive(z,nboot=3000):
    mask=z['codes']>0
    keep=mask.sum(0)>=8
    events=keep[z['cols']]
    n,nf=len(mask),int(keep.sum())
    report=dict(n_histories=n,events_total=len(z['rows']),events_eligible=int(events.sum()),n_atoms=nf,
                all_atoms=z['atoms'].tolist(),active_counts=mask.sum(0).tolist(),eligible_atom_ids=z['atoms'][keep].tolist(),
                receiver=str(z['target_model']),source=str(z['source_model']),context=int(z['context']),
                positions=[int(v) for v in z['target_pos']],source_positions=[int(v) for v in z['source_pos']],
                fd_source=float(z['fd_source']),fd_receiver=float(z['fd_receiver']),weight=float(z['weight']),
                radius=float(z['radius']),overlaps={},length={},endpoints={})
    for arm in ARM_NAMES:
        if arm+'_overlap' in z:
            all_values=z[arm+'_overlap'];v=all_values[z['rows'][events]]
            report['overlaps'][arm]=dict(history_mean=float(all_values.mean()),history_median=float(np.median(all_values)),
                                         event_median=float(np.median(v)) if len(v) else None,
                                         event_mean=float(v.mean()) if len(v) else None,
                                         event_p10_p90=np.percentile(v,[10,90]).tolist() if len(v) else [])
        if np.any(events):
            ratios=z[arm+'_length'][events]/z['dense_length'][events]
            report['length'][arm]=dict(median=float(np.median(ratios)),minimum=float(np.min(ratios)),p05=float(np.quantile(ratios,.05)),
                    at_least_99pct=float(np.mean(ratios>=.99)),cap_count=int(np.sum(z[arm+'_scale'][events]<1.)))
    if nf==0:return report
    mask=mask[:,keep];hdraws=resampling_counts(n,draws=nboot)
    fdraws=np.random.default_rng(221106).multinomial(nf,np.full(nf,1/nf),nboot)
    def cube(arm,leads,center):
        arr=z[arm][:,leads].astype(float)
        if center:arr-=arr.mean(axis=1,keepdims=True)
        target=np.zeros((n,len(keep),arr.shape[1]))
        target[z['rows'],z['cols']]=arr
        return target[:,keep]
    for segment,leads in [('early',slice(0,8)),('late',slice(8,16))]:
        for center in (True,False):
            for reference in ('dense','source'):
                label='_'.join((segment,'shape' if center else 'full',reference))
                truth=cube(reference,leads,center)
                estimates={arm:intervals(mask,cube(arm,leads,center),truth,hdraws,fdraws)[:3]
                           for arm in (*ARM_NAMES,'dense')}
                estimates['mean_swap']=reduce_draws([estimates[f'swap_{seed}'] for seed in SEEDS])
                cell={key:entry(value) for key,value in estimates.items()};report['endpoints'][label]=cell
                for a,b in [('native','mean_swap'),('native','minimum_overlap'),('native','frequent'),
                            ('native','pca'),('native','isotropic'),('native','own_0'),('frequent','mean_swap'),
                            ('own_24','own_0'),('own_16','own_0')]:
                    cell[f'{a}_minus_{b}']=difference(estimates[a],estimates[b])
                if label=='early_shape_dense':
                    energies=np.sum(truth**2,axis=(0,2));detail=[]
                    for k in ('native','mean_swap','minimum_overlap','frequent','pca','isotropic','own_0','own_24'):
                        members=[f'swap_{s}' for s in SEEDS] if k=='mean_swap' else [k]
                        sse=np.mean([np.sum((cube(arm,leads,center)-truth)**2,axis=(0,2)) for arm in members],axis=0)
                        values=np.where(energies>1e-10*8*mask.sum(0),1-sse/np.maximum(energies,1e-20),np.nan)
                        detail.append((k,[float(x) if np.isfinite(x) else None for x in values]))
                    report['per_atom_early_shape_dense']=dict(detail)
    return report


def score(out):
    result={}
    for cohort in ('long','short'):
        result[cohort]={}
        for receiver in ('bolt','timesfm'):
            with np.load(out/(cohort+'_'+receiver+'.npz')) as z:result[cohort][receiver]=score_archive(dict(z))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    score(p.parse_args().out)
