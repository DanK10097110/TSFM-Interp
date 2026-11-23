"""Paired equal-atom and pooled-energy recovery within held-out native H16 forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts,terms
from research.activation_transfer.moment.dictionary_transfer.weather.score import centered,interval,metrics
from research.activation_transfer.shape_transfer.cross_seed_groups.study import ARM,OUT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.score import eligible_mask


def summaries(mat,sw,fw):
    sums=mat.sum(0);n=len(sw)
    draws=(sw@mat.reshape(len(mat),-1)).reshape(n,*mat.shape[1:])
    per=metrics(sums)['skill'];bs=metrics(draws)['skill'];w=fw*np.isfinite(bs)
    crossed=np.divide(np.nansum(bs*w,axis=1),w.sum(1),out=np.full(n,np.nan),where=w.sum(1)>0)
    equal=(float(np.nanmean(per)),np.nanmean(bs,axis=1),crossed)
    pool=(float(metrics(sums.sum(0))['skill']),metrics(draws.sum(1))['skill'],metrics((draws*fw[:,:,None]).sum(1))['skill'])
    return equal,pool,per


def entry(v):return dict(skill=v[0],case95=interval(v[1]),crossed95=interval(v[2]))


def panel(z,fit):
    station=z['station'];r=z['rows'];c=z['cols'];active=z['z']>0;k=eligible_mask(active,station)
    idx=np.flatnonzero(k);n=len(idx);ev=k[c];sd=z['sd'][r,None];ids=z['atoms']
    co=z['target_z'][r,c]>0
    rows=[]
    for j,atom in enumerate(ids):
        use=c==j
        record=dict(atom=int(atom),target=int(fit['match_ids'][j]),events=int(use.sum()),
                    sites=int(np.unique(station[r[use]]).size),cofires=int(np.sum(use&co)),eligible=bool(k[j]),
                    train_pearson=float(fit['match_corr'][j]),train_cos=float(fit['cos_at_match'][j]))
        for arm in ('rank1','rank4','rank16','rotated16'):
            record[arm+'_r2']=float(fit[arm+'_r2'][j])
        v=z['donor'][use]
        record['physical_rms']=float(np.sqrt(np.mean(v**2))) if len(v) else None
        record['physical_shape_rms']=float(np.sqrt(np.mean(centered(v)**2))) if len(v) else None
        rows.append(record)
    meta=dict(n_series_or_station=len(np.unique(station)),n_histories=len(station),n_eligible_atoms=n,atoms=rows,
              source_fires=len(r),eligible_fires=int(ev.sum()),eligible_sites=len(np.unique(station[r[ev]])),cofires=int((ev&co).sum()),
              target_pool=len(fit['pool']),donor_replay_max=float(z['donor_replay_max']),zero_max=float(z['zero_max']),
              score={})
    if n==0:return meta
    sw=resampling_counts(len(np.unique(station)),draws=3000);fw=np.random.default_rng(26112304).multinomial(n,np.ones(n)/n,3000)
    for width,sl in [('h16',slice(None)),('early',slice(0,8)),('late',slice(8,16))]:
        for how in ('centered','full'):
            def wave(key):
                v=z[key][:,sl].astype(float)/sd
                if how=='centered':v=centered(v)
                t=np.zeros((*active.shape,v.shape[1]));t[r,c]=v;return t[:,idx]
            ref=wave('donor');stat={};fits={}
            for arm in ARM:
                value=wave(arm);mom=terms(active[:,idx],value,ref)
                by=np.stack([mom[station==s].sum(0) for s in np.unique(station)])
                eq,po,per=summaries(by,sw,fw);fits[arm]=(eq,po)
                s=by.sum(0);refen=s[:,0].sum();esten=s[:,1].sum()
                cos=metrics(s)['cosine'];mask=active[:,idx]
                stat[arm]=dict(equal=entry(eq),pooled=entry(po),per_atom={str(ids[j]):float(v) if np.isfinite(v) else None for j,v in zip(idx,per)},
                               cosine_macro=float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
                               pooled_rms_ratio=float(np.sqrt(esten/refen)),candidate_sd_rms=float(np.sqrt(np.mean(value[mask]**2))),
                               reference_sd_rms=float(np.sqrt(np.mean(ref[mask]**2))))
            diff={}
            for a,b in [('rank16','forced'),('rank16','own'),('rank16','rotated16'),('rank16','rank1'),('rank16','rank4'),('forced','own')]:
                diff[a+'_minus_'+b]={}
                for i,name in [(0,'equal'),(1,'pooled')]:diff[a+'_minus_'+b][name]=entry(tuple(x-y for x,y in zip(fits[a][i],fits[b][i])))
            meta['score'][width+'_'+how]=dict(arms=stat,contrasts=diff)
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=('bolt','timesfm'));p.add_argument('pop',choices=('daily','monash_traffic','noaa'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    f=dict(np.load(a.out/(a.model+'_fit.npz')));z=dict(np.load(a.out/(a.model+'_'+a.pop+'.npz')))
    (a.out/(a.model+'_'+a.pop+'.json')).write_text(json.dumps(panel(z,f),indent=2,allow_nan=False)+'\n')
