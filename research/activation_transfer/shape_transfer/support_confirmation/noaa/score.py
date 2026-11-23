"""Paired station-block uncertainty on original donor atoms which fire in eight distinct stations."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import metrics,resampling_counts,terms
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import OUT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.run import SEEDS,SUPPORT
from research.activation_transfer.shape_transfer.support_confirmation.score import reduce_draws


def eligible_mask(active,station,minimum=8):
    return np.stack([active[station==s].any(0) for s in np.unique(station)]).sum(0)>=minimum


def cluster_skill(active,estimate,truth,station,blocks,features,equal_stations=False):
    material=terms(active,estimate,truth)
    by_station=np.stack([material[station==s].sum(0) for s in np.unique(station)])
    if equal_stations:
        firing=by_station[:,:,-1]
        by_station=by_station/np.maximum(1.,firing)[:,:,None]
    per=metrics(by_station.sum(0))['skill'];valid=np.isfinite(per)
    if not valid.any():raise ValueError('No measurable donor feature in this endpoint')
    boot=(blocks@by_station.reshape(len(by_station),-1)).reshape(len(blocks),*by_station.shape[1:])
    skills=metrics(boot)['skill'][:,valid]
    weights=features[:,valid]*np.isfinite(skills)
    crossed=np.divide(np.nansum(weights*skills,axis=1),weights.sum(1),
                      out=np.full(len(blocks),np.nan),where=weights.sum(1)>0)
    return np.nanmean(per),np.nanmean(skills,axis=1),crossed


def cohort(z):
    active=z['codes']>0;station=z['station'];origin=z['origin'];ids=z['atoms'];rows=z['rows'];cols=z['cols']
    sites=np.unique(station)
    k=eligible_mask(active,station)
    live=k[cols];m=active[:,k];nf=int(k.sum())
    per=[]
    for c,atom in enumerate(ids):
        per.append(dict(atom=int(atom),station_count=int(np.unique(station[active[:,c]]).size),
                        origin_count=int(active[:,c].sum()),per_season=[int(active[origin==i,c].sum()) for i in range(4)],
                        eligible=bool(k[c])))
    result=dict(stations=len(sites),histories=len(station),n_eligible_station=int(np.unique(station[rows[live]]).size),
                source_fires_total=len(rows),source_fires_eligible=int(live.sum()),n_eligible_atoms=nf,
                eligible_atoms=ids[k].tolist(),per_atom=per,
                eligible_firing_events_by_season=[int(np.sum(live&(origin[rows]==i))) for i in range(4)],
                fd_error=float(z['fd_target']),statistics={},overlap={},realized_norm_fraction={},per_atom_shape={},reference_rms={})
    if not nf:return result
    for ref_name in ('source','dense'):
        for unit,suffix in [('history_sd',''),('celsius','_celsius')]:
            for lead,sl in [('early',slice(0,8)),('late',slice(8,16))]:
                value=z[ref_name+suffix][live,sl].astype(float)
                cen=value-value.mean(1,keepdims=True)
                result['reference_rms']['_'.join([lead,ref_name,unit])]={
                    'full':float(np.sqrt(np.mean(value**2))),'shape':float(np.sqrt(np.mean(cen**2))),
                    'shape_by_atom':{str(ids[c]):float(np.sqrt(np.mean(cen[cols[live]==c]**2))) for c in np.flatnonzero(k)}}
    count=resampling_counts(len(sites),draws=3000)
    fweights=np.random.default_rng(221106).multinomial(nf,np.full(nf,1/nf),3000)
    def summary(v):return dict(mean=float(v.mean()),p10_p50_p90=np.quantile(v,[.1,.5,.9]).tolist())
    for seed in SEEDS:result['overlap'][str(seed)]=summary(z[f'shared_{seed}'][rows[live]])
    for sup in SUPPORT:
        result['overlap'][sup]=summary(np.array([np.intersect1d(a,b).size for a,b in zip(z[sup+'_ids'],z['native_ids'])])[rows[live]])
        for policy in ('current',*map(str,SEEDS)):
            key=sup+'_'+policy;result['realized_norm_fraction'][key]=summary(z[key+'_norm'][live]/z['budget'][live])
    for seg,sl in [('early',slice(0,8)),('late',slice(8,16))]:
        for kind in ('shape','full','level'):
            def wave(y):
                v=y[:,sl].astype(float)
                if kind=='shape':v-=v.mean(1,keepdims=True)
                if kind=='level':v=v.mean(1,keepdims=True)
                t=np.zeros((len(station),len(k),v.shape[1]))
                t[rows,cols]=v
                return t[:,k]
            for reference in ('dense','source'):
                truth=wave(z[reference])
                for equal in ([False,True] if seg=='early' and kind=='shape' else [False]):
                    label='_'.join([seg,kind,reference]+(['station_equal'] if equal else []))
                    fits={}
                    for sup in SUPPORT:
                        one=[cluster_skill(m,wave(z[f'{sup}_{seed}']),truth,station,count,fweights,equal) for seed in SEEDS]
                        fits[sup+'_wrong']=reduce_draws(one)
                        fits[sup+'_current']=cluster_skill(m,wave(z[sup+'_current']),truth,station,count,fweights,equal)
                    estimates={key:entry(value) for key,value in fits.items()}
                    for a,b in [('native_current','zero_current'),('native_current','frequent_current'),
                                ('native_current','native_wrong'),('zero_current','zero_wrong'),
                                ('frequent_current','frequent_wrong'),('native_wrong','zero_wrong')]:
                        estimates[a+'_minus_'+b]=entry(tuple(p-q for p,q in zip(fits[a],fits[b])))
                    result['statistics'][label]=estimates
                    if kind=='shape' and not equal:
                        per_feature={}
                        for sup in SUPPORT:
                            def atom_scores(tag):
                                moments=terms(m,wave(z[sup+'_'+tag]),truth).sum(0)
                                return metrics(moments)['skill']
                            current=atom_scores('current')
                            wrong=np.nanmean(np.stack([atom_scores(str(seed)) for seed in SEEDS]),axis=0)
                            per_feature[sup]={str(atom):dict(current=float(cs) if np.isfinite(cs) else None,
                                wrong=float(ws) if np.isfinite(ws) else None) for atom,cs,ws in zip(ids[k],current,wrong)}
                        result['per_atom_shape'][label]=per_feature
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    report={}
    for name in ('bolt','timesfm'):
        with np.load(a.out/(name+'.npz')) as f:report[name]=cohort(dict(f))
    (a.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
