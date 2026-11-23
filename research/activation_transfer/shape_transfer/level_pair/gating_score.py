"""Audit cue overlap, match series on observables, and score actual controlled-domain firing."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.level_pair.gating import CONDITIONS,HERE

FIELDS=('on','z','pre','cutoff','margin','rank')


def values(z,condition):
    return np.column_stack([z[condition+'_z']>0,*[z[condition+'_'+key] for key in FIELDS[1:]]])


def intervals(point,boot):
    return {k:dict(value=float(v),ci95=np.percentile(boot[:,i],[2.5,97.5]).tolist()) for i,(k,v) in enumerate(zip(FIELDS,point))}


def matching(stats,calipers):
    daily,traffic=stats
    components=np.abs(traffic[:,None,:len(calipers)]-daily[None,:,:len(calipers)])
    scaled=components/np.asarray(calipers)
    return np.where(np.max(scaled,axis=2)<=1.,np.sum(scaled**2,axis=2),np.inf)


def find_pairs(distance,traffic,daily):
    reduced=distance[np.ix_(traffic,daily)];nearest=np.argmin(reduced,axis=1)
    keep=np.isfinite(reduced[np.arange(len(traffic)),nearest])
    return traffic[keep],daily[nearest[keep]]


def analyze(out):
    outdata={};rng=np.random.default_rng(2026112186)
    n=128;ds=rng.integers(0,n,size=(2000,n));ts=rng.integers(0,n,size=(2000,n))
    for m in ('bolt','timesfm'):
        with np.load(out/(m+'_gating.npz')) as f,np.load(out/(m+'_effects.npz')) as real:
            z=dict(f);np.testing.assert_array_equal(z['ids'],real['ids'])
            np.testing.assert_allclose(z['clean_z'],real['clean_z'],atol=2e-4)
            for j,key in enumerate(('minus','plus')):
                np.testing.assert_allclose(z[key+'_z'],real['edited_z'][:,0,j],atol=2e-4)
                np.testing.assert_allclose(z[key+'_pre'],real['edited_pre'][:,0,j],atol=2e-4)
        day=z['split']=='daily';traffic=z['split']=='monash_traffic';assert day.sum()==traffic.sum()==n
        stats=[z['native_stats'][mask] for mask in (day,traffic)];ids=[z['ids'][mask] for mask in (day,traffic)]
        outdata[m]=result=dict(k=int(z['k']),width=int(z['width']),support={},natural={},matched={},controlled={})
        for bounds,q in [('range',(0,1)),('central90',(.05,.95))]:
            lim=np.stack([np.quantile(s[:,0],q) for s in stats]);lo=float(lim[:,0].max());hi=float(lim[:,1].min());support={}
            for name,mask in [('daily',day),('traffic',traffic)]:
                ok=mask&(z['native_stats'][:,0]>=lo)&(z['native_stats'][:,0]<=hi)
                support[name]={'n':int(ok.sum()),'active':int((z['clean_z'][ok]>0).sum()),'domain_bounds':lim[0 if name=='daily' else 1].tolist()}
            result['support'][bounds]=dict(bounds=[lo,hi],domains=support)
        for condition in CONDITIONS:
            v=values(z,condition);summaries={};result['natural' if condition in ('clean','minus','plus') else 'controlled'][condition]=summaries
            selected=[]
            for name,mask in [('daily',day),('traffic',traffic)]:
                ok=z[condition+'_eligible']&mask;w=v[ok];selected.append(w)
                summaries[name]=dict(n=len(w),fires=int(w[:,0].sum()),
                    mean={f:float(k) for f,k in zip(FIELDS,w.mean(0))},
                    median={f:float(k) for f,k in zip(FIELDS,np.median(w,axis=0))},
                    margin_p10_p90=np.quantile(w[:,4],[.1,.9]).tolist(),
                    positive_pre=int((w[:,2]>0).sum()),rank_p10_p90=np.quantile(w[:,5],[.1,.9]).tolist())
            if len(selected[0])==len(selected[1])==128:
                a,b=selected;summaries['daily_minus_traffic']=intervals(a.mean(0)-b.mean(0),a[ds].mean(1)-b[ts].mean(1))
        for mode in ('native','joint'):
            usable=z[mode+'_low_eligible']&z[mode+'_high_eligible'];lo=values(z,mode+'_low');hi=values(z,mode+'_high')
            effect=hi-lo;report={};result['controlled'][mode+'_high_minus_low']=report
            for name,mask,boot in [('daily',day,ds),('traffic',traffic,ts)]:
                delta=effect[mask&usable]
                if len(delta)==n:report[name]=intervals(delta.mean(0),delta[boot].mean(1))
        good=z['clean_z']>0
        for mode,cal in [('recent',[.10]),('joint',[.20,.35,.35])]:
            dist=matching(stats,cal);aa,bb=find_pairs(dist,np.arange(n),np.arange(n));boots=[]
            a=values(z,'clean')[day];b=values(z,'clean')[traffic];extras={}
            for d,t in zip(ds,ts):
                i,j=find_pairs(dist,t,d)
                if len(i):boots.append((a[j]-b[i]).mean(0))
            if len(aa):
                gap=abs(stats[0][bb]-stats[1][aa]);extras=dict(daily_minus_traffic=intervals((a[bb]-b[aa]).mean(0),np.stack(boots)),
                    daily_means=dict(zip(FIELDS,a[bb].mean(0).tolist())),traffic_means=dict(zip(FIELDS,b[aa].mean(0).tolist())),
                    gaps_mean=gap.mean(0).tolist(),gaps_max=gap.max(0).tolist(),pair_ids=list(zip(ids[1][aa].tolist(),ids[0][bb].tolist())))
            result['matched'][mode]=dict(eligible_traffic=len(aa),unique_daily=len(np.unique(bb)),draws_with_matches=len(boots),**extras)
    return outdata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    (a.out/'gating.json').write_text(json.dumps(analyze(a.out),indent=2,allow_nan=False)+'\n')
