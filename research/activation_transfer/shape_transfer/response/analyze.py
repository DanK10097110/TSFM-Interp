"""Score the same actual edits against the user's request and the actual source response."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def moments(estimate,reference,weights,center):
    """Paired sufficient statistics; recompute the series template within each draw."""
    p,r=np.asarray(estimate,dtype=float),np.asarray(reference,dtype=float)
    n=weights.sum(axis=1)[:,None]
    xx=weights@np.sum(p*p,axis=-1); yy=weights@np.sum(r*r,axis=-1); xy=weights@np.sum(p*r,axis=-1)
    if center:
        ps=(weights@p.reshape(len(p),-1)).reshape(-1,*p.shape[1:])
        rs=(weights@r.reshape(len(r),-1)).reshape(-1,*r.shape[1:])
        xx=np.maximum(0,xx-np.sum(ps*ps,axis=-1)/n)
        yy=np.maximum(0,yy-np.sum(rs*rs,axis=-1)/n)
        xy=xy-np.sum(ps*rs,axis=-1)/n
        # Exact cross-item constants have zero centered effect and a defined zero cosine.
        const=np.ptp(p,axis=0).max(axis=-1)==0
        xx[:,const]=0.; xy[:,const]=0.
    if (yy<1e-16*n).any(): raise ValueError('Reference energy at numerical floor')
    return {'skill':(2*xy-xx)/yy,'gain':xy/yy,
            'cosine':np.divide(xy,np.sqrt(xx*yy),out=np.zeros_like(xy),where=xx*yy>1e-25),
            'reference_rms':np.sqrt(yy/n/p.shape[-1]),'estimate_rms':np.sqrt(xx/n/p.shape[-1])}


def interval(values):
    return dict(value=float(values[0]),ci95=np.quantile(values[1:],[.025,.975]).tolist())


def summarize():
    z={}
    for name in ('bolt','timesfm'):
        with np.load(HERE/'results'/(name+'_effects.npz')) as f:z[name]=dict(f)
    fields=('arms','waves','goals','ids')
    for k in fields:np.testing.assert_array_equal(z['bolt'][k],z['timesfm'][k])
    arms=z['bolt']['arms'].tolist(); waves=z['bolt']['waves'].tolist(); out={}
    for target,archive in z.items():
        source='timesfm' if target=='bolt' else 'bolt';out[target]={}
        for ds in ('daily','weekly'):
            use=np.char.startswith(archive['ids'],ds+'/'); n=int(use.sum()); counts=np.random.default_rng(2026112064).multinomial(n,np.full(n,1/n),size=4000)
            weights=np.vstack([np.ones(n),counts]); group={};out[target][ds]=group
            native=z[source]['effects'][use,:,arms.index('native')].astype(float)
            true=np.broadcast_to(archive['goals'],native.shape)
            est={a:archive['effects'][use,:,i].astype(float) for i,a in enumerate(arms)}
            est['source_native']=native;est['forecast_template']=np.broadcast_to(archive['donor_free_source'],native.shape)
            for mode in ('requested','source','source_centered_by_series','source_horizon_shape'):
                ref=true if mode=='requested' else native
                centered=mode=='source_centered_by_series'; horizon=mode=='source_horizon_shape'
                if horizon: ref=ref-ref.mean(-1,keepdims=True)
                stats={a:moments(pred-pred.mean(-1,keepdims=True) if horizon else pred,ref,weights,centered) for a,pred in est.items()}
                overall={}; comparisons={}
                for a,m in stats.items():
                    overall[a]={w:{k:interval(v[:,j]) for k,v in m.items()} for j,w in enumerate(waves)}
                    overall[a]['novel_macro']={k:interval(v[:,3:].mean(axis=1)) for k,v in m.items()}
                for other in ('clean_matched','clean_capped','mean_matched','mean_capped','shuffled_matched','shuffled_capped','native','forecast_template'):
                    delta=stats['response']['skill']-stats[other]['skill']
                    comps={w:interval(delta[:,j]) for j,w in enumerate(waves)}
                    avg=delta[:,3:].mean(axis=1);comps['novel_macro']=interval(avg)
                    if mode=='source' and other=='clean_matched':
                        comps['novel_macro']['lower_one_sided_98_75']=float(np.quantile(avg[1:],.0125))
                    comparisons[other]=comps
                group[mode]=dict(methods=overall,response_minus=comparisons)
    return dict(waves=waves,source_for_recipient={'bolt':'timesfm','timesfm':'bolt'},domains=out)


if __name__=='__main__':
    dest=HERE/'results'/'summary.json';dest.write_text(json.dumps(summarize(),indent=2,allow_nan=False)+'\n')
