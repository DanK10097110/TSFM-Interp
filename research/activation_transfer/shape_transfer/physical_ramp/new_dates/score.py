"""Crossed station/date inference for the native isolated ramp responses in Celsius."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.run import ARMS, OUT

def energy(x, station_weights, date_weights, remove_date_mean=False):
    """Weighted sum of squared entries for every bootstrap draw."""
    stations,dates,leads=x.shape
    square=np.square(x).sum(axis=2)
    total=((station_weights@square)*date_weights).sum(axis=1)
    if remove_date_mean:
        sums=(station_weights@x.reshape(stations,dates*leads)).reshape(-1,dates,leads)
        fitted=np.square(sums).sum(axis=2)/station_weights.sum(axis=1)[:,None]
        total-=(fitted*date_weights).sum(axis=1)
    return np.maximum(total,0.)

def weights(n,seed):
    rng=np.random.default_rng(seed)
    return np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=3000)])

def report(values):
    return dict(value=float(values[0][0]),**{
        label:np.quantile(v[1:],[.025,.975]).tolist()
        for label,v in zip(('crossed95','station95','date95'),values)
    })

def main(out):
    f=dict(np.load(out/'transfer.npz'))
    ns,nd=35,11
    np.testing.assert_equal(f['station'],np.repeat(np.arange(ns),nd))
    np.testing.assert_equal(f['origin'],np.tile(np.arange(nd),ns))
    sw,dw=weights(ns,26112319),weights(nd,26112320)
    sampling=((sw,dw),(sw,np.ones_like(dw)),(np.ones_like(sw),dw))
    res=dict(stations=ns,dates=f['dates'].tolist(),stations_list=f['stations'].tolist(),n=385,
             history_sha=str(f['raw_input_sha']),map_sha=str(f['map_sha']),geometry={},reference={},metrics={},per_date={})
    native=f['hidden_truth'];nt=np.linalg.norm(native,axis=1)
    for arm in ARMS[:-1]:
        v=f[arm+'_vec'];nv=np.linalg.norm(v,axis=1)
        res['geometry'][arm]=dict(skill=float(1-np.square(v-native).sum()/np.square(native).sum()),
            median_cos=float(np.median(np.sum(v*native,axis=1)/(nv*nt))),norm_native_ratio=float(np.median(nv/nt)),
            max_budget_error=float(abs(nv-f['budget']).max()))
    res['zero_budget']=int((f['budget']==0).sum())
    for seg,sl in (('early',slice(8)),('late',slice(8,16)),('h16',slice(16))):
        for center in (True,False):
            name=seg+('_centered' if center else '_full')
            def arr(key):
                a=f[key][:,sl].reshape(ns,nd,-1).astype(float)
                return a-a.mean(axis=2,keepdims=True) if center else a
            target=arr('truth');whole=arr('whole')
            centered=target-target.mean(0,keepdims=True)
            res['reference'][name]=dict(isolated_celsius_rms=float(np.sqrt(np.mean(target**2))),
                whole_input_celsius_rms=float(np.sqrt(np.mean(whole**2))),
                between_station_fraction=float(np.square(centered).sum()/np.square(target).sum()))
            for remove in (False,True):
                stem=name+('_between' if remove else '')
                den=[energy(target,s,d,remove) for s,d in sampling];fits={};stats={}
                for arm in ARMS:
                    v=arr(arm);err=target-v
                    fits[arm]=[1-energy(err,s,d,remove)/z for (s,d),z in zip(sampling,den)]
                    stats[arm]=report(fits[arm])
                    vf=v-v.mean(0) if remove else v;tf=centered if remove else target
                    stats[arm]['raw_celsius_rms']=float(np.sqrt(np.mean(vf**2)))
                    stats[arm]['pooled_cosine']=float(np.sum(vf*tf)/(np.linalg.norm(vf)*np.linalg.norm(tf)))
                for arm in ARMS[1:]:
                    stats['mapped_minus_'+arm]=report([a-b for a,b in zip(fits['mapped'],fits[arm])])
                res['metrics'][stem]=stats
            if center:
                den=np.square(target).sum((0,2));by={k:(1-np.square(arr(k)-target).sum((0,2))/den) for k in ARMS}
                res['per_date'][name]={k:v.tolist() for k,v in by.items()}
                res['per_date'][name]['mapped_minus_wrong_step']=(by['mapped']-by['wrong_step']).tolist()
    (out/'scores.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);main(p.parse_args().out)
