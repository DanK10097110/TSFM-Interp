"""Quantify the level and horizon-shape contributions to request error separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from research.activation_transfer.forecast_requests.run import CUES
from research.activation_transfer.shape_transfer.level_geometry.run import HERE,cos


def item_metrics(effect,wanted,native):
    effect=np.asarray(effect,dtype=np.float64);wanted=np.asarray(wanted,dtype=np.float64)
    native=np.asarray(native,dtype=np.float64)
    n=len(effect);y=effect-effect.mean(-1,keepdims=True);w=wanted-wanted.mean();need=np.mean(wanted**2)
    center_native=native-native.mean(-1,keepdims=True)
    dc=np.square((effect-wanted).mean(-1))/need
    shape=np.square(y-w).mean(-1)/need
    result={'skill':1.-dc-shape,'dc_error':dc,'shape_error':shape,
            'signed_cosine':cos(effect,np.broadcast_to(wanted,(n,len(wanted)))),
            'signed_gain':np.mean(effect*wanted,axis=-1)/need,
            'mean_shift':effect.mean(-1),'absolute_mean_shift':np.abs(effect.mean(-1)),
            'effect_rms':np.sqrt(np.square(effect).mean(-1)),
            'shape_cosine_to_native':cos(y,center_native)}
    if np.linalg.norm(w)>1e-12:
        result['shape_cosine_to_request']=cos(y,np.broadcast_to(w,(n,len(w))))
    return result


def interval(v,draws):
    return {'mean':float(v.mean()),'ci95':np.quantile(v[draws].mean(1),[.025,.975]).tolist()}


def difference(a,b):
    return {k:a[k]-b[k] for k in a}

def exact_zero_writes(effects,scalars):
    blank=scalars==0
    floor=float(np.max(np.abs(effects[:,blank]),initial=0.))
    if floor>3e-4:
        raise ValueError('Zero-dose live patch has an unexpectedly large numerical residual')
    return np.where(blank[None,:,None],0.,effects),floor


def summarize(model,root):
    with np.load(root/(model+'_effects.npz')) as z:
        ids=z['ids'];labels=z['labels'].tolist();requests=z['desired'];doses=z['cal_scalar'];index=z['cue'];modes={k:z[k] for k in ['equal_budget','calibrated']}
    modes['calibrated'],floor=exact_zero_writes(modes['calibrated'],doses)
    report={'zero_dose_cal_replay_max':floor,'zero_dose_labels':[l for l,s in zip(labels,doses) if s==0]}
    for domain in ['daily','weekly']:
        use=np.char.startswith(ids,domain+'/');n=int(use.sum());assert n==128
        draws=np.random.default_rng(2026112031).integers(n,size=(4_000,n));items={}
        for mode,effects in modes.items():
            result={};per={};shape_contrasts=[]
            for c,cue in enumerate(CUES):
                i0=labels.index(cue+'/native');native=modes['equal_budget'][use,i0].astype(float)
                raw={}
                for i in np.flatnonzero(index==c):
                    label=labels[i].split('/')[-1]
                    raw[label]=item_metrics(effects[use,i],requests[i],native)
                if c==0:
                    random=[raw.pop('random'+str(i)) for i in range(8)]
                    raw['random_mean']={k:np.mean([v[k] for v in random],axis=0) for k in random[0]}
                    raw['mapped_minus_random']=difference(raw['mapped'],raw['random_mean'])
                    result['random_lift_skill_across_directions']=[float(v['skill'].mean()) for v in random]
                else:
                    raw['corrected_minus_mapped']=difference(raw['corrected'],raw['mapped'])
                    shape_contrasts.append(raw['corrected_minus_mapped'])
                result[cue]={arm:{k:interval(v,draws) for k,v in vec.items()} for arm,vec in raw.items()}
            result['mean_shape_corrected_minus_mapped']={k:interval((shape_contrasts[0][k]+shape_contrasts[1][k])*.5,draws) for k in shape_contrasts[0]}
            items[mode]=result
        report[domain]=items
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=HERE/'results')
    a=p.parse_args();(a.results/'scores.json').write_text(json.dumps({m:summarize(m,a.results) for m in ['bolt','timesfm']},indent=2,allow_nan=False)+'\n')
