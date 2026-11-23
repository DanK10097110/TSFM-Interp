"""Decompose the follow-up's error into native dose, released waveform and signed gain."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.calibration_budget.study import HERE, FRESH_RESULTS, MODELS


def summarize(out):
    result={}
    for s in MODELS:
        with np.load(out/(s+'_single.npz')) as low,np.load(out/(s+'_fresh.npz')) as standard,np.load(FRESH_RESULTS/(s+'_native.npz')) as ref:
            np.testing.assert_array_equal(low['ids'],standard['ids']);np.testing.assert_array_equal(low['ids'],ref['ids'])
            result[s]=domains={}
            for domain in np.unique(low['split']):
                keep=low['split']==domain;active=low['codes'][keep]>0;f=active.sum(0)>=8;mask=active[:,f,None]
                y=ref['native'][keep][:,f].astype(float);y-=y.mean(-1,keepdims=True)
                yy=np.sum(y*y*mask,axis=(0,2));freq=active[:,f].sum(0);live=yy>1e-10*8*freq
                data={'ids':low['atoms'][f][live].tolist(),'reference_centered_rms':np.sqrt(yy[live]/(8*freq[live])).tolist()}
                for arm,archive in [('1x24',low),('2x24',low),('16x24',standard),('baseline',standard)]:
                    effect=archive[arm];effect=effect[None] if effect.ndim==3 else effect
                    clips=archive[arm+'_clip'];clips=clips[None] if clips.ndim==2 else clips
                    arr=[]
                    for idx,edits in enumerate(effect):
                        x=edits[keep][:,f].astype(float);x-=x.mean(-1,keepdims=True)
                        xx=(x*x*mask).sum((0,2));xy=(x*y*mask).sum((0,2))
                        gain=(xy/yy)[live];relative=(xx/yy)[live]
                        row=dict(gain=gain.tolist(),energy_ratio=relative.tolist(),
                            cosine=(xy/np.maximum(np.sqrt(xx*yy),1e-20))[live].tolist(),
                            skill=(2*gain-relative).tolist(),clips=clips[idx,keep][:,f].sum(0)[live].tolist())
                        arr.append(row)
                    data[arm]=arr
                domains[str(domain)]=data
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args()
    (a.out/'amplitude.json').write_text(json.dumps(summarize(a.out),indent=2,allow_nan=False)+'\n')
