"""Construct isolated, symmetric edits at fixed history mean, variance and two other statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE, HERE as BLIND

HERE = Path(__file__).resolve().parent
CUES = ('recent','last','slope','translation')
DOSES = np.array([.25,.5,.5,.25])
SIGNS = np.array([-1.,1.])


def statistic_rows(length=480, window=32):
    rows = np.zeros((3,length),dtype=float)
    rows[0] = -1/length; rows[0,-window:] += 1/window
    rows[1] = -1/length; rows[1,-1] += 1
    t = np.arange(window,dtype=float) - (window-1)/2
    rows[2,-window:] = (window-1)*t / np.dot(t,t)
    return rows


def isolated_histories(inputs):
    x = np.asarray(inputs,dtype=float)
    mean = x.mean(axis=1,keepdims=True)
    y = x-mean
    sd = np.sqrt(np.mean(y*y,axis=1))
    rows = statistic_rows(x.shape[1]); dual = np.linalg.solve(rows@rows.T,rows)
    stats = y @ rows.T
    projection = stats @ dual
    residual = y-projection
    remainder = (residual*residual).sum(axis=1)
    assert np.all(remainder>1e-13)
    edits = np.empty((len(x),4,2,x.shape[1]),dtype='f4')
    doses = sd[:,None]*DOSES[None]
    scales = np.ones_like(doses)
    for idx in range(3):
        increment = doses[:,idx,None]*dual[idx]
        quadratic = (increment*increment).sum(axis=1)
        linear = np.abs((projection*increment).sum(axis=1))
        # Largest common factor for either sign, solving a*f^2 + 2*|b|*f <= residual energy.
        max_factor = remainder/(np.sqrt(linear*linear+quadratic*remainder)+linear)
        scales[:,idx] = np.minimum(1.,max_factor*(1-1e-12))
        direction = increment*scales[:,idx,None]
        for j,sign in enumerate(SIGNS):
            moved = projection+sign*direction
            new_energy = np.sum(y*y,axis=1)-np.sum(moved*moved,axis=1)
            assert np.min(new_energy)>-1e-8
            edited = mean+moved+residual*np.sqrt(np.maximum(new_energy,0)/remainder)[:,None]
            edits[:,idx,j] = edited.astype('f4')
    for j,sign in enumerate(SIGNS):edits[:,3,j]=(x+sign*doses[:,3,None]).astype('f4')
    achieved=edits.astype(float)@rows.T
    diff=achieved-stats[:,None,None,:]
    goal=np.zeros_like(diff)
    for k in range(3):goal[:,k,:,k]=(doses[:,k]*scales[:,k])[:,None]*SIGNS
    np.testing.assert_allclose(diff,goal,atol=5e-6,rtol=1e-6)
    achieved_mean=edits.astype(float).mean(-1)
    achieved_sd=edits.astype(float).std(-1)
    desired_mean=np.broadcast_to(mean[:,:,None],achieved_mean.shape).copy()
    desired_mean[:,3]+=doses[:,3,None]*SIGNS
    np.testing.assert_allclose(achieved_mean,desired_mean,atol=2e-6)
    np.testing.assert_allclose(achieved_sd,np.broadcast_to(sd[:,None,None],achieved_sd.shape),atol=2e-6)
    return dict(inputs=x.astype('f4'),edits=edits,clean_stats=stats,edited_stats=achieved,
                doses=doses,scales=scales,mean=mean[:,0],sd=sd,edited_mean=achieved_mean,edited_sd=achieved_sd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.cache/'data.npz') as f:
        x,ids,split=f['x'],f['ids'],f['split']
    meta=json.loads((BLIND/'results/manifest.json').read_text());assert hashlib.sha256(x.tobytes()).hexdigest()==meta['input_sha256']
    assert len(x)==256 and len(set(ids))==256 and x.shape[1]==480
    np.savez_compressed(a.out/'inputs.npz',ids=ids,split=split,cues=np.array(CUES),signs=SIGNS,**isolated_histories(x))
