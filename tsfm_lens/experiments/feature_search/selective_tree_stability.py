"""Separate adaptive-tree instability from native hidden-state sensitivity.

A hard split threshold or swapped peak order can make an adapter look jagged even
when the native network is locally smooth. Match tones before comparing trees.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments.feature_search.selective_tree import decompose
from experiments.feature_search.coherence_study import read,write


def run(out):
    """Perturb past histories only, using exactly the native sensitivity directions."""
    x=np.load(out/'inputs.npz')['history'];rows=read(out/'rows.json');result=[]
    for i,history in enumerate(x):
        levels,meta=decompose(history);base=levels[-1];rng=np.random.default_rng(4944000+i)
        for direction,value in [('sinusoid',np.sin(2*np.pi*5.35*np.arange(-128,0)/128+.3)),('noise',rng.normal(size=128))]:
            value=value/np.sqrt(np.mean(value**2))*history.std()
            for epsilon in [.001,.01,.1]:
                for sign in [-1,1]:
                    pert,detail=decompose(history+sign*epsilon*value);other=pert[-1]
                    n=max(len(base)-1,len(other)-1);a=np.zeros((n,128));b=a.copy()
                    if len(base)>1:a[:len(base)-1]=base[:-1]
                    if len(other)>1:b[:len(other)-1]=other[:-1]
                    if n:
                        cost=np.mean((a[:,None]-b[None])**2,axis=-1);left,right=linear_sum_assignment(cost);change=float(cost[left,right].sum())
                    else:change=0.
                    residual_change=float(np.mean((base[-1]-other[-1])**2));scale=max(float(np.var(history)),1e-12)
                    result.append(dict(row=i,source=rows[i]['source'],domain=rows[i]['domain'],case=rows[i]['case'],split=rows[i]['split'],direction=direction,epsilon=epsilon,sign=sign,base_depth=len(meta),perturbed_depth=len(detail),depth_changed=len(meta)!=len(detail),matched_partition_rms=float(np.sqrt((change+residual_change)/scale)),input_rms=epsilon))
    write(out/'tree_stability.json',result);print('tree perturbations',len(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
