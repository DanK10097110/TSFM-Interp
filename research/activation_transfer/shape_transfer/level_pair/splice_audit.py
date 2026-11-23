"""Replay hybrid inputs using independent blockwise regressions and singleton checkpoint encodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.shape_transfer.level_pair.inputs import HERE
from research.activation_transfer.shape_transfer.level_pair.splice import OFFSETS,hybrid_grid
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.sae.train import load_sae_checkpoint


def independent_hybrids(x,daily,traffic):
    x=np.asarray(x,float);mu=x.mean(1);sd=x.std(1);m=x[:,-32:].shape[1];n=x.shape[1]
    # Regress the two disjoint time blocks separately, using no global basis.
    # The projected prefix has one constant and the last 32 have three constraints.
    left=x[:,:-m]-x[:,:-m].mean(1)[:,None]
    t=np.arange(m,dtype=float)-(m-1)/2
    regressors=np.column_stack([np.ones(m),np.eye(m)[:,-1],t])
    right=x[:,-m:]-(regressors@np.linalg.lstsq(regressors,x[:,-m:].T,rcond=None)[0]).T
    energy1=(left**2).sum(1);energy2=(right**2).sum(1)
    left/=np.sqrt(energy1)[:,None];right/=np.sqrt(energy2)[:,None]
    f=energy2/(energy1+energy2)
    target_left=-1.25*m/(n-m)
    constraints=np.vstack([np.ones(m)/m,np.eye(m)[:,-1],t*(m-1)/(t@t)])
    target_right=np.linalg.lstsq(constraints,[1.25,.5,0.],rcond=None)[0]
    budget=n-target_left**2*(n-m)-target_right@target_right
    waves={}
    for bits in range(8):
        a,b,c=np.where([bits&1,bits&2,bits&4],1,0)
        sources=[traffic,daily];ia,ib,ic=sources[a],sources[b],sources[c]
        early=target_left+left[ia]*np.sqrt(budget*(1-f[ic]))[:,None]
        recent=target_right+right[ib]*np.sqrt(budget*f[ic])[:,None]
        waves[bits]=(np.hstack([early,recent])*sd[ia,None]+mu[ia,None]).astype('f4')
    return waves


def replay(model,out,inp):
    saved=dict(np.load(out/(model+'_splice.npz')));primary=dict(np.load(out/(model+'_effects.npz')))
    enc=adapter_for(model);layer=MODELS[model][-1];sae=load_sae_checkpoint(str(dictionary_path(model)))
    idx=int(primary['atoms'][0]);record=[]
    for off in OFFSETS:
        d,t=saved['daily'],saved['traffic_'+str(off)]
        alternate=independent_hybrids(inp['inputs'],d,t)
        original,_,_=hybrid_grid(inp['inputs'],d,t)
        for k,w in alternate.items():
            drift=float(abs(w-original[k]).max())
            np.testing.assert_allclose(w,original[k],atol=2e-6)
            # Deterministic index covers different paired histories across all eight cells.
            i=(29*k+off)%len(t);x=w[i:i+1]
            with torch.inference_mode():
                hidden=capture_raw_tokens(enc,x,[layer])[layer][0,primary['positions']].mean(0)
                pre=((hidden-sae.b_dec)@sae.W_enc+sae.b_enc).numpy()
            competitors=np.delete(pre,idx);cutoff=max(0.,float(np.sort(competitors)[-sae.k]))
            margin=float(pre[idx])-cutoff
            actual=dict(pre=float(pre[idx]),z=float(pre[idx]) if margin>0 else 0.,margin=margin,
                        cutoff=cutoff,rank=int(1+np.count_nonzero(competitors>pre[idx])))
            diff={}
            for label,v in actual.items():
                expected=float(saved[f'{off}_{k}_{label}'][i])
                diff[label]=float(abs(expected-v))
                if label=='rank':
                    assert diff[label]<=np.count_nonzero(abs(competitors-pre[idx])<.00022)
                else:np.testing.assert_allclose(expected,v,atol=.00022,rtol=.0001)
            record.append(dict(offset=off,cell=k,pair_row=i,grid_input_max_abs=drift,errors=diff))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args();torch.set_num_threads(1)
    f=dict(np.load(a.out/'inputs.npz'))
    r={m:replay(m,a.out,f) for m in MODELS}
    (a.out/'splice_replay.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
