"""Short separately recoverable derivative measurements for the 480-observation weather panel."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.continuation.probe import SCRATCH as CAL
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.support_confirmation.data import adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import CACHE,OUT,verify_normalization
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.extraction.hooks import token_patch


def atomic(path,**arrays):
    tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(path)


def run(model,stage,cache,out):
    with np.load(cache/'data.npz') as f:x=f['x'];ids=f['ids']
    location=cache/(model+'.npz');site=MODELS[model][-1]
    enc=adapter_for_length(model,480)
    if stage=='tokens':
        spans=enc.token_time_spans()
        pos=np.flatnonzero((spans[:,0]>=448)&(spans[:,1]<=480))
        assert spans[pos[0],0]==448 and spans[pos[-1],1]==480
        np.testing.assert_array_equal(spans[pos[1:],0],spans[pos[:-1],1])
        if location.exists():
            h,saved_positions,saved_ids=read_materials(cache,model)
            np.testing.assert_array_equal(ids,saved_ids);np.testing.assert_array_equal(pos,saved_positions)
            actual=capture_raw_tokens(enc,x[:1],[site])[site][:,pos].numpy()
            np.testing.assert_allclose(actual,h[:1],atol=5e-5,rtol=1e-4)
            return
        states=np.concatenate([capture_raw_tokens(enc,x[i:i+8],[site])[site][:,pos].numpy() for i in range(0,len(x),8)])
        atomic(location,tokens=states,ids=ids,positions=pos)
        return
    h,pos,check=read_materials(cache,model);np.testing.assert_array_equal(ids,check);sel=last_window_slice(enc,pos)
    bucket=cache/'parts';bucket.mkdir(exist_ok=True)
    if stage!='merge':
        phase=int(stage);idx=np.array_split(np.arange(len(ids)),4)[phase]
        waves=[];gradient=[]
        for i in range(0,len(idx),4):
            ix=idx[i:i+4];state=torch.from_numpy(h[ix])
            delta=torch.zeros_like(state[:,0],requires_grad=True)
            with torch.enable_grad(),token_patch(enc.module,site,sel,state+delta[:,None]):
                y=enc.predict_differentiable(x[ix],8)
                g=[torch.autograd.grad(y[:,j].sum(),delta,retain_graph=j<7)[0] for j in range(8)]
            gradient.append(torch.stack(g,dim=1).detach().numpy())
            waves.append(forecast16(enc,site,x[ix],h[ix],None,sel,4))
        atomic(bucket/f'{model}_{phase}.npz',indices=idx,ids=ids[idx],jac=np.concatenate(gradient),forecast=np.concatenate(waves))
        print(model,phase,len(idx),flush=True)
        return
    chunks=[dict(np.load(bucket/f'{model}_{q}.npz')) for q in range(4)]
    np.testing.assert_array_equal(np.concatenate([f['indices'] for f in chunks]),np.arange(len(ids)))
    np.testing.assert_array_equal(np.concatenate([f['ids'] for f in chunks]),ids)
    j=np.concatenate([f['jac'] for f in chunks]);base=np.concatenate([f['forecast'] for f in chunks])
    with np.load(CAL/(model+'_jac16.npz')) as f:radius=float(f['radius'])
    idx=np.linspace(0,len(ids)-1,12,dtype=int)
    v=np.random.default_rng(20261122012).normal(size=(len(idx),h.shape[-1]))
    v=(.002*radius*v/np.linalg.norm(v,axis=1)[:,None]).astype('f4')
    plus,minus=[forecast16(enc,site,x[idx],h[idx],sgn*v,sel,4) for sgn in (1,-1)]
    fd=(plus[:,:8]-minus[:,:8])/2;pred=np.einsum('nhd,nd->nh',j[idx],v)
    error=np.linalg.norm(fd-pred)/np.linalg.norm(fd)
    assert np.isfinite(j).all() and error<.08,error
    atomic(cache/(model+'_early.npz'),ids=ids,jac=j,forecast=base,radius=radius,fd_rows=idx,fd_actual=fd,fd_linear=pred,fd_error=error)
    verify_normalization(model,cache,out)
    print(model,'finite difference',error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);p.add_argument('stage',choices=['tokens','0','1','2','3','merge'])
    p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();torch.set_num_threads(1);run(a.model,a.stage,a.cache,a.out)
