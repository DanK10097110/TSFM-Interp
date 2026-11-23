"""Three native-quantile derivatives, checked against the ordinary forecasting path."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.depth.geometry import FRESH,histories
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as OLD_JAC
from tsfm_lens.extraction.hooks import token_patch

HERE=Path(__file__).resolve().parent
SCRATCH=Path("/tmp/tsfm_quantiles_20261121")
LEVELS=[.1,.5,.9]


def components(values):
    """On the trailing quantile axis, preserving all leading axes."""
    low,med,high=np.moveaxis(values,-1,0)
    return np.stack([med,(high-low)/2,(high+low)/2-med],axis=-1)


def differential(adapter,name,x,levels=LEVELS):
    """Use the native decoder's distribution before detaching it for the public predict API."""
    if name=="bolt":
        tensor=adapter.pipeline.model(context=adapter.prepare(x).to(adapter.device)).quantile_preds
        return tensor[:,[adapter.pipeline.quantiles.index(q) for q in levels],:8].transpose(1,2)
    raw=[]
    decode=adapter.tfm.model.decode
    def observe(*args,**kwargs):
        outputs=decode(*args,**kwargs)
        raw.append(outputs[0])
        return outputs
    with patch.object(adapter.tfm.model,"decode",new=observe):
        adapter.predict_differentiable(x,8)
    if len(raw)!=1:
        raise ValueError("Expected one TimesFM decode")
    return raw[0][:,-1,:8,[1+adapter._native_quantiles.index(q) for q in levels]]


def ordinary_or_patch(adapter,layer,x,tokens,delta,selector,batch):
    outputs=[]
    for i in range(0,len(x),batch):
        obs=x[i:i+batch]
        if delta is None:
            y=adapter.predict(obs,8,LEVELS)["quantiles"]
        else:
            moved=torch.from_numpy(tokens[i:i+batch]+delta[i:i+batch,None,:])
            with token_patch(adapter.module,layer,selector,moved):
                y=adapter.predict(obs,8,LEVELS)["quantiles"]
        outputs.append(y.reshape(len(obs),24))
    return np.concatenate(outputs)


def material(name):
    ids,x,labels=histories()
    h,pos,old_ids=read_materials(CACHE,name)
    fresh,other,new_ids=read_materials(FRESH,name)
    np.testing.assert_array_equal(pos,other)
    np.testing.assert_array_equal(ids,np.r_[old_ids,new_ids])
    tokens=np.concatenate([h,fresh])
    return ids,x,labels,tokens,pos


def run(name,scratch,batch):
    ids,x,labels,tokens,pos=material(name)
    train=labels=="train"
    radius=float(np.median(np.linalg.norm(tokens[train].mean(axis=1),axis=1)))
    ids,x,labels,tokens=ids[~train],x[~train],labels[~train],tokens[~train]
    enc=adapter_for(name)
    for p in enc.module.parameters():
        p.requires_grad_(False)
    layer=MODELS[name][-1]
    select=last_window_slice(enc,pos)
    jac,base=[],[]
    for i in range(0,len(x),batch):
        obs=x[i:i+batch]
        h=torch.from_numpy(tokens[i:i+batch])
        d=torch.zeros_like(h[:,0],requires_grad=True)
        with torch.enable_grad(),token_patch(enc.module,layer,select,h+d[:,None]):
            y=differential(enc,name,obs)
            flat=y.reshape(len(obs),24)
            grad=[torch.autograd.grad(flat[:,k].sum(),d,retain_graph=k<23)[0] for k in range(24)]
        ordinary=enc.predict(obs,8,LEVELS)["quantiles"]
        np.testing.assert_allclose(y.detach().numpy(),ordinary,atol=1e-4,rtol=3e-5)
        jac.append(torch.stack(grad,dim=1).reshape(len(obs),8,3,-1).numpy())
        base.append(ordinary)
    j,base=np.concatenate(jac),np.concatenate(base)
    with np.load(OLD_JAC/(name+"_jac.npz")) as old,np.load(FRESH/(name+"_geometry.npz")) as f:
        np.testing.assert_array_equal(ids,np.r_[old["ids"],f["ids"]])
        med=np.concatenate([old["jac"],f["jac"]])
        error=np.linalg.norm(j[:,:,1,:]-med)/np.linalg.norm(med)
        if error>.0003:
            raise ValueError(f"Median derivative replay error {error}")
    idx=np.concatenate([np.flatnonzero(labels==lab)[:4] for lab in np.unique(labels)])
    u=np.random.default_rng(1293).normal(size=(len(idx),j.shape[-1])).astype("f4")
    u*=(.002*radius/np.linalg.norm(u,axis=1))[:,None]
    hi,lo=[ordinary_or_patch(enc,layer,x[idx],tokens[idx],s*u,select,batch).reshape(len(idx),8,3) for s in (1,-1)]
    observed=components((hi-lo)/2)
    predicted=components(np.einsum("nhqd,nd->nhq",j[idx],u))
    fd=np.linalg.norm(observed-predicted,axis=(0,1))/np.linalg.norm(observed,axis=(0,1))
    if not np.isfinite(fd).all() or fd.max()>.08:
        raise ValueError(f"Native distribution derivative check failed: {fd}")
    scratch.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(scratch/(name+".npz"),ids=ids,split=labels,jac=j,base=base,radius=radius,
                        median_replay=error,fd_error=fd)
    print(name,error,fd,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--batch",type=int,default=8)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.model,a.scratch,a.batch)
