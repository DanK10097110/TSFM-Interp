"""Collect generic donor median Jacobians on all 1,024 historical training items."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.replication.data import CACHE as OLD
from research.activation_transfer.shape_transfer.full_space.audit import hook

HERE=Path(__file__).resolve().parent
CACHE=Path("/tmp/tsfm_state_predictor_20261121")


def collect(name,batch=8):
    with np.load(OLD/"data.npz") as f:
        train=f["split"]=="train";x=f["x"][train];ids=f["ids"][train]
    assert len(ids)==1024 and len(set(ids))==1024
    enc=adapter_for(name)
    for p in enc.module.parameters():p.requires_grad_(False)
    width=512 if name=="bolt" else 1280
    jac=[];clean=[]
    for i in range(0,len(x),batch):
        obs=x[i:i+batch]
        d=torch.zeros((len(obs),width),requires_grad=True)
        with torch.enable_grad(),hook(enc,name,d):
            y=enc.predict_differentiable(obs,8)
            grad=[torch.autograd.grad(y[:,h].sum(),d,retain_graph=h<7)[0] for h in range(8)]
        jac.append(torch.stack(grad,dim=1).detach().numpy())
        clean.extend(y.detach().numpy())
    jac=np.concatenate(jac)
    assert np.isfinite(jac).all()
    idx=np.random.default_rng(20261121082).choice(len(ids),16,replace=False)
    with np.load(OLD/(name+".npz")) as f:
        np.testing.assert_array_equal(f["ids"][train],ids)
        radius=np.median(np.linalg.norm(f["tokens"][train].mean(1),axis=1))
    u=np.random.default_rng(20261121083).normal(size=(len(idx),width)).astype("f4")
    u*= (.002*radius/np.linalg.norm(u,axis=1))[:,None]
    ys=[]
    for sign in (1.,-1.):
        with hook(enc,name,torch.from_numpy(sign*u)):ys.append(enc.predict(x[idx],8,[.5])["quantiles"][:,:,0])
    observed=(ys[0]-ys[1])/2
    expected=np.einsum("nhd,nd->nh",jac[idx],u)
    rel=np.linalg.norm(expected-observed)/np.linalg.norm(observed)
    if rel>.08:raise ValueError(f"Production finite-difference verification failed: {rel}")
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE/(name+"_train.npz"),ids=ids,jac=jac,fd_ids=ids[idx],fd_relative=rel,
                        radius=radius,clean=np.array(clean))
    print(name,jac.shape,"finite difference",rel,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("name",choices=MODELS)
    a=parser.parse_args();torch.set_num_threads(1);collect(a.name)
