"""Measure five receiving sites without interrupting the gradient path between blocks."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as OLD_JAC
from tsfm_lens.extraction.hooks import _primary, _rebuild

HERE=Path(__file__).resolve().parent
SCRATCH=Path("/tmp/tsfm_depth_20261121")


def histories():
    with np.load(CACHE/"data.npz") as a,np.load(FRESH/"data.npz") as b:
        ids=np.concatenate([a["ids"],b["ids"]])
        if len(ids)!=len(np.unique(ids)):
            raise ValueError("Overlapping histories")
        return ids,np.concatenate([a["x"],b["x"]]),np.concatenate([a["split"],np.char.add("extra_",b["split"])])


def extract(adapter,layers,positions,x,with_gradient):
    """Independent additive leaves allow upstream writes to propagate through later monitored sites."""
    select=last_window_slice(adapter,positions)
    states={}
    leaves={}
    def at(layer):
        def hook(module,args,output):
            if layer in states:
                raise ValueError("Site invoked more than once")
            h=_primary(output)
            interval=select(h.shape[1])
            part=h[:,interval,:]
            states[layer]=part.detach().cpu().numpy().copy()
            if with_gradient:
                d=torch.zeros_like(part[:,0],requires_grad=True)
                leaves[layer]=d
                new=h.clone()
                new[:,interval,:]=part+d[:,None,:]
                return _rebuild(output,new)
        return hook
    jac={}
    with ExitStack() as stack:
        for l in layers:
            stack.enter_context(adapter.module.get_submodule(l).register_forward_hook(at(l)))
        if with_gradient:
            with torch.enable_grad():
                pred=adapter.predict_differentiable(x,8)
                values=[torch.autograd.grad(pred[:,j].sum(),tuple(leaves[l] for l in layers),retain_graph=j<7)
                        for j in range(8)]
            jac={l:torch.stack([v[i] for v in values],dim=1).numpy() for i,l in enumerate(layers)}
            point=pred.detach().numpy()
        else:
            point=adapter.predict(x,8,[.5])["quantiles"][...,0]
    if set(states)!=set(layers):
        raise ValueError("Missing block")
    return states,jac,point


def collect(name,scratch,batch):
    ids,x,split=histories()
    adapter=adapter_for(name)
    for p in adapter.module.parameters():
        p.requires_grad_(False)
    all_layers=adapter.all_layer_names()
    n=len(all_layers)
    depths=[0,n//4,n//2,3*n//4,n-1]
    layers=[all_layers[i] for i in depths]
    if layers[2]!=MODELS[name][-1]:
        raise ValueError("Middle site differs from source study")
    spans=adapter.token_time_spans()
    pos=np.where(spans[:,0]>=448)[0]
    if (spans[pos[0],0],spans[pos[-1],1])!=(448,480):
        raise ValueError("Invalid final window")
    train=split=="train"
    norms={l:[] for l in layers}
    for i in range(0,train.sum(),batch):
        st,_,_=extract(adapter,layers,pos,x[train][i:i+batch],False)
        for l in layers:
            norms[l].extend(np.linalg.norm(st[l].mean(axis=1),axis=1))
    xs=x[~train]
    states={l:[] for l in layers}
    jac={l:[] for l in layers}
    base=[]
    for i in range(0,len(xs),batch):
        obs=xs[i:i+batch]
        h,j,point=extract(adapter,layers,pos,obs,True)
        ordinary=adapter.predict(obs,16,[.5])["quantiles"][...,0]
        np.testing.assert_allclose(point,ordinary[:,:8],atol=7e-5,rtol=3e-5)
        base.append(ordinary)
        for l in layers:
            states[l].append(h[l])
            jac[l].append(j[l])
    scratch.mkdir(parents=True,exist_ok=True)
    for depth,l in zip(depths,layers):
        j=np.concatenate(jac[l])
        if not np.isfinite(j).all():
            raise ValueError("Nonfinite receiver derivative")
        if l==layers[2]:
            with np.load(OLD_JAC/(name+"_jac.npz")) as old,np.load(FRESH/(name+"_geometry.npz")) as fresh:
                np.testing.assert_array_equal(ids[~train],np.r_[old["ids"],fresh["ids"]])
                previous=np.concatenate([old["jac"],fresh["jac"]])
                relative=np.linalg.norm(j-previous)/np.linalg.norm(previous)
                if relative>3e-4:
                    raise ValueError(f"Middle derivative replay disagrees by {relative}")
                print(name,"middle derivative relative replay",relative,flush=True)
        np.savez_compressed(scratch/f"{name}_{depth}.npz",ids=ids[~train],split=split[~train],jac=j,
                            states=np.concatenate(states[l]),base=np.concatenate(base),
                            radius=np.median(norms[l]),layer=l,positions=pos,depth=depth,block_count=n)
        print(name,l,"recorded",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--batch",type=int,default=8)
    a=p.parse_args()
    torch.set_num_threads(2)
    collect(a.model,a.scratch,a.batch)
