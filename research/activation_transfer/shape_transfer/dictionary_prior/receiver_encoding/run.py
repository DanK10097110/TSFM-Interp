"""Re-encode one bounded cross-checkpoint receiver write on the receiver's real SAE and live tokens."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND, HERE as OBSERVED
from research.activation_transfer.shape_transfer.dictionary_prior.refit.encoding import endpoint, regularized
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape, solve_adaptive
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
ARMS = ("dense", "hard", "locked", "shoulder", "span")
BATCH = 32


def cap_norm(v, limit):
    n=np.linalg.norm(v,axis=1)
    return v*np.minimum(1.,limit/np.maximum(n,1e-16))[:,None], n


def project(v, bank):
    """Vector projection into an arbitrary possibly rank-deficient row span."""
    _,s,vt=np.linalg.svd(bank.astype(float),full_matrices=False)
    if not len(s) or s[0]==0:
        return np.zeros_like(v)
    use=vt[s>s[0]*max(bank.shape)*np.finfo(float).eps]
    return (v@use.T)@use


def material(source):
    target=next(k for k in MODELS if k!=source)
    with np.load(BLIND/"data.npz") as f:x,ids,split=f["x"],f["ids"],f["split"]
    with np.load(OBSERVED/"results"/(source+"_fresh.npz")) as f:
        np.testing.assert_array_equal(ids,f["ids"])
        z,atoms,active=f["codes"],f["atoms"],f["active"]
        np.testing.assert_array_equal(z>0,active)
        weight,radius=float(f["weight"]),float(f["radius"])
        native,original=f["source"],f["minimum"]
    J={}
    for m in (source,target):
        with np.load(BLIND/(m+"_early.npz")) as f:
            np.testing.assert_array_equal(ids,f["ids"]); J[m]=f["jac"].astype(float)
            if m==target: base=f["forecast"]
    h,pos,order=read_materials(BLIND,target)
    np.testing.assert_array_equal(ids,order)
    sae_src=load_sae_checkpoint(str(dictionary_path(source)))
    D=sae_src.W_dec.detach().numpy()[atoms].astype(float).T
    vectors=solve_adaptive(boost_shape(J[target],weight), boost_shape(J[source]@D,weight))
    row,col=np.nonzero(active)
    v,requested=cap_norm(-z[row,col,None]*vectors[row,:,col],.2*radius)
    return target,x,ids,split,z,atoms,native,original,h,pos,base,radius,row,col,v.astype("f4"),requested


def encode_moves(sae,state,change):
    hard,locked,n_enter=endpoint(sae,state,change,1.)
    variants=regularized(sae,state,change,hard,locked)
    start=sae.encode(state)>0
    end=sae.encode(state+change)>0
    np.testing.assert_array_equal(n_enter.numpy(),(end&~start).sum(1).numpy())
    return ({k:variants[k] for k in ("hard","locked","shoulder")},
            n_enter,(start&~end).sum(1),start.sum(1))


def run(source):
    target,x,ids,split,z,atoms,native,original,h,pos,base,radius,row,col,v,requested=material(source)
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    clean=h.mean(axis=1)
    decoder=sae.W_dec.detach().numpy()
    model=adapter_for(target);layer=MODELS[target][-1];selector=last_window_slice(model,pos)
    rec=dict(ids=ids,split=split,codes=z,atoms=atoms,active=z>0,native=native,original_dense=original,
             rows=row,cols=col,full_dense_norm=requested,cap=.2*radius)
    moves={k:np.zeros_like(v) for k in ARMS};moves["dense"]=v
    with torch.inference_mode():
        clean_codes=sae.encode(torch.from_numpy(clean))
        for i in np.unique(row):
            selected=row==i
            support=(clean_codes[i]>0).numpy()
            moves["span"][selected]=project(v[selected],decoder[support]).astype("f4")
        enter,leave,clean_count=[],[],[]
        for lo in range(0,len(row),BATCH):
            group=slice(lo,lo+BATCH)
            state=torch.from_numpy(clean[row[group]])
            change=torch.from_numpy(v[group])
            variants,n_enter,n_leave,n_clean=encode_moves(sae,state,change)
            for arm,vector in variants.items():moves[arm][group]=vector.numpy()
            enter.extend(n_enter.numpy());leave.extend(n_leave.numpy());clean_count.extend(n_clean.numpy())
    rec.update(enter=np.asarray(enter),leave=np.asarray(leave),clean_count=np.asarray(clean_count))
    zero=forecast16(model,layer,x[row],h[row],np.zeros_like(v),selector,BATCH)
    rec["zero_rebatch_max"]=np.max(np.abs(zero-base[row]))
    base_len=np.linalg.norm(v.astype(float),axis=1)
    for arm in ARMS:
        vec=moves[arm].astype(float)
        adjusted,length=cap_norm(vec,.2*radius)
        if arm=="dense":adjusted=v  # already capped in full precision prior to the forward's float32 cast
        norm=np.linalg.norm(adjusted,axis=1)
        rec[arm+"_pre_norm"]=length;rec[arm+"_norm"]=norm
        rec[arm+"_cap"]=(requested>.2*radius) if arm=="dense" else length>.2*radius
        rec[arm+"_cos"]=np.divide(np.sum(vec*v,axis=1),length*base_len,out=np.zeros_like(length),where=length*base_len>0)
        rec[arm+"_effect"]=forecast16(model,layer,x[row],h[row],adjusted.astype("f4"),selector,BATCH)-zero
        if arm=="dense":
            err=np.max(np.abs(rec["dense_effect"]-original[row,col]))
            rec["max_dense_reference_error"]=err
            np.testing.assert_allclose(rec["dense_effect"],original[row,col],atol=3e-5,rtol=2e-4)
        print(source,target,arm,len(row),flush=True)
    np.savez_compressed(HERE/"results"/(source+".npz"), **rec)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    torch.set_num_threads(1)
    run(p.parse_args().source)
