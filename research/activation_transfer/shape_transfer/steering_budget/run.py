"""Compare fixed-radius ridge solutions in active SAE and independently calibrated 32-dimensional spaces."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND, HERE as OBSERVED
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import material
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.response_subspace.run import FIT
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
BASES = ("sae", "pca", "response", "random")
SOLVERS = ("radial", "optimal", "centered")


def coefficients(matrix, goal, limit):
    """Solve ridge with a ball constraint. The extra KKT multiplier is shared by all axes in a row."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    proj = np.einsum("nij,ni->nj", u, goal)
    numerator = s*proj
    ridge = .001*s[:,0]**2
    denom = s*s + ridge[:,None]
    def coordinates(multiplier):
        denom_shift = denom + multiplier[:,None]
        return np.divide(numerator, denom_shift, out=np.zeros_like(numerator), where=denom_shift>0)
    raw = coordinates(np.zeros(len(matrix)))
    size = np.linalg.norm(raw,axis=1)
    radial = raw * np.minimum(1.,limit/np.maximum(1e-30,size))[:,None]
    lo = np.zeros(len(matrix))
    hi = np.linalg.norm(numerator,axis=1)/np.maximum(limit,1e-30)
    shrink = size>limit
    for _ in range(48):
        mid = (hi+lo)/2
        exceeds = np.linalg.norm(coordinates(mid),axis=1)>limit
        lo = np.where(exceeds,mid,lo)
        hi = np.where(exceeds,hi,mid)
    optimal = coordinates(np.where(shrink,hi,0.))
    # Orthogonal coordinates preserve the physical length.
    return (np.einsum("nk,nkd->nd", a,vt) for a in (radial,optimal))


def solve(jac, goal, basis, limits, weight):
    reduced = jac @ basis.swapaxes(-1,-2)
    balanced = boost_shape(reduced,weight)
    rhs = boost_shape(goal[:,:,None],weight)[:,:,0]
    radial, optimal = coefficients(balanced,rhs,limits)
    centered = reduced-reduced.mean(1,keepdims=True)
    _, shape = coefficients(centered,goal-goal.mean(1,keepdims=True),limits)
    return {k:np.einsum("nr,nrd->nd",a,basis)
            for k,a in zip(SOLVERS,(radial,optimal,shape))}


def run(source,out,batch):
    target,x,ids,split,z,atoms,donor,_,h,pos,_,_,row,col,v,_=material(source)
    with np.load(BLIND/(target+"_early.npz")) as f:
        np.testing.assert_array_equal(ids,f["ids"])
        jac=f["jac"].astype(float)
    with np.load(OBSERVED/"results"/(source+"_fresh.npz")) as f:
        w=float(f["weight"])
    parent=HERE.parent/"dictionary_prior/receiver_encoding/results"
    with np.load(parent/(source+"_span_inverse.npz")) as f:
        np.testing.assert_array_equal(row,f["rows"])
        np.testing.assert_array_equal(col,f["cols"])
        dense=f["dense"]
        old_active=f["active_same"]
        old_random=f["random_same"]
    with np.load(FIT/(target+".npz")) as f:
        assert not (set(ids) & (set(f["fit_ids"]) | set(f["train_ids"])))
        pca=f["pca"][:32]
        response=f["response"][:32]
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        active=torch.topk(sae.encode(torch.from_numpy(h.mean(1))),sae.k).indices.numpy()
    active_basis=np.linalg.svd(sae.W_dec.detach().numpy()[active].astype(float),full_matrices=False)[2]
    rng=np.random.default_rng(5181201)
    random=np.linalg.qr(rng.normal(size=(jac.shape[-1],32)))[0].T
    bases=dict(sae=active_basis,pca=pca,response=response,random=random)
    adapter=adapter_for(target)
    site=MODELS[target][-1]
    select=last_window_slice(adapter,pos)
    vals=dict(row=row,col=col,ids=ids,split=split,active=z>0,atoms=atoms,
              dense=dense,donor=donor[row,col],length=np.linalg.norm(v.astype(float),axis=1))
    goal=np.einsum("nhd,nd->nh",jac[row],v)
    vals["goal"]=goal
    baseline=forecast16(adapter,site,x[row],h[row],np.zeros_like(v),select,batch)
    for name,basis in bases.items():
        for method in SOLVERS:
            key=name+"_"+method
            vals[key]=np.empty_like(dense)
            vals[key+"_linear"]=np.empty_like(goal,dtype="f4")
            vals[key+"_length"]=np.empty(len(row),dtype="f4")
        for lo in range(0,len(row),batch):
            sl=slice(lo,lo+batch)
            r=row[sl]
            q=basis[r] if name=="sae" else np.broadcast_to(basis,(len(r),*basis.shape))
            writes=solve(jac[r],goal[sl],q,vals["length"][sl],w)
            for method,delta in writes.items():
                key=name+"_"+method
                vals[key+"_linear"][sl]=np.einsum("nhd,nd->nh",jac[r],delta)
                vals[key+"_length"][sl]=np.linalg.norm(delta,axis=1)
                y=forecast16(adapter,site,x[r],h[r],delta.astype("f4"),select,batch)
                vals[key][sl]=y-baseline[sl]
        print(source,name,flush=True)
    for method,original in (("sae_radial",old_active),("random_radial",old_random)):
        np.testing.assert_allclose(vals[method],original,atol=2e-5,rtol=.002)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(source+".npz"),**vals)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=24)
    args=p.parse_args()
    torch.set_num_threads(1)
    run(args.source,args.out,args.batch)
