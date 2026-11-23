"""Fit forecast changes inside the very same 32 active decoder axes that failed direct re-encoding."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND,HERE as OBSERVED
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import HERE,material
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape,solve_adaptive
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from tsfm_lens.sae.train import load_sae_checkpoint

OUT=HERE/"results"
ARMS=("active_same","active_cap","random_same","random_cap")


def constrained(j,q,v,w):
    """Same ridge solve in orthonormal subspace coordinates, followed by conversion to the native basis."""
    lead=j@q.swapaxes(1,2)
    goal=j@v[:,:,None]
    coef=solve_adaptive(boost_shape(lead,w),boost_shape(goal,w))[:,:,0]
    return np.einsum("nr,nrd->nd",coef,q)


def run(source,out):
    target,x,ids,split,z,atoms,donor,previous,h,pos,base,radius,row,col,v,_=material(source)
    with np.load(BLIND/(target+"_early.npz")) as f:
        np.testing.assert_array_equal(ids,f["ids"])
        jac=f["jac"].astype(float)
    with np.load(OBSERVED/"results"/(source+"_fresh.npz")) as f:
        w=float(f["weight"])
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        support=torch.topk(sae.encode(torch.from_numpy(h.mean(1))),sae.k,dim=1)
    if support.values.min()<=0:
        raise ValueError("Expected 32 active directions")
    decoder=sae.W_dec.detach().numpy().astype(float)
    _,s,q=np.linalg.svd(decoder[support.indices.numpy()],full_matrices=False)
    if np.any(s[:,-1]<1e-7*s[:,0]):
        raise ValueError("Found non-full-rank dictionary span")
    randomq=np.linalg.qr(np.random.default_rng(5181201).normal(size=(q.shape[2],32)))[0].T
    with np.load(OUT/(source+".npz")) as parent:
        np.testing.assert_array_equal(row,parent["rows"])
        np.testing.assert_array_equal(col,parent["cols"])
        dense,hard,proj=(parent[k+"_effect"] for k in ("dense","hard","span"))
    repeats=np.arange(len(row))
    enc=adapter_for(target)
    sel=last_window_slice(enc,pos)
    site=MODELS[target][-1]
    rec=dict(ids=ids,split=split,atoms=atoms,active=z>0,rows=row,cols=col,
             source=donor[row,col],dense=dense,hard=hard,projection=proj)
    for k in ARMS:
        rec[k]=np.empty((len(row),16),dtype="f4")
        rec[k+"_scale"]=np.empty(len(row),dtype="f4")
        rec[k+"_norm_ratio"]=np.empty(len(row),dtype="f4")
    for start in range(0,len(row),24):
        ix=repeats[start:start+24]
        r=row[ix]
        v0=v[ix].astype(float)
        length=np.linalg.norm(v0,axis=1)
        zero=forecast16(enc,site,x[r],h[r],np.zeros_like(v[ix]),sel,24)
        for domain,axes in (("active",q[r]),("random",np.broadcast_to(randomq,(len(r),*randomq.shape)))):
            candidate=constrained(jac[r],axes,v0,w)
            size=np.linalg.norm(candidate,axis=1)
            for budget,limit in (("same",length),("cap",.2*radius)):
                key=domain+"_"+budget
                scale=np.minimum(1.,limit/np.maximum(size,1e-16))
                delta=(candidate*scale[:,None]).astype("f4")
                rec[key][ix]=forecast16(enc,site,x[r],h[r],delta,sel,24)-zero
                rec[key+"_scale"][ix]=scale
                rec[key+"_norm_ratio"][ix]=size*scale/length
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(source+"_span_inverse.npz"),**rec)


def score(out):
    result={}
    for src in ("bolt","timesfm"):
        with np.load(out/(src+"_span_inverse.npz")) as f:
            d=dict(f)
        rows,cols=d["rows"],d["cols"]
        result[src]={}
        for dom in np.unique(d["split"]):
            use=d["split"]==dom
            keep=d["active"][use].sum(axis=0)>=8
            m=d["active"][use][:,keep]
            events=use[rows]&keep[cols]
            hweights=resampling_counts(len(m),draws=3000)
            nf=m.shape[1]
            fweights=np.random.default_rng(221106).multinomial(nf,np.full(nf,1/nf),size=len(hweights))
            cell=dict(events=int(events.sum()),features=nf,
                      median_ratio={arm:float(np.median(d[arm+"_norm_ratio"][events])) for arm in ARMS})
            for part,sl in (("early",slice(0,8)),("late",slice(8,16))):
                def read(key):
                    values=np.zeros((*d["active"].shape,8))
                    values[rows,cols]=d[key][:,sl]
                    values=values[use][:,keep]
                    return values-values.mean(axis=-1,keepdims=True)
                for reference in ("dense","source"):
                    truth=read(reference)
                    vals={arm:intervals(m,read(arm),truth,hweights,fweights)[:3] for arm in (*ARMS,"hard","projection")}
                    block={arm:entry(val) for arm,val in vals.items()}
                    for a,b in (("active_cap","random_cap"),("active_same","random_same"),("active_cap","projection"),("active_same","projection")):
                        block[a+"_minus_"+b]=entry(tuple(x-y for x,y in zip(vals[a],vals[b])))
                    cell[part+"_"+reference]=block
            result[src][str(dom)]=cell
    (out/"span_inverse.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("bolt","timesfm","score"))
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.stage=="score":score(a.out)
    else:run(a.stage,a.out)
