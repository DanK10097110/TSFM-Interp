"""Evaluate precomputed receiver libraries on fresh series, including station-blocked weather data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.run import FIT
from research.activation_transfer.shape_transfer.steering_budget.run import HERE,coefficients
from research.activation_transfer.shape_transfer.steering_budget.score import component,difference,entry
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.support_confirmation import data as fresh
from research.activation_transfer.shape_transfer.support_confirmation.noaa import data as weather
from research.activation_transfer.shape_transfer.support_confirmation.noaa.score import cluster_skill,eligible_mask
from research.activation_transfer.shape_transfer.support_confirmation.run import archive_inputs
from tsfm_lens.sae.train import load_sae_checkpoint

BASES=("sae","pca","response8","response16","response32")


def run(cohort,target,out,batch):
    if cohort=="noaa":
        cache=weather.CACHE
        archive=cache/(target+"_reference.npz")
    else:
        cache=fresh.CACHE if cohort=="long" else fresh.CACHE/cohort
        archive=fresh.OUT/(cohort+"_"+target+".npz")
    with np.load(archive) as f:
        ids,row,col,dense_vector=f["ids"],f["rows"],f["cols"],f["dense_vector"]
        weight=float(f["weight"]);atoms=f["atoms"];codes=f["codes"]
        reference,source=f["dense"],f["source"]
        base=f["target_ordinary_celsius" if cohort=="noaa" else "ordinary_target"]
        budget=f["budget" if cohort=="noaa" else "dense_length"]
    with np.load(cache/"data.npz") as f:
        x=f["x"];np.testing.assert_array_equal(ids,f["ids"])
        sd,group=(f["sd"],f["station"]) if cohort=="noaa" else (np.ones(len(ids)),np.arange(len(ids)))
    h,positions,jac,_,_,_=archive_inputs(target,cache,ids);jac/=sd[:,None,None]
    goal=np.einsum("nhd,nd->nh",jac[row],dense_vector)
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        support=sae.encode(torch.from_numpy(h.mean(1))).topk(sae.k).indices.numpy()
    own=np.linalg.svd(sae.W_dec.detach().numpy()[support].astype(float),full_matrices=False)[2]
    with np.load(FIT/(target+".npz")) as f:
        response,pca=f["response"],f["pca"]
        np.testing.assert_equal(bool(set(ids)&(set(f["fit_ids"]) | set(f["train_ids"]))),False)
    adapter=fresh.adapter_for_length(target,x.shape[1])
    select=last_window_slice(adapter,positions);layer=MODELS[target][-1]
    zero=forecast16(adapter,layer,x[row],h[row],np.zeros_like(dense_vector),select,batch)
    np.testing.assert_allclose(zero,base,atol=3e-4,rtol=4e-4)
    result=dict(ids=ids,row=row,col=col,group=group,active=codes>0,atoms=atoms,source=source,
                dense=reference,budget=budget,sd=sd,goal=goal,weight=weight)
    for name in BASES:
        if name=="sae":axes=own
        elif name=="pca":axes=pca
        else:axes=response[:int(name.removeprefix("response"))]
        effects=[];lengths=[]
        for lo in range(0,len(row),batch):
            ix=slice(lo,lo+batch);r=row[ix]
            q=axes[r] if axes.ndim==3 else np.broadcast_to(axes,(len(r),*axes.shape))
            j=boost_shape(jac[r]@q.swapaxes(-1,-2),weight)
            y=boost_shape(goal[ix,:,None],weight)[:,:,0]
            radial,opt=coefficients(j,y,budget[ix])
            delta=np.einsum("nr,nrd->nd",opt,q).astype("f4")
            effects.append((forecast16(adapter,layer,x[r],h[r],delta,select,batch)-zero[ix])/sd[r,None])
            lengths.append(np.linalg.norm(delta.astype(float),axis=1))
        result[name]=np.concatenate(effects);result[name+"_length"]=np.concatenate(lengths)
        assert np.all(result[name+"_length"]<=budget*(1+1e-6))
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"{cohort}_{target}.npz",**result)


def score(cohort,target,out):
    with np.load(out/f"{cohort}_{target}.npz") as f:d=dict(f)
    groups=d["group"];keep=eligible_mask(d["active"],groups)
    n=int(keep.sum());mask=d["active"][:,keep]
    ans=dict(eligible=n,atoms=d["atoms"][keep].tolist(),events=int(mask.sum()),independent_groups=len(np.unique(groups)))
    if n:
        gd=resampling_counts(len(np.unique(groups)),draws=3000)
        fd=np.random.default_rng(2262201).multinomial(n,np.full(n,1/n),size=3000)
        def cube(y):
            w=np.zeros((*d["active"].shape,y.shape[-1]))
            w[d["row"],d["col"]]=y
            return w[:,keep]
        for label,sl in (("early",slice(8)),("late",slice(8,16))):
            for kind in ("shape","full"):
                for name in ("source","dense"):
                    truth=cube(component(d[name][:,sl],kind))
                    vals={b:cluster_skill(mask,cube(component(d[b][:,sl],kind)),truth,groups,gd,fd) for b in BASES}
                    stats={b:entry(v) for b,v in vals.items()}
                    for ctrl in ("sae","pca","response8","response16"):
                        stats["response32_minus_"+ctrl]=difference(vals["response32"],vals[ctrl])
                    ans["_".join((label,name,kind))]=stats
    (out/f"{cohort}_{target}.json").write_text(json.dumps(ans,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cohort",choices=("short","long","noaa"))
    p.add_argument("target",choices=MODELS);p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=24);a=p.parse_args();torch.set_num_threads(1)
    run(a.cohort,a.target,a.out,a.batch);score(a.cohort,a.target,a.out)
