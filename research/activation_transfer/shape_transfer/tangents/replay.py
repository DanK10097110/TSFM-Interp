"""Independently resolve primary maps, refetch observed-window states and replay live test patches."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
import torch

from research.activation_transfer.forecast_requests.run import adapter_for, live_effect, scale_factors
from research.activation_transfer.shape_transfer.tangents.capture import CACHE, NFIT, NTRAIN, ORIG, OUT
from research.activation_transfer.shape_transfer.tangents.forecast import changing_effect
from research.activation_transfer.steering.study import sites_and_states
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite


def independent_writes(xs,ys,dirs,penalty):
    """Direct dual action on an independently assembled ridge normal equation."""
    G=xs.T@xs
    top=eigh(G,subset_by_index=[len(G)-1,len(G)-1],eigvals_only=True,check_finite=False)[0]
    coef=cho_solve(cho_factor(G+penalty*top*np.eye(len(G)),check_finite=False),dirs.T,check_finite=False)
    return (coef.T@xs.T)@ys


def run(target,cache,original,out):
    source="timesfm" if target=="bolt" else "bolt"
    d=dict(np.load(cache/"inputs.npz"))
    a,b=[dict(np.load(cache/(m+"_deltas.npz"))) for m in (source,target)]
    patches=dict(np.load(cache/(target+"_patches.npz")))
    stem=source+"_to_"+target
    release=dict(np.load(out/(stem+"_forecasts.npz")))
    meta=json.loads((out/(stem+"_map.json")).read_text())
    x,y=[z["clean"][:NTRAIN].astype(float) for z in (a,b)]
    cx=x-x.mean(0);cy=y-y.mean(0)
    norm=np.sqrt((cx*cx).sum())
    cx,cy=cx/norm,cy/norm
    wave=(d["plus"][:NTRAIN,:5].astype(float)-d["minus"][:NTRAIN,:5].astype(float))/2
    rms=np.maximum(np.sqrt((wave**2).mean(-1,keepdims=True)),.02)
    xs,ys=[z["delta"][:NTRAIN,:5].astype(float)/rms for z in (a,b)]
    scales=np.sqrt((xs*xs).sum(axis=(0,2)))*np.sqrt(5.)
    xs,ys=xs/scales[None,:,None],ys/scales[None,:,None]
    tx,ty=[m.reshape(-1,m.shape[-1]) for m in (xs,ys)]
    report={}
    for name,(u,v) in {"clean":(cx,cy),"delta":(tx,ty),"joint":(np.vstack([cx,tx]),np.vstack([cy,ty]))}.items():
        want=patches["write_"+name]
        actual=independent_writes(u,v,a["direction"],meta["penalties"][name])
        report[name+"_normal_equation_max_abs"]=float(np.max(np.abs(want-actual)))
        np.testing.assert_allclose(want,actual,atol=5e-5,rtol=5e-5)
    doses=np.stack([scale_factors(release["calibration_effects"][:,i].astype(float),release["requests"])
                    for i in range(len(release["arms"]))])
    learned=release["arms"]!="legacy"
    np.testing.assert_allclose(doses[learned],release["doses"][learned],atol=5e-5,rtol=1e-5)
    stack=np.stack([patches["write_"+k] for k in release["arms"]])
    actions=np.stack([stack,stack*release["doses"][:,:,None]],axis=1)
    idx=np.array([0,10,64,77])
    rows=idx+NFIT
    f=np.load(original/(target+"_fit.npz"))
    site=InterventionSite(**json.loads(str(f["metadata"]))["site"][0])
    model=SteeringModel.from_adapter(adapter_for(target),quantiles=[.5])
    x=d["x"][rows]
    norms=np.linalg.norm(b["clean"][rows],axis=1)
    fresh=live_effect(model,site,x,norms,actions.reshape(-1,actions.shape[-1]),12)
    ref=release["effects"][idx].reshape(fresh.shape)
    report["all_source_and_calibrated_writes_max_abs"]=float(np.abs(fresh-ref).max())
    np.testing.assert_allclose(fresh,ref,atol=3e-4,rtol=1e-5)
    pair=np.r_[d["plus"][rows,5],d["minus"][rows,5]]
    _,states=sites_and_states(model,[site.module],pair,4)
    high,low=np.split(states[site.module],2)
    repeat=(high-low)/2
    np.testing.assert_allclose(repeat,b["delta"][rows,5],atol=3e-4,rtol=1e-4)
    moves=np.stack([repeat if k=="native" else patches["shift_"+k][idx] for k in release["natural_keys"]],axis=1)
    observed=changing_effect(model,site,x,norms,moves)
    report["natural_patch_max_abs"]=float(np.abs(observed-release["natural"][idx]).max())
    np.testing.assert_allclose(observed,release["natural"][idx],atol=3e-4,rtol=1e-5)
    report["replayed_ids"]=release["ids"][idx].tolist()
    (out/(stem+"_verification.json")).write_text(json.dumps(report,indent=2)+"\n")
    print(stem,report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=("bolt","timesfm"))
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--original",type=Path,default=ORIG)
    p.add_argument("--out",type=Path,default=OUT)
    opt=p.parse_args()
    torch.set_num_threads(2)
    run(opt.target,opt.cache,opt.original,opt.out)
