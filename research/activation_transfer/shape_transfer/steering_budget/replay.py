"""Recompute real matched-budget writes with a scalar KKT root and live additive hooks."""
import argparse
import json
import numpy as np
import torch
from scipy.optimize import brentq

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.response_subspace.run import FIT
from research.activation_transfer.shape_transfer.support_confirmation import data as fresh
from research.activation_transfer.shape_transfer.support_confirmation.noaa import data as weather
from research.activation_transfer.shape_transfer.support_confirmation.run import archive_inputs
from research.activation_transfer.shape_transfer.steering_budget.run import HERE
from tsfm_lens.sae.train import load_sae_checkpoint


def oracle(e,y,cap,weight):
    transform=np.eye(8)*weight+(1-weight)*np.ones((8,8))/8
    a,b=transform@e,transform@y
    g=a@a.T
    lam=.001*np.linalg.eigvalsh(g)[-1]
    if lam<=0:return np.zeros(a.shape[-1])
    def solve(eta):return a.T@np.linalg.solve(g+(lam+eta)*np.eye(8),b)
    if np.linalg.norm(solve(0))<=cap:return solve(0)
    eta=brentq(lambda v:np.linalg.norm(solve(v))/cap-1,0,max(1,np.linalg.norm(a.T@b)/cap),xtol=1e-17)
    return solve(eta)


def run(cohort,target):
    cache=weather.CACHE if cohort=="noaa" else fresh.CACHE/("short" if cohort=="short" else "")
    archive=HERE/"results"/f"{cohort}_{target}.npz"
    with np.load(archive) as f:d=dict(f)
    with np.load(cache/"data.npz") as f:x,ids=f["x"],f["ids"]
    np.testing.assert_array_equal(ids,d["ids"])
    h,pos,j,_,_,_=archive_inputs(target,cache,ids)
    j/=d["sd"][:,None,None]
    a=fresh.adapter_for_length(target,x.shape[1]);selector=last_window_slice(a,pos)
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    with np.load(FIT/(target+".npz")) as f:fixed,pca=f["response"][:32],f["pca"]
    rng=np.random.default_rng(22112295)
    events=np.sort(rng.choice(len(d["row"]),min(18,len(d["row"])),replace=False))
    row=d["row"][events];obs=x[row];weight=float(d["weight"])
    ordinary=a.predict(obs,16,[.5])["quantiles"][:,:,0]
    with torch.inference_mode():
        code=sae.encode(torch.from_numpy(h[row].mean(1))).numpy()
    # Take active positive IDs in numpy; all audited rows must have exactly k positive gates.
    lst=[np.flatnonzero(v>0) for v in code]
    assert all(len(v)==sae.k for v in lst)
    summary={}
    for name in ("sae","response32","pca"):
        directions=[]
        for i,event in enumerate(events):
            basis=np.linalg.qr(sae.W_dec.detach().numpy()[lst[i]].astype(float).T)[0].T if name=="sae" else (fixed if name=="response32" else pca)
            coeff=oracle(j[row[i]]@basis.T,d["goal"][event],d["budget"][event],weight)
            directions.append(coeff@basis)
        delta=np.stack(directions).astype("f4")
        def hook(_,args,out):
            t=out if torch.is_tensor(out) else out[0]
            v=t.clone();v[:,selector(v.shape[1])]+=torch.as_tensor(delta[:,None]).to(v)
            return v if torch.is_tensor(out) else (v,*out[1:])
        handle=a.module.get_submodule(MODELS[target][-1]).register_forward_hook(hook)
        try:measured=(a.predict(obs,16,[.5])["quantiles"][:,:,0]-ordinary)/d["sd"][row,None]
        finally:handle.remove()
        physical=d[name][events]
        gap=measured-physical
        norm_gap=np.linalg.norm(delta.astype(float),axis=1)-d[name+"_length"][events]
        np.testing.assert_allclose(gap,0.,atol=2e-5)
        np.testing.assert_allclose(norm_gap,0.,atol=1e-5,rtol=2e-6)
        summary[name]=dict(max_abs_forecast_gap=float(np.max(np.abs(gap))),
                           max_abs_norm_gap=float(np.max(np.abs(norm_gap))),
                           relative_forecast_error=float(np.linalg.norm(gap)/np.linalg.norm(physical)))
    record=dict(events=events.tolist(),ids=ids[row].tolist(),methods=summary)
    archive.with_name(f"{cohort}_{target}_replay.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cohort",choices=("short","long","noaa"));p.add_argument("target",choices=MODELS)
    a=p.parse_args();torch.set_num_threads(1);run(a.cohort,a.target)
