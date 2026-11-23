"""Evaluate the already-fitted full hidden-space SAE policy on the locked new cohorts."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.materials import MODELS, adapter_for, capture, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE, OUT
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as ORIGINAL, SCRATCH
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape, solve_adaptive, solve_fixed
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint


def collect(name,cache,old_scratch,out,batch):
    capture(name,cache)
    data=dict(np.load(cache/"data.npz"))
    x=data["x"]
    h,pos,ids=read_materials(cache,name)
    np.testing.assert_array_equal(ids,data["ids"])
    with np.load(REPLICATION/"results"/(name+"_seed0.npz")) as reg:
        atoms=reg["atoms"]
    archive=np.load(ORIGINAL/"results"/(name+".npz"))
    np.testing.assert_array_equal(atoms,archive["atoms"])
    sae=load_sae_checkpoint(str(dictionary_path(name)))
    with torch.inference_mode():
        codes=sae.encode(torch.from_numpy(h.mean(axis=1))).numpy()[:,atoms]
    dirs=sae.W_dec.detach().numpy()[atoms]
    model=adapter_for(name)
    layer=MODELS[name][-1]
    selector=last_window_slice(model,pos)
    old=dict(np.load(old_scratch/(name+"_jac.npz")))
    radius=float(old["radius"])
    gradients,points=[],[]
    for start in range(0,len(x),batch):
        rows=torch.from_numpy(h[start:start+batch])
        delta=torch.zeros_like(rows[:,0],requires_grad=True)
        with torch.enable_grad(), token_patch(model.module,layer,selector,rows+delta[:,None]):
            point=model.predict_differentiable(x[start:start+batch],8)
            jac=[torch.autograd.grad(point[:,j].sum(),delta,retain_graph=j<7)[0] for j in range(8)]
        gradients.append(torch.stack(jac,dim=1).detach().numpy())
        points.append(point.detach().numpy())
    jac=np.concatenate(gradients)
    ordinary=change_forecasts(model,layer,x,h,None,selector,batch)
    np.testing.assert_allclose(np.concatenate(points),ordinary,atol=8e-5,rtol=2e-5)
    zero=change_forecasts(model,layer,x,h,np.zeros_like(h[:,0]),selector,batch)
    np.testing.assert_allclose(zero,ordinary,atol=8e-5,rtol=2e-5)
    chk=np.r_[0:4,128:132]
    direction=np.random.default_rng(8903).normal(size=(len(chk),h.shape[-1])).astype("f4")
    direction*= (.002*radius/np.linalg.norm(direction,axis=1))[:,None]
    high,low=[change_forecasts(model,layer,x[chk],h[chk],sign*direction,selector,batch) for sign in (1,-1)]
    predicted=np.einsum("nhd,nd->nh",jac[chk],direction)
    observed=(high-low)/2
    errors=[float(np.linalg.norm(v-p)/max(np.linalg.norm(v),1e-15)) for v,p in
            [(observed[:4],predicted[:4]),(observed[4:],predicted[4:])]]
    if not np.isfinite(jac).all() or max(errors)>.08:
        raise ValueError(f"{name} signed finite-difference error {errors}")
    native=run_arm(model,layer,selector,x,h,ordinary,codes>0,codes,dirs,32)
    with dictionary_path(name).open("rb") as f:
        identity=file_digest(f,"sha256").hexdigest()
    checks={"finite_difference_relative_error":errors,"native_ordinary_max_abs":float(abs(np.concatenate(points)-ordinary).max()),
            "zero_patch_max_abs":float(abs(zero-ordinary).max()),"source_dictionary_sha256":identity}
    np.savez_compressed(cache/(name+"_geometry.npz"),ids=ids,jac=jac,forecast=ordinary,radius=radius)
    np.savez_compressed(out/(name+"_native.npz"),ids=ids,split=data["split"],atoms=atoms,codes=codes,native=native)
    (out/(name+"_checks.json")).write_text(json.dumps(checks,indent=2)+"\n")
    print(name,"new cohort geometry and actual source ablations",errors,flush=True)


def transfer(source,cache,old_scratch,out,batch):
    target=next(m for m in MODELS if m!=source)
    data=dict(np.load(cache/"data.npz"))
    x=data["x"]
    own=dict(np.load(out/(source+"_native.npz")))
    sg,tg=[dict(np.load(cache/(m+"_geometry.npz"))) for m in (source,target)]
    ot,os=[dict(np.load(old_scratch/(m+"_jac.npz"))) for m in (target,source)]
    for z in (own,sg,tg):
        np.testing.assert_array_equal(z["ids"],data["ids"])
    np.testing.assert_array_equal(ot["ids"],os["ids"])
    np.testing.assert_array_equal(ot["split"],os["split"])
    cal=os["split"]=="cal"
    assert int(cal.sum())==96 and not set(os["ids"][cal]).intersection(data["ids"])
    source_cal=os["jac"][cal].astype(float)
    target_cal=ot["jac"][cal].astype(float)
    common=source_cal.mean(axis=1,keepdims=True)
    weight=np.sqrt(np.mean(common**2)/np.mean((source_cal-common)**2))
    original=dict(np.load(ORIGINAL/"results"/(source+".npz")))
    np.testing.assert_allclose(original["shape_weight"],weight,rtol=1e-12)
    dirs=load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[own["atoms"]].astype(float).T
    req=sg["jac"].astype(float)@dirs
    jt=tg["jac"].astype(float)
    old_res=source_cal@dirs
    fixed=solve_fixed(boost_shape(target_cal,weight),boost_shape(old_res,weight))
    model=adapter_for(target)
    h,pos,ids=read_materials(cache,target)
    np.testing.assert_array_equal(ids,data["ids"])
    selector=last_window_slice(model,pos)
    layer=MODELS[target][-1]
    per={"source_linear":(-own["codes"][:,:,None]*req.swapaxes(1,2)).astype("f4")}
    for key in ("fixed","adaptive","wrong_leads","mean_donor"):
        if key=="fixed":
            write=fixed
        else:
            jac=jt[:,[3,6,1,7,0,2,5,4]] if key=="wrong_leads" else jt
            forecast=np.broadcast_to(old_res.mean(0),req.shape) if key=="mean_donor" else req
            write=solve_adaptive(boost_shape(jac,weight),boost_shape(forecast,weight))
        per[key],per[key+"_clipped"]=apply(model,layer,selector,x,h,tg["forecast"],own["codes"],write,float(ot["radius"]),batch)
        print(source,"->",target,key,"complete",flush=True)
    np.savez_compressed(out/(source+"_writes.npz"),ids=ids,split=data["split"],atoms=own["atoms"],active=own["codes"]>0,
                        shape_weight=weight,old_target_radius=ot["radius"],calibration_ids=os["ids"][cal],**per)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["collect","transfer"])
    p.add_argument("model",choices=MODELS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--old-scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=="collect":
        collect(a.model,a.cache,a.old_scratch,a.out,a.batch or 8)
    else:
        transfer(a.model,a.cache,a.old_scratch,a.out,a.batch or 32)
