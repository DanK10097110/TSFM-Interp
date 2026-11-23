"""Physically optimize against the pre-intervention donor estimate, keeping the true source dose fixed."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.feedback.data import HERE, OUT, SCRATCH
from research.activation_transfer.shape_transfer.depth.geometry import HERE as DEPTH
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL, boost_shape, solve_adaptive
from tsfm_lens.extraction.hooks import token_patch


def loss(effect, goal, weight):
    diff=effect[:,:8].astype(float)-goal
    level=diff.mean(axis=1,keepdims=True)
    return np.mean((level+weight*(diff-level))**2,axis=1)


def limit(v,cap):
    return v*np.minimum(1.,cap/np.maximum(np.linalg.norm(v,axis=1),1e-15))[:,None]


def derivative(enc,layer,sel,x,h,shift):
    delta=torch.zeros((len(h),h.shape[-1]),dtype=torch.float32,requires_grad=True)
    states=torch.from_numpy(h+shift[:,None].astype("f4"))
    with torch.enable_grad(),token_patch(enc.module,layer,sel,states+delta[:,None]):
        point=enc.predict_differentiable(x,8)
        rows=[torch.autograd.grad(point[:,i].sum(),delta,retain_graph=i<7)[0] for i in range(8)]
    return torch.stack(rows,dim=1).numpy().astype(float)


def run(depth,scratch,out,batch):
    with np.load(scratch/"data.npz") as f:
        ids,x,split=f["ids"],f["x"],f["split"]
    with np.load(scratch/"bolt_3.npz") as s,np.load(scratch/f"timesfm_{depth}.npz") as t:
        np.testing.assert_array_equal(ids,s["ids"])
        np.testing.assert_array_equal(ids,t["ids"])
        codes,atoms,native=s["codes"],s["atoms"],s["native"]
        source_j=s["jac"].astype(float)@s["dec"].astype(float)
        j,h,base,radius,layer,pos=t["jac"].astype(float),t["states"],t["base"],float(t["radius"]),str(t["layer"]),t["pos"]
    with np.load(FULL/"results/bolt.npz") as f:
        weight=float(f["shape_weight"])
    active=codes>0
    rows,cols=np.nonzero(active)
    enc=adapter_for("timesfm")
    for p in enc.module.parameters():
        p.requires_grad_(False)
    sel=last_window_slice(enc,pos)
    record=dict(ids=ids,split=split,active=active,atoms=atoms,source=native,
                goal=(-codes[:,:,None]*source_j.swapaxes(1,2)).astype("f4"),radius=radius,weight=weight)
    for arm in ("open","scalar","corrected"):
        record[arm]=np.zeros((*active.shape,16),dtype="f4")
        record[arm+"_loss"]=np.zeros(active.shape,dtype="f8")
        record[arm+"_norm"]=np.zeros(active.shape,dtype="f4")
    for arm in ("scalar","corrected"):
        record[arm+"_step"]=np.zeros(active.shape,dtype="f4")
    for start in range(0,len(rows),batch):
        i,c=rows[start:start+batch],cols[start:start+batch]
        goal=record["goal"][i,c].astype(float)
        def measure(v):
            return forecast16(enc,layer,x[i],h[i],v.astype("f4"),sel,batch)-base[i]
        def inverse(jac,remaining):
            return solve_adaptive(boost_shape(jac,weight),boost_shape(remaining[:,:,None],weight))[:,:,0]
        v=limit(inverse(j[i],goal),.2*radius)
        y=measure(v)
        def save(arm,z,vector,error,step=None):
            record[arm][i,c]=z
            record[arm+"_loss"][i,c]=error
            record[arm+"_norm"][i,c]=np.linalg.norm(vector,axis=1)/radius
            if step is not None:
                record[arm+"_step"][i,c]=step
        save("open",y,v,loss(y,goal,weight))
        # Tie-breaking favors the shortest scalar. The clean response change is exactly zero.
        best=np.zeros_like(y)
        shift=np.zeros_like(v)
        best_loss=loss(best,goal,weight)
        chosen=np.zeros(len(i))
        for step in (.25,.50,.75,1.):
            zz=y if step==1. else measure(step*v)
            err=loss(zz,goal,weight)
            use=err<best_loss
            best[use]=zz[use]; shift[use]=step*v[use]; best_loss[use]=err[use]; chosen[use]=step
        save("scalar",best,shift,best_loss,chosen)
        js=derivative(enc,layer,sel,x[i],h[i],shift)
        update=inverse(js,goal-best[:,:8])
        origin=shift.copy()
        chosen[:]=0.
        for step in (.25,.50,1.):
            proposed=limit(origin+step*update,.2*radius)
            zz=measure(proposed)
            err=loss(zz,goal,weight)
            use=err<best_loss
            best[use]=zz[use]; shift[use]=proposed[use]; best_loss[use]=err[use]; chosen[use]=step
        save("corrected",best,shift,best_loss,chosen)
    if np.any(record["scalar_loss"]>record["open_loss"]+1e-12) or np.any(record["corrected_loss"]>record["scalar_loss"]+1e-12):
        raise ValueError("Target-only search failed to preserve its best candidate")
    with np.load(DEPTH/"results"/f"timesfm_{depth}.npz") as old:
        m=old["split"]=="machines"
        np.testing.assert_allclose(record["open"][split=="original"],old["local_20"][m],atol=4e-4,rtol=.004)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"block{depth}.npz",**record)
    print(depth,"complete",len(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("depth",type=int,choices=(0,10,19))
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=12)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.depth,a.scratch,a.out,a.batch)
