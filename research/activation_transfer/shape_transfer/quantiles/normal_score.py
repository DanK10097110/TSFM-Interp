"""Test a Gaussian interpolation control and five-anchor steering on untouched decile outcomes."""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtri
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.transfer import apply, solve_adaptive
from research.activation_transfer.shape_transfer.quantiles.geometry import HERE, SCRATCH, components, differential, material
from research.activation_transfer.shape_transfer.quantiles.interior import DECILES, INNER, all_deciles
from research.activation_transfer.shape_transfer.quantiles.intervene import stack, weights
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

SITES = [.1,.3,.5,.7,.9]
NORMAL = np.abs(ndtri(np.array(DECILES)[INNER])/ndtri(.1))


def five_components(values):
    """Median, half-width, asymmetry, left and right intermediate departures from normal scores."""
    low,left,med,right,high = np.moveaxis(values,-1,0)
    base = components(np.stack((low,med,high),axis=-1))
    return np.concatenate((base, (left-med-NORMAL[1]*(low-med))[...,None],
                                 (right-med-NORMAL[4]*(high-med))[...,None]),axis=-1)


def geometry(name,scratch,batch):
    enc = adapter_for(name)
    for p in enc.module.parameters():
        p.requires_grad_(False)
    ids,x,split,h,pos = material(name)
    cal=split=="cal"
    h_b, other, id_b = read_materials(BLIND,name)
    np.testing.assert_array_equal(pos,other)
    with np.load(BLIND/"data.npz") as f:
        np.testing.assert_array_equal(id_b,f["ids"])
        x = np.concatenate((x[cal],f["x"]))
        labels = np.r_[split[cal],f["split"]]
    ids,h = np.r_[ids[cal],id_b],np.concatenate((h[cal],h_b))
    if len(np.unique(ids))!=len(ids):
        raise ValueError("Overlapping split")
    layer=MODELS[name][-1]
    sel=last_window_slice(enc,pos)
    jac,base=[],[]
    if batch>96 or 96%batch:
        raise ValueError("Geometry batches must respect the calibration boundary")
    for i in range(0,len(ids),batch):
        # Match the two independent reference archives' CPU reduction order.
        torch.set_num_threads(2 if i<96 else 1)
        obs=x[i:i+batch]
        t=torch.from_numpy(h[i:i+batch])
        delta=torch.zeros_like(t[:,0],requires_grad=True)
        with torch.enable_grad(),token_patch(enc.module,layer,sel,t+delta[:,None]):
            y=differential(enc,name,obs,SITES)
            flat=y.reshape(len(obs),40)
            g=[torch.autograd.grad(flat[:,k].sum(),delta,retain_graph=k<39)[0] for k in range(40)]
        ordinary=enc.predict(obs,16,DECILES)["quantiles"]
        np.testing.assert_allclose(y.detach().numpy(),np.take(ordinary[:,:8], [0,2,4,6,8],axis=2),atol=1e-4,rtol=3e-5)
        base.append(ordinary)
        jac.append(torch.stack(g,dim=1).reshape(len(obs),8,5,-1).numpy())
    j,base=np.concatenate(jac),np.concatenate(base)
    with np.load(SCRATCH/(name+".npz")) as f:
        np.testing.assert_allclose(j[:96][:,:,[0,2,4],:],f["jac"][f["split"]=="cal"],atol=4e-5,rtol=.002)
        rad=float(f["radius"])
    with np.load(BLIND/(name+"_early.npz")) as f:
        np.testing.assert_allclose(j[96:,:,2],f["jac"],atol=4e-5,rtol=.002)
    # Direct signed finite difference on the two added curvature channels.
    test=np.r_[96:104,224:232]
    u=np.random.default_rng(402177).normal(size=(len(test),j.shape[-1])).astype("f4")
    u*=(.004*rad/np.linalg.norm(u,axis=1))[:,None]
    hi,lo=(all_deciles(enc,layer,x[test],h[test],sign*u,sel,batch).reshape(len(test),16,9)[:,:8]
           for sign in (1,-1))
    obs=five_components(np.take((hi-lo)/2,[0,2,4,6,8],axis=-1))
    pred=five_components(np.einsum("nhqd,nd->nhq",j[test],u))
    fd=np.linalg.norm(obs-pred,axis=(0,1))/np.linalg.norm(obs,axis=(0,1))
    if not np.all(np.isfinite(fd)) or fd.max()>.10:
        raise ValueError(f"Failed curvature check {fd}")
    scratch.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(scratch/(name+"_five.npz"),ids=ids,split=labels,jac=j,base=base,radius=rad,fd_error=fd)
    print(name,fd,flush=True)


def run(source,scratch,out,batch):
    target=next(n for n in MODELS if n!=source)
    _,_,atoms,dec,_,_=source_material(source)
    with np.load(scratch/(source+"_five.npz")) as s,np.load(scratch/(target+"_five.npz")) as t:
        ids,labels=s["ids"],s["split"]
        np.testing.assert_array_equal(ids,t["ids"])
        ev=labels!="cal"
        def reformat(f):
            channels=five_components(np.moveaxis(f["jac"].astype(float),2,-1))
            return np.moveaxis(channels,-1,1)
        sj,tj=reformat(s),reformat(t)
        base={source:s["base"][ev],target:t["base"][ev]}
        radius=float(t["radius"])
        fd={n:f["fd_error"] for n,f in ((source,s),(target,t))}
    scale,gain=weights(sj[~ev])
    template=sj[ev]@dec
    hs,_,_=read_materials(BLIND,source)
    sae=load_sae_checkpoint(str(dictionary_path(source)))
    with torch.inference_mode():
        z=sae.encode(torch.from_numpy(hs.mean(axis=1))).numpy()[:,atoms]
    with np.load(BLIND/"data.npz") as f:
        x=f["x"]
        np.testing.assert_array_equal(f["ids"],ids[ev])
    record=dict(ids=ids[ev], split=labels[ev], atoms=atoms, active=z>0, scales=scale, gains=gain,
                linear=(-z[:,:,None,None]*template.transpose(0,3,2,1)).astype("f4"),
                fd_source=fd[source],fd_target=fd[target])
    for name in (source,target):
        h,pos,rows=read_materials(BLIND,name)
        np.testing.assert_array_equal(rows,ids[ev])
        enc=adapter_for(name)
        sel=last_window_slice(enc,pos)
        args=(enc,MODELS[name][-1],sel,x,h,base[name].reshape(len(x),-1),z)
        if name==source:
            ys,_=apply(*args,dec,np.inf,batch,all_deciles)
            record["source"]=ys.reshape(len(x),len(atoms),16,9)
        else:
            for nc,arm in ((3,"three"),(5,"five")):
                w=solve_adaptive(stack(tj[ev,:nc],scale[:nc],gain[:nc]),stack(template[:,:nc],scale[:nc],gain[:nc]))
                ys,record[arm+"_capped"]=apply(*args,w,radius,batch,all_deciles)
                record[arm]=ys.reshape(len(x),len(atoms),16,9)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(source+"_normal.npz"),**record)
    print(source,"finished",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("geometry","run"))
    p.add_argument("name",choices=MODELS)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=8)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.mode=="geometry":
        geometry(a.name,a.scratch,a.batch)
    else:
        run(a.name,a.scratch,a.out,a.batch)
