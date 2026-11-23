"""Locate the first crossing along actual hidden knockouts and trace a preselected example."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.dictionary_prior.refit.confirmation import OUT,TMP
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.run import CHECKPOINTS
from tsfm_lens.sae.train import load_sae_checkpoint


def same_support(pre,clean,sae):
    return ((sae.sparsify(pre)>0)==clean).all(1)


@torch.inference_mode()
def run(model):
    src=load_sae_checkpoint(str(CHECKPOINTS/f"seed1_{model}.pt"))
    dst=load_sae_checkpoint(str(CHECKPOINTS/f"seed2_{model}.pt"))
    with np.load(TMP/"data.npz") as f:
        ids,x,split=f["ids"],f["x"],f["split"]
    with np.load(OUT/f"confirm_{model}_1.npz") as f:
        np.testing.assert_array_equal(f["ids"],ids)
        active,atoms=f["active"],f["atoms"]
    overlap=json.loads((OUT/"confirm_overlap.json").read_text())
    np.testing.assert_array_equal(overlap["ids"],ids)
    strict=np.array(overlap["frozen_order_mask"],bool)&(split=="daily")
    masks=active&strict[:,None]&(active[strict].sum(0)>=8)[None,:]
    rows,cols=np.nonzero(masks)
    state,pos,check=read_materials(TMP,model)
    np.testing.assert_array_equal(check,ids)
    hs=torch.from_numpy(state.mean(1))
    strengths=src.encode(hs).numpy()
    v=-torch.from_numpy(strengths[rows,atoms[cols],None])*src.W_dec[atoms[cols]]
    # The affine preactivation is evaluated in 64-bit during bisection to avoid rounding at ties.
    clean=dst.pre_activations(hs[rows]).double()
    slope=(v@dst.W_enc).double()
    initial=dst.sparsify(clean)>0
    crossed=~same_support(clean+slope,initial,dst)
    lo=torch.zeros(len(rows),dtype=torch.float64)
    hi=torch.ones_like(lo)
    for _ in range(22):
        mid=(lo+hi)/2
        still=same_support(clean+mid[:,None]*slope,initial,dst)
        lo=torch.where(still,mid,lo)
        hi=torch.where(still,hi,mid)
    first=hi.numpy()
    ix=None
    for q in np.flatnonzero(crossed.numpy()&(first>=.05)&(first<=.90)):
        h=hs[rows[q]:rows[q]+1]
        t=first[q]
        z1,z2=[dst.encode(h+(t+s)*v[q]) for s in (-.001,.001)]
        if int(((z1>0)&(z2==0)).sum())==int(((z2>0)&(z1==0)).sum())==1:
            ix=q
            break
    if ix is None:
        raise ValueError("No qualifying geometry-only example")
    r=rows[ix];t=float(first[ix]);h=hs[r:r+1]
    lower,upper=t-.001,t+.001
    fixed=(dst.encode(h)>0)
    # The actual FP32 encoder can place a tie slightly differently than the affine calculation.
    for _ in range(18):
        mid=(lower+upper)/2
        if torch.equal(dst.encode(h+mid*v[ix])>0,fixed): lower=mid
        else: upper=mid
    t=(upper+lower)/2
    z0=dst.encode(h)
    steps=np.array([.01,.001,.0001,.00001])
    grid=np.unique(np.r_[np.linspace(t-.04,t+.04,41),t-steps,t+steps])
    offset=torch.tensor(grid[:,None],dtype=torch.float32)*v[ix]
    pre=dst.pre_activations(h+offset)
    hard=dst.sparsify(pre);locked=pre.relu()*fixed
    zlo,zhi=[dst.encode(h+(t+sign*1e-5)*v[ix]) for sign in (-1,1)]
    before=torch.nonzero((zlo>0)&(zhi==0))[:,1].tolist()
    after=torch.nonzero((zhi>0)&(zlo==0))[:,1].tolist()
    if len(before)!=1 or len(after)!=1:
        raise ValueError("Selected FP32 example is unresolved at the smallest checked step")
    with np.load(DEPTH/f"{model}_{3 if model=='bolt' else 10}.npz") as f:
        cap=.2*float(f["radius"])
    vectors={"dense":offset,"hard":(hard-z0)@dst.W_dec,"locked":(locked-z0)@dst.W_dec}
    enc=adapter_for(model)
    sel=last_window_slice(enc,pos)
    xx=np.repeat(x[r:r+1],len(grid),axis=0)
    hh=np.repeat(state[r:r+1],len(grid),axis=0)
    rec={}
    for arm,vector in vectors.items():
        a=vector.numpy()
        length=np.linalg.norm(a,axis=1)
        edited=a*np.minimum(1.,cap/np.maximum(length,1e-12))[:,None]
        y=forecast16(enc,MODELS[model][-1],xx,hh,edited,sel,24)[:,:8].astype(float)
        rec[arm]=dict(forecast=y.tolist(),crossings={})
        for u in steps:
            i,j=np.searchsorted(grid,[t-u,t+u])
            rec[arm]["crossings"][str(u)]=dict(hidden=float(np.linalg.norm(edited[j]-edited[i])),
                                                forecast_rms=float(np.mean((y[j]-y[i])**2)**.5))
    return dict(n=92,firings=len(rows),ever_cross=int(crossed.sum()),
                crossing_fraction_quantiles=np.quantile(first[crossed.numpy()],[.1,.5,.9]).tolist(),
                series=str(ids[r]),donor=int(atoms[cols[ix]]),leaving=before[0],entering=after[0],first=t,
                dose=grid.tolist(),leaving_strength=hard[:,before[0]].tolist(),entering_strength=hard[:,after[0]].tolist(),
                arms=rec)


if __name__=="__main__":
    torch.set_num_threads(2)
    data={m:run(m) for m in ("bolt","timesfm")}
    (OUT/"boundary_trace.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
