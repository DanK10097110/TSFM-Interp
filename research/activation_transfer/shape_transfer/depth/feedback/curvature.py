"""Compare nonlinear local response across norm-matched directions and replay selected downstream states."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.feedback.data import OUT,SCRATCH
from research.activation_transfer.shape_transfer.depth.feedback.run import limit
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL,boost_shape,solve_adaptive
from tsfm_lens.extraction.hooks import _primary,_rebuild,token_patch


def trace(enc,site,sel,x,h,v,j):
    """A single-example, all-token selection at each following residual output."""
    sites=enc.all_layer_names()[10:]
    def pass_at(alpha,monitor=False,replace=None):
        states={}
        def hook(l):
            def store(module,args,output):
                if l in states:
                    raise ValueError("Repeated block")
                live=_primary(output)
                states[l]=live.detach().numpy().copy()
                if replace and l==replace[0]:
                    new=torch.from_numpy(replace[1]).to(live)
                    return _rebuild(output,new)
            return store
        with ExitStack() as stack:
            stack.enter_context(token_patch(enc.module,site,sel,torch.from_numpy(h+alpha*v[None,None].astype("f4"))))
            if monitor or replace:
                for l in sites:
                    stack.enter_context(enc.module.get_submodule(l).register_forward_hook(hook(l)))
            y=enc.predict(x,8,[.5])["quantiles"][0,:,0].astype(float)
        if monitor and len(states)!=len(sites):
            raise ValueError("Missing block activation")
        return y,states
    ordinary,clean=pass_at(0.,True)
    full,actual=pass_at(1.,True)
    predicted=j@v
    samples={}
    for alpha in np.linspace(0,1,11):
        y=pass_at(alpha)[0]-ordinary
        samples[f"{alpha:.1f}"]=dict(actual=y.tolist(),linear=(alpha*predicted).tolist())
    st={}
    for step in (.01,.02):
        _,plus=pass_at(step,True)
        _,minus=pass_at(-step,True)
        st[step]={l:(plus[l].astype(float)-minus[l])/(2*step) for l in sites}
    entries=[]
    for l in sites:
        exact=pass_at(1.,replace=(l,actual[l]))[0]
        np.testing.assert_allclose(exact,full,atol=1e-6,rtol=1e-6)
        tan=st[.01][l]
        curv=actual[l].astype(float)-clean[l]-tan
        lin=pass_at(1.,replace=(l,(clean[l]+tan).astype("f4")))[0]-ordinary
        check=pass_at(1.,replace=(l,(clean[l]+st[.02][l]).astype("f4")))[0]-ordinary
        entries.append(dict(layer=l,relative_state_nonlinearity=float(np.linalg.norm(curv)/np.linalg.norm(tan)),
                            relative_tangent_step_sensitivity=float(np.linalg.norm(tan-st[.02][l])/np.linalg.norm(tan)),
                            output_after_linear_replacement=lin.tolist(),other_step_output=check.tolist(),
                            remaining_nonlinear_output_rms=float(np.sqrt(np.mean((lin-predicted)**2)))))
    return dict(dose_curve=samples,layers=entries,full_nonlinear_rms=float(np.sqrt(np.mean((full-ordinary-predicted)**2))))


def run(scratch,out,batch):
    with np.load(scratch/"data.npz") as f:
        ids,x,split=f["ids"],f["x"],f["split"]
    with np.load(scratch/"bolt_3.npz") as s,np.load(scratch/"timesfm_10.npz") as t:
        z,atoms=s["codes"],s["atoms"]
        r=s["jac"].astype(float)@s["dec"].astype(float)
        j=t["jac"].astype(float)
        h,layer,pos,rad=t["states"],str(t["layer"]),t["pos"],float(t["radius"])
    with np.load(FULL/"results/bolt.npz") as f:
        weight=float(f["shape_weight"])
    k=int(np.flatnonzero(atoms==5876)[0])
    used=np.flatnonzero(z[:,k]>0)
    w=solve_adaptive(boost_shape(j[used],weight),boost_shape(r[used],weight))
    nat=limit(-z[used,k,None]*w[:,:,k],.2*rad)
    lengths=np.linalg.norm(nat,axis=1)
    unit=-w/np.linalg.norm(w,axis=1,keepdims=True)
    rnd=np.random.default_rng(701044).normal(size=(j.shape[-1],4))
    rnd/=np.linalg.norm(rnd,axis=0)
    dirs=np.concatenate((unit, -unit[:,:,k,None],np.broadcast_to(rnd,(len(used),*rnd.shape))),axis=2)
    enc=adapter_for("timesfm")
    sel=last_window_slice(enc,pos)
    xx,hh=x[used],h[used]
    clean=forecast16(enc,layer,xx,hh,np.zeros_like(nat).astype("f4"),sel,batch)[:,:8].astype(float)
    pred=np.einsum("nhd,ndv,n->nvh",j[used],dirs,lengths)
    record=dict(ids=ids[used],split=split[used],source_rows=atoms,named_index=k,
                norm_fraction=lengths/rad,linear=pred.astype("f4"))
    for tag,dose in (("full",1.),("half",.5)):
        effect=[]
        for c in range(dirs.shape[2]):
            delta=dirs[:,:,c]*(dose*lengths[:,None])
            effect.append(forecast16(enc,layer,xx,hh,delta.astype("f4"),sel,batch)[:,:8].astype(float)-clean)
        record[tag]=np.stack(effect,axis=1).astype("f4")
    with np.load(out/"block10.npz") as f:
        np.testing.assert_allclose(record["full"][:,k],f["open"][used,k,:8],atol=3e-4,rtol=.004)
    np.savez_compressed(out/"norm_matched.npz",**record)
    summary={}
    for d in ("original","new"):
        entry={}
        for label,range_test in (("large",lengths/rad>=.05),("small",lengths/rad<.05)):
            mask=(split[used]==d)&range_test
            if not mask.any():
                continue
            cases=dict(n=int(mask.sum()))
            for arm,dose in (("full",1.),("half",.5)):
                err=record[arm][mask]-pred[mask]*dose
                mag=np.sqrt(np.mean(err**2,axis=2))
                signal=np.sqrt(np.mean((pred[mask]*dose)**2,axis=2))
                block={}
                for label2,cols in (("named",[k]),("other_source",[i for i in range(len(atoms)) if i!=k]),
                                   ("opposite",[len(atoms)]),("random",list(range(len(atoms)+1,dirs.shape[2])))):
                    block[label2]=dict(median_rms_error=float(np.median(mag[:,cols])),
                                       rms_ratio=float(np.linalg.norm(mag[:,cols])/np.linalg.norm(signal[:,cols])))
                block["named_larger_than_other_fraction"]=float(np.mean(mag[:,k,None]>mag[:,[i for i in range(len(atoms)) if i!=k]]))
                cases[arm]=block
            entry[label]=cases
        summary[d]=entry
    cases={}
    for id_ in ("machines/fastStorage_801","machines/fastStorage_697"):
        i=int(np.flatnonzero(ids[used]==id_)[0])
        cases[id_]=trace(enc,layer,sel,xx[i:i+1],hh[i:i+1],nat[i],j[used[i]])
    (out/"curvature.json").write_text(json.dumps(dict(norm_matching=summary,cases=cases),indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.scratch,a.out,a.batch)
