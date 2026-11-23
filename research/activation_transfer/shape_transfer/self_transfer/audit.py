"""Independent additive-hook replay with singleton Jacobians and an SVD form of the balanced inverse."""
import argparse
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH_FIT, histories
from research.activation_transfer.shape_transfer.self_transfer.run import HERE, NAMES, SCRATCH, materials
from research.activation_transfer.shape_transfer.third_receiver import timer


@contextmanager
def additive(module,layer,selector,vector):
    def shift(mod,args,out):
        h=out[0] if isinstance(out,(tuple,list)) else out
        v=vector if torch.is_tensor(vector) else torch.from_numpy(np.asarray(vector,dtype="f4"))
        # Out-of-place update on the live forward tensor; no cached hidden state is used here.
        idx=selector(h.shape[1])
        new=torch.cat((h[:,:idx.start] if idx.start is not None else h[:,:0],
                       h[:,idx]+v.reshape(len(h),1,-1),
                       h[:,idx.stop:] if idx.stop is not None else h[:,:0]),dim=1)
        if isinstance(out,tuple):
            return (new,*out[1:])
        if isinstance(out,list):
            return [new,*out[1:]]
        return new
    handle=module.get_submodule(layer).register_forward_hook(shift)
    try:
        yield
    finally:
        handle.remove()


def live_jac(enc,layer,selector,x,width):
    v=torch.zeros((len(x),width),requires_grad=True)
    with torch.enable_grad(),additive(enc.module,layer,selector,v):
        y=enc.predict_differentiable(x,8)
        rows=[torch.autograd.grad(y[:,k].sum(),v,retain_graph=k<7)[0] for k in range(8)]
    return torch.stack(rows,dim=1).numpy().astype(float)


def inverse(j,goal,weight):
    # Explicit H×H mean-preserving metric, followed by an SVD ridge inverse.
    h=len(j)
    balance=weight*np.eye(h)+(1-weight)*np.ones((h,h))/h
    u,s,vt=np.linalg.svd(balance@j,full_matrices=False)
    filt=s/(s*s+.001*s.max()**2)
    return vt.T@(filt*(u.T@(balance@goal)))


def run(out):
    all_ids,x,split=histories()
    use=~np.isin(split,["cal","train"])
    x,ids,split=x[use],all_ids[use],split[use]
    models={}
    for n in NAMES:
        if n=="timer":
            enc,selector,_=timer.adapter_and_selector(x)
            path=SCRATCH/"timer.npz"
            layer=timer.LAYER
        else:
            enc=adapter_for(n)
            path=DEPTH_FIT/f"{n}_{3 if n=='bolt' else 10}.npz"
            with np.load(path) as f:
                selector=last_window_slice(enc,f["positions"])
            layer=MODELS[n][-1]
        for p in enc.module.parameters():
            p.requires_grad_(False)
        with np.load(path) as f:
            dim=f["jac"].shape[-1]
            rad=float(f["radius"])
        models[n]=(enc,layer,selector,dim,rad)
    records=[]
    for s in NAMES:
        src_ids,code,atoms,dec,_,weight=materials(s,SCRATCH)
        code=code[-len(ids):]
        np.testing.assert_array_equal(src_ids[-len(ids):],ids)
        source_model,source_layer,source_select,dsource,_=models[s]
        for domain in np.unique(split):
            rng=np.random.default_rng({"daily":27,"machines":31,"extra_daily":88,"extra_monash_traffic":93}[str(domain)])
            present=np.argwhere((code>0)&(split==domain)[:,None])
            if not len(present):
                continue
            ix,feat=present[rng.integers(len(present))]
            obs=x[ix:ix+1]
            js=live_jac(source_model,source_layer,source_select,obs,dsource)[0]
            z=float(code[ix,feat])
            d=-z*dec[:,feat]
            goal=js@d
            for target in NAMES:
                mod,site,sel,dim,radius=models[target]
                jt=js if target==s else live_jac(mod,site,sel,obs,dim)[0]
                v=inverse(jt,goal,weight)
                v*=min(1.,.2*radius/max(np.linalg.norm(v),1e-15))
                vectors={"adaptive":v}
                if target==s:
                    # A Moore-Penrose inverse of the *unweighted*, undamped Jacobian supplies the visible portion.
                    visible=np.linalg.pinv(js)@goal
                    vectors.update(native=d,projection=visible,null=d-visible)
                base=mod.predict(obs,16,[.5])["quantiles"][0,:,0]
                with np.load(HERE/"results"/f"{s}_to_{target}.npz") as f:
                    np.testing.assert_array_equal(ids,f["ids"])
                    for arm,direction in vectors.items():
                        with additive(mod.module,site,sel,direction[None]):
                            delta=mod.predict(obs,16,[.5])["quantiles"][0,:,0]-base
                        ref=f[arm][ix,feat]
                        np.testing.assert_allclose(delta,ref,rtol=.004,atol=3e-4)
                        records.append(dict(source=s,target=target,domain=str(domain),id=str(ids[ix]),atom=int(atoms[feat]),
                                            arm=arm,max_abs=float(np.max(abs(delta-ref)))))
    result=dict(n=len(records),max_abs=max(r["max_abs"] for r in records),records=records)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result["n"],result["max_abs"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results/audit.json")
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.out)
