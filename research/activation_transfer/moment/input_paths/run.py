"""Factorial mediation of a native forecast's response to physical-unit input edits."""
import argparse
import itertools
import json
from math import factorial
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER
from research.activation_transfer.moment.forecast_quality.predict import adapter
from research.activation_transfer.moment.head_test.capture import scaling
from research.activation_transfer.moment.token_paths.geometry import layers
from research.activation_transfer.moment.forecast_quality.score import boot_weights
from tsfm_lens.extraction.hooks import ActivationCatcher

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
EDITS=("global","recent","ramp")
CELLS=tuple(itertools.product((0,1),repeat=3))


def modify(x,name):
    y=x.copy()
    if name=="global":y+=2
    elif name=="recent":y[:,-8:]+=2
    elif name=="ramp":y[:,-8:]+=np.linspace(-2,2,8)
    return y


def capture(a,x,site,batch):
    ys=[];states=[];shifts=[];scales=[]
    for k in range(0,len(x),batch):
        with ActivationCatcher(a.module,[site]) as hook:
            ys.append(a.predict(x[k:k+batch],16,[.5])["point"])
            states.append(hook.collect()[site].cpu().numpy())
        shift,scale=scaling(a.module.normalizer);shifts.extend(shift);scales.extend(scale)
    return np.concatenate(ys),np.concatenate(states),np.asarray(shifts),np.asarray(scales)


def routed(a,x,site,states,obs,future,with_stats=None,batch=16):
    ys=[]
    for k in range(0,len(x),batch):
        sl=slice(k,k+batch)
        def patch(_,inputs,output):
            v=output if isinstance(output,torch.Tensor) else output[0]
            result=v.clone()
            h=torch.as_tensor(states[sl],dtype=v.dtype,device=v.device)
            if obs:result[:,2:62]=h[:,2:62]
            if future:result[:,62:64]=h[:,62:64]
            return result if isinstance(output,torch.Tensor) else (result,*output[1:])
        hk=a.module.get_submodule(site).register_forward_hook(patch)
        def before_denorm(norm,args,kwargs):
            if kwargs.get("mode")=="denorm":
                um,us=with_stats
                norm.mean=torch.as_tensor(um[sl],device=norm.mean.device).to(norm.mean).view_as(norm.mean)
                norm.stdev=torch.as_tensor(us[sl],device=norm.stdev.device).to(norm.stdev).view_as(norm.stdev)
        hs=None if with_stats is None else a.module.normalizer.register_forward_pre_hook(before_denorm,with_kwargs=True)
        try:ys.append(a.predict(x[sl],16,[.5])["point"])
        finally:
            hk.remove()
            if hs is not None:hs.remove()
    return np.concatenate(ys)


def run(key,batch):
    with np.load(WEATHER/"panel.npz") as f:ids,x=f["ids"],f["context"].reshape(-1,480)
    a=adapter(key)
    OUT.mkdir(exist_ok=True)
    if not key.startswith("moment"):
        path=OUT/f"{key}.npz"
        rec={"ids":ids}
        def direct(v):return np.concatenate([a.predict(v[k:k+batch],16,[.5])["quantiles"][...,0] for k in range(0,len(v),batch)])
        y0=direct(x);rec["base"]=y0
        for arm in EDITS:rec[arm]=direct(modify(x,arm))-y0
        np.savez_compressed(path,**rec);return
    site=layers(a)["middle"]
    y0,h0,mu0,sd0=capture(a,x,site,batch)
    norm=a.module.normalizer
    w=norm.affine_weight.detach().item()+norm.eps**2 if norm.affine else 1.
    b=norm.affine_bias.detach().item() if norm.affine else 0.
    rec=dict(ids=ids,base=y0)
    for name in EDITS:
        dest=OUT/f"{key}_{name}.npz"
        if dest.exists():continue
        pos=modify(x,name)
        y1,h1,mu1,sd1=capture(a,pos,site,batch)
        rec.update(true=y1-y0)
        for obs,fut in itertools.product((0,1),repeat=2):
            pred=routed(a,x,site,h1,obs,fut,batch=batch) if (obs or fut) else y0
            rec[f"cell_{obs}{fut}0"]=pred-y0
            rec[f"cell_{obs}{fut}1"]=(pred-mu0[:,None])*(sd1/sd0)[:,None]+mu1[:,None]-y0
        err=float(np.max(np.abs(rec["cell_111"]-rec["true"])))
        if err>2e-4:raise ValueError(f"All-path replay failed by {err} Celsius")
        probe=np.arange(0,len(x),max(1,len(x)//16))[:16]
        physical=routed(a,x[probe],site,h1[probe],True,True,
                        with_stats=((mu1+b*sd1)[probe],(sd1*w)[probe]),batch=batch)
        np.testing.assert_allclose(physical,y1[probe],atol=1e-4,rtol=5e-5)
        rec["physical_replay_max"]=float(np.max(np.abs(physical-y1[probe])))
        for label,sl in (("observed",slice(2,62)),("future",slice(62,64))):
            num=np.linalg.norm(h1[:,sl]-h0[:,sl],axis=(1,2))
            rec[label+"_state_motion"]=num/np.linalg.norm(h0[:,sl],axis=(1,2))
        np.savez_compressed(dest,**rec)
        print(key,name,err,flush=True)


def shapley(record):
    parts=[]
    for axis in range(3):
        effect=np.zeros_like(record["true"],dtype=float)
        for c in CELLS:
            if c[axis]:continue
            other=list(c);other[axis]=1
            before=record["cell_"+"".join(map(str,c))]
            after=record["cell_"+"".join(map(str,other))]
            k=sum(c);weight=factorial(k)*factorial(2-k)/6
            effect+=weight*(after-before)
        parts.append(effect)
    np.testing.assert_allclose(sum(parts),record["cell_111"]-record["cell_000"],atol=2e-6)
    return parts


def report():
    with np.load(WEATHER/"panel.npz") as f:ids=f["ids"]
    w=boot_weights("noaa-input-path",len(ids))
    overall={};cross={}
    def stats(v):
        per=v.reshape(len(ids),-1).mean(1)
        return [float(np.mean(per)),*np.quantile(w@per,[.025,.975]).tolist()]
    for edit in EDITS:
        waves={}
        for model in ("moment_small","moment_large"):
            with np.load(OUT/f"{model}_{edit}.npz") as f:d=dict(f)
            np.testing.assert_array_equal(d["ids"],ids)
            parts=shapley(d);truth=d["true"]
            waves[model]=truth
            cell=dict(replay_max=float(np.max(abs(d["cell_111"]-truth))),native_physical_replay_max=float(d["physical_replay_max"]),
                      mean_relative_hidden_change={k:stats(d[k+"_state_motion"]) for k in ("observed","future")})
            for tag,sl in (("first",slice(8)),("second",slice(8,16))):
                y=truth[:,sl];energy=(y*y).reshape(len(ids),-1).mean(1)
                sub={"mean_response_C":stats(y),"rms_C":float(np.sqrt(energy.mean()))}
                for name,v in zip(("observed","future","normalization"),parts):
                    v=v[:,sl]
                    xy=(v*y).reshape(len(ids),-1).mean(1)
                    share=(w@xy)/(w@energy)
                    sub[name]=dict(gain=[float(xy.sum()/energy.sum()),*np.quantile(share,[.025,.975]).tolist()],
                                   mean_C=stats(v),rms_C=float(np.sqrt(np.mean(v**2))))
                cell[tag]=sub
            overall[model+"_"+edit]=cell
        for k in ("bolt","timesfm"):
            with np.load(OUT/f"{k}.npz") as f:
                np.testing.assert_array_equal(f["ids"],ids);waves[k]=f[edit]
        for k in waves:
            for ref in waves:
                if k==ref:continue
                cell={}
                for kind in ("full","shape"):
                    y,q=waves[ref].astype(float),waves[k].astype(float)
                    if kind=="shape":y=y-y.mean(1,keepdims=True);q=q-q.mean(1,keepdims=True)
                    e=(y*y).reshape(len(ids),-1).sum(1)
                    if np.sqrt(e.sum()/y.size)<1e-4:continue
                    sse=((q-y)**2).reshape(len(ids),-1).sum(1)
                    v=1-(w@sse)/(w@e)
                    cell[kind]=[float(1-sse.sum()/e.sum()),*np.quantile(v,[.025,.975]).tolist()]
                cross[k+"_vs_"+ref+"_"+edit]=cell
    (OUT/"summary.json").write_text(json.dumps(dict(mediation=overall,agreement=cross),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("moment_small","moment_large","bolt","timesfm","score"))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="score":report()
    else:run(a.stage,a.batch)
