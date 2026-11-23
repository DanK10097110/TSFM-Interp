"""Transport the state change produced by the same observed temperature edit in two MOMENT checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH,hidden
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER
from research.activation_transfer.moment.forecast_quality.score import boot_weights
from research.activation_transfer.moment.input_paths.run import modify
from research.activation_transfer.moment.token_paths.geometry import CACHE, layers, load
from tsfm_lens.extraction.hooks import ActivationCatcher

OUT=Path(__file__).resolve().parent/"results"
EDITS=("recent","ramp")


def prepare(size,batch):
    with np.load(WEATHER/"panel.npz") as f:stations,x=f["ids"],f["context"].reshape(-1,480)
    a=load(size);site=layers(a)["middle"]
    data=dict(ids=stations)
    for k,h in [("base",x)]+[(k,modify(x,k)) for k in EDITS]:
        states=[];ys=[]
        for i in range(0,len(h),batch):
            with ActivationCatcher(a.module,[site]) as hook:
                ys.append(a.predict(h[i:i+batch],16,[.5])["point"])
                states.append(hook.collect()[site][:,61].numpy())
        data[k+"_state"]=np.concatenate(states);data[k+"_forecast"]=np.concatenate(ys)
        if k=="base":continue
        delta=data[k+"_state"]-data["base_state"]
        truth=[]
        for i in range(0,len(h),batch):
            with addition(a,delta[i:i+batch]):
                truth.append(a.predict(x[i:i+batch],16,[.5])["point"])
        data[k+"_true_site"]=np.concatenate(truth)-data["base_forecast"]
    np.savez_compressed(OUT/f"{size}_native.npz",**data)


def rotation(n):
    order=np.random.default_rng(22112391).permutation(n)
    mapping=np.empty(n,int);mapping[order]=np.roll(order,1)
    assert np.all(mapping!=np.arange(n))
    return (4*mapping[:,None]+np.arange(4)).ravel()


def skill(got,ref,w):
    n=w.shape[1]
    y=ref.reshape(n,-1).astype(float);z=got.reshape(n,-1).astype(float)
    den=(y*y).sum(1);loss=((z-y)**2).sum(1)
    if den.sum()<=1e-14:raise ValueError("Indistinguishable effect")
    return [float(1-loss.sum()/den.sum()),*np.quantile(1-(w@loss)/(w@den),[.025,.975]).tolist()]


def ordinary(source,target,W):
    with (MATCH/f"{source}_to_{target}_map.json").open() as f:meta=json.load(f)
    with np.load(CACHE/source/"train.npz") as f:lookup={s:i for i,s in enumerate(f["ids"])}
    a,b=hidden(source),hidden(target)
    tr=[lookup[s] for s in meta["training_ids"]];te=[lookup[s] for s in meta["blind_map_ids"]]
    sa=a[tr].mean((0,1),dtype=float);sb=b[tr].mean((0,1),dtype=float)
    bias=sb-sa@W
    true=b[te].astype(float);pred=a[te]@W+bias
    position=b[tr].mean(0,dtype=float)
    def fraction(z,y,base):return float(1-np.sum((z-y)**2)/np.sum((base-y)**2))
    dy=true-np.roll(true,1,axis=0);dz=pred-np.roll(pred,1,axis=0)
    return dict(ordinary=fraction(pred,true,sb),ordinary_last=fraction(pred[:,-1],true[:,-1],position[-1]),
                position_only=fraction(np.broadcast_to(position,true.shape),true,sb),
                between_history=fraction(dz,dy,0.),between_history_last=fraction(dz[:,-1],dy[:,-1],0.))


def run(source,target,batch):
    with np.load(WEATHER/"panel.npz") as f:stations,x=f["ids"],f["context"].reshape(-1,480)
    states={}
    for size in (source,target):
        with np.load(OUT/f"{size}_native.npz") as f:
            np.testing.assert_array_equal(f["ids"],stations);states[size]=dict(f)
    with np.load(MATCH/f"{source}_to_{target}_map.npz") as f:W=f["w"].astype(float)
    w=boot_weights("MOMENT-frozen-input-transfer",len(stations))
    wrong=rotation(len(stations));a=load(target)
    results={"ordinary_state":ordinary(source,target,W)}
    waves=dict(ids=stations,wrong=wrong)
    for edit in EDITS:
        donor=states[source][edit+"_state"]-states[source]["base_state"]
        delta=states[target][edit+"_state"]-states[target]["base_state"]
        projected=donor@W
        magn=lambda v:np.linalg.norm(v,axis=1)
        alt=projected[wrong]*(magn(projected)/magn(projected[wrong]))[:,None]
        exact_norm=projected*(magn(delta)/magn(projected))[:,None]
        vecs=dict(mapped=projected,other_station=alt,oracle_length=exact_norm)
        ref=states[target][edit+"_true_site"]
        rec=dict(true_RMS_C=float(np.sqrt(np.mean(ref.astype(float)**2))),
                 true_centered_RMS_C=float(np.sqrt(np.mean((ref-ref.mean(1,keepdims=True))**2))))
        waves[edit+"_truth"]=ref
        for name,v in vecs.items():
            folder=CACHE/"input_delta_transfer"/f"{source}_{target}"/edit/name
            folder.mkdir(exist_ok=True,parents=True)
            edits=[]
            for i in range(0,len(x),batch):
                path=folder/f"{i}_{batch}.npy"
                if not path.exists():
                    with addition(a,v[i:i+batch]):
                        y=a.predict(x[i:i+batch],16,[.5])["point"]
                    tmp=path.with_suffix(".tmp.npy");np.save(tmp,y);tmp.replace(path)
                edits.append(np.load(path))
            actual=np.concatenate(edits)-states[target]["base_forecast"]
            waves[edit+"_"+name]=actual
            rec[name]=dict(hidden=skill(v,delta,w),cosine=float(np.median(np.sum(v*delta,axis=1)/(magn(v)*magn(delta)))),
                           median_norm_ratio=float(np.median(magn(v)/magn(delta))))
        for name,eff in {**{k:waves[edit+"_"+k] for k in vecs},"donor_wave":states[source][edit+"_true_site"]}.items():
            out=rec.setdefault(name,{})
            for side,sl in (("early",slice(8)),("late",slice(8,16))):
                y=ref[:,sl];z=eff[:,sl]
                out[side+"_full"]=skill(z,y,w)
                out[side+"_shape"]=skill(z-z.mean(1,keepdims=True),y-y.mean(1,keepdims=True),w)
        results[edit]=rec
    np.savez_compressed(OUT/f"{source}_to_{target}.npz",**waves)
    (OUT/f"{source}_to_{target}.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","run"));p.add_argument("source",choices=("small","large"))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="prepare":prepare(a.source,a.batch)
    else:run(a.source,"small" if a.source=="large" else "large",a.batch)
