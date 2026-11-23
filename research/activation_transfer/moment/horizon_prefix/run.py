"""Score a common lead prefix across short forecasts requesting one to four future patches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.data import OUT as QUALITY
from research.activation_transfer.moment.forecast_quality.score import item_metrics,boot_weights
from research.activation_transfer.moment.head_test.data import OUT as HEADS,SCRATCH
from research.activation_transfer.moment.head_test.capture import adapter,batch_features
from research.activation_transfer.moment.horizon_identity.factorial import collect
from research.activation_transfer.moment.horizon_identity.score import fractional
from research.activation_transfer.moment.token_paths.geometry import CACHE

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
WEATHER=QUALITY.parent/"post_release/results"
HORIZONS=(8,16,24,32)


def capture(domain,horizon,batch):
    root=QUALITY if domain=="old" else WEATHER
    with np.load(root/"panel.npz") as f:
        ids=f["id" if domain=="old" else "ids"]
        x=f["context"].reshape(-1,480)
    folder=CACHE/"horizon_prefix"/domain/f"h{horizon}"
    folder.mkdir(exist_ok=True,parents=True)
    model=adapter("large",0)
    for lo in range(0,len(x),128):
        path=folder/f"{lo}.npz"
        if path.exists():continue
        rows=[batch_features(model,x[i:min(i+batch,lo+128)],horizon=horizon) for i in range(lo,min(len(x),lo+128),batch)]
        rec={k:np.concatenate([a[k] for a in rows]) for k in rows[0]}
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,**rec)
        tmp.replace(path)
        print(domain,horizon,lo,len(x),flush=True)
    blocks=[]
    for lo in range(0,len(x),128):
        with np.load(folder/f"{lo}.npz") as f:blocks.append(dict(f))
    v={k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]}
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/f"{domain}_{horizon}.npz",ids=ids,h=v["h"][:,0],mu=v["mu"],sd=v["sd"],native=v["pub"][:,0])


def load_forecasts(domain,horizon):
    if horizon<=16:
        name="test" if domain=="old" else "external"
        with np.load(SCRATCH/"large_0"/f"{name}{horizon}.npz") as f:
            ids,h,mu,sd,native=f["ids"],f["h"][:,0],f["mu"],f["sd"],f["pub"][:,0]
    else:
        with np.load(OUT/f"{domain}_{horizon}.npz") as f:ids,h,mu,sd,native=(f[k] for k in ("ids","h","mu","sd","native"))
    pred={"published":native}
    with np.load(HEADS/"large_0_heads.npz") as f:
        for head in ("interior","terminal"):
            pred[head]=(h@f[head+"_hidden_weight"]+f[head+"_hidden_bias"])*sd[:,None]+mu[:,None]
    return ids,{k:v.reshape(len(ids),-1,8) for k,v in pred.items()}


def summarize():
    summary={};policy={};train={}
    for dom,root in (("old",QUALITY),("noaa",WEATHER)):
        with np.load(root/"panel.npz") as f:
            x,y=f["context"],f["future"][:,:,:8]
            ids=f["id" if dom=="old" else "ids"]
            ds,split=(f["domain"],f["split"]) if dom=="old" else (np.repeat("noaa",len(x)),np.repeat("holdout",len(x)))
        by={}
        for horizon in HORIZONS:
            k,pred=load_forecasts(dom,horizon)
            np.testing.assert_array_equal(k,ids)
            for head,v in pred.items():
                vals=item_metrics(v,y,x)["mase1"] if dom=="old" else np.mean(np.abs(v-y),axis=(1,2))
                by.setdefault(head,{})[horizon]=vals
        for head,v in by.items():
            for group in np.unique(ds):
                group=str(group)
                hold=(ds==group)&(split=="holdout")
                if dom=="old":
                    cal=(ds==group)&~hold
                    assert cal.sum()==24 and hold.sum()==96
                    raw=np.array([v[h][cal].mean() for h in HORIZONS])
                    pref=HORIZONS[raw.argmin()]
                    train.setdefault(head,[]).append(raw/raw[0])
                else:
                    raw=np.mean(train[head],axis=0)
                    pref=HORIZONS[raw.argmin()]
                w=boot_weights("prefix-decision-"+group,len(np.flatnonzero(hold)))
                def stat(a):return [float(a.mean()),*np.quantile(w@a,[.025,.975]).tolist()]
                z={str(h):dict(error=stat(v[h][hold]),delta=stat((v[h]-v[8])[hold])) for h in HORIZONS}
                z.update(choice=pref,calibration_losses=raw.tolist(),chosen_minus_direct=z[str(pref)]["delta"])
                summary.setdefault(head,{})[group]=z
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


def factorial(size,batch):
    with np.load(WEATHER/"panel.npz") as f:ids,x,y=f["ids"],f["context"],f["future"][:,:,:8]
    shape=y.shape;x=x.reshape(-1,480);y=y.reshape(-1,8)
    a=adapter(size,0)
    result={}
    for arm in ("native8","native16","position8","one_mask","both8"):
        h=8 if arm=="native8" else 16
        result[arm]=collect(a,x,h,arm,batch)[0][:,:8]
    np.testing.assert_allclose(result["native8"],result["both8"],atol=5e-5,rtol=2e-5)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/f"noaa_{size}_factorial.npz",ids=ids,**result)
    rng=boot_weights("noaa-prefix-factors",len(ids))
    def stats(arr):
        vals=arr.reshape(len(ids),-1).mean(1)
        return [float(vals.mean()),*np.quantile(rng@vals,[.025,.975]).tolist()]
    delta=result["native16"]-result["native8"]
    pos=.5*((result["one_mask"]-result["both8"])+(result["native16"]-result["position8"]))
    mask=.5*((result["position8"]-result["both8"])+(result["native16"]-result["one_mask"]))
    inter=result["native16"]-result["one_mask"]-result["position8"]+result["both8"]
    d={}
    for arm in result:d[arm+"_mae_C"]=stats(np.abs(result[arm]-y))
    for name,v in (("total",delta),("position",pos),("attention",mask),("interaction",inter)):
        rms=np.sqrt(np.mean(v.astype(float)**2,axis=1))
        d[name+"_rms_C"]=stats(rms)
        d[name+"_rms_stdev"]=stats(rms/x.std(1))
    for name,v in (("position",pos),("attention",mask)):
        d[name+"_share"]=fractional(v.reshape(len(ids),-1),delta.reshape(len(ids),-1),rng)
    d["paired_H16_prefix_minus_H8_MAE_C"]=stats(np.abs(result["native16"]-y)-np.abs(result["native8"]-y))
    d["joint_replay_max_C"]=float(np.max(np.abs(result["both8"]-result["native8"])))
    (OUT/f"noaa_{size}_factorial.json").write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("capture","score","factorial"));p.add_argument("--domain",choices=("old","noaa"))
    p.add_argument("--horizon",type=int,choices=(24,32));p.add_argument("--size",choices=("small","large"),default="large")
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="capture":capture(a.domain,a.horizon,a.batch)
    elif a.stage=="score":summarize()
    else:factorial(a.size,a.batch)
