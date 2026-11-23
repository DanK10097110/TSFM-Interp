"""Confirm a frozen receiver-only feature detector using disjoint weather stations and dates."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.match import dictionary, encode
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import OUT as DETECTOR,infer_doses,SEED,TARGET
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle
from research.activation_transfer.moment.horizon_prefix.new_dates.run import OUT as WEATHER
from research.activation_transfer.moment.token_paths.geometry import CACHE,layers,load
from tsfm_lens.extraction.hooks import ActivationCatcher

OUT=Path(__file__).resolve().parent/"results"
WORK=CACHE/"external_dictionary_dates"
ARMS=("donor","oracle","inferred","natural")


def panel():
    with np.load(WEATHER/"panel.npz") as f:
        take=f["cohort"]=="secondary"
        x=f["x"][take].reshape(-1,480);ids=f["ids"][take]
    dates=tuple(json.loads((WEATHER/"manifest.json").read_text())["origins"])
    assert x.shape==(385,480) and len(ids)==35 and len(dates)==11
    return ids,dates,x


def capture(batch):
    ids,dates,x=panel()
    a=load("large");site=layers(a)["middle"]
    states=[];base=[]
    for i in range(0,len(x),batch):
        with ActivationCatcher(a.module,[site]) as cap:
            base.append(a.predict(x[i:i+batch],16,[.5])["point"])
            states.append(cap.collect()[site].numpy())
    WORK.mkdir(exist_ok=True,parents=True)
    full=np.concatenate(states);np.save(WORK/"live_hidden.npy",full)
    i=a.patch_layout(480,16)[0].stop-1;assert i==61
    np.savez_compressed(OUT/"ordinary.npz",ids=ids,dates=dates,base=np.concatenate(base),last=full[:,i],sd=x.std(1),
                        input_sha256=sha256(x.tobytes()).hexdigest())


def policy(arm,hidden,fit):
    if arm in ("natural","inferred"):
        sae=dictionary("large",TARGET);codes=encode(sae,hidden)
        if arm=="inferred":return infer_doses(codes,fit)[1],fit["group"]
        return codes[:,fit["match"]],sae.W_dec.detach().numpy()[fit["match"]]
    source=dictionary("large",SEED)
    dose=encode(source,hidden)[:,fit["atoms"]]
    return dose,source.W_dec.detach().numpy()[fit["atoms"]] if arm=="donor" else fit["group"]


def apply_arm(arm,batch):
    ids,dates,x=panel()
    basefile=OUT/"ordinary.npz";fitfile=DETECTOR/"fit.npz"
    with np.load(basefile) as f:base,hidden,sd=f["base"],f["last"],f["sd"]
    with np.load(fitfile) as f:fit=dict(f)
    doses,directions=policy(arm,hidden,fit)
    r,c=np.nonzero(doses>0);moves=(-doses[r,c,None]*directions[c]).astype("f4")
    assert np.isfinite(moves).all()
    model=load("large");site=layers(model)["middle"]
    full=np.load(WORK/"live_hidden.npy",mmap_mode="r");np.testing.assert_array_equal(full[:,61],hidden)
    out=np.zeros((*doses.shape,16),dtype="f4")
    folder=WORK/arm;folder.mkdir(exist_ok=True,parents=True)
    for lo in range(0,len(r),80):
        dest=folder/f"{lo}.npz"
        sig=sha256(moves[lo:lo+80].tobytes()+r[lo:lo+80].tobytes()+c[lo:lo+80].tobytes()).hexdigest()
        if not dest.exists():
            arr=[]
            for j in range(lo,min(lo+80,len(r)),batch):
                ix=slice(j,min(j+batch,lo+80,len(r)))
                with cached_middle(model,full[r[ix]],site),addition(model,moves[ix]):
                    arr.append(model.predict(x[r[ix]],16,[.5])["point"]-base[r[ix]])
            tmp=dest.with_suffix(".tmp.npz");np.savez_compressed(tmp,y=np.concatenate(arr),signature=sig);tmp.replace(dest)
        with np.load(dest) as f:
            assert sig==str(f["signature"]);out[r[lo:lo+80],c[lo:lo+80]]=f["y"]
    # Independent FULL graph with direct hook and its own baseline (including the final partial batch).
    sel=np.unique(np.linspace(0,len(r)-1,min(12,len(r)),dtype=int));orig=model.predict(x[r[sel]],16,[.5])["point"]
    with addition(model,moves[sel]):changed=model.predict(x[r[sel]],16,[.5])["point"]
    delta=float(np.max(abs(changed-orig-out[r[sel],c[sel]])))
    assert delta<.00015,delta
    np.savez_compressed(OUT/f"{arm}.npz",ids=ids,dates=dates,atoms=fit["atoms"],dose=doses,wave=out,sd=sd,
                        native_max_C=delta,live_check_events=sel,fit_sha256=sha256(fitfile.read_bytes()).hexdigest())


def run_scores():
    ids,dates,x=panel();d={}
    for name in ARMS:
        with np.load(OUT/f"{name}.npz") as f:
            d[name]=dict(f);np.testing.assert_array_equal(ids,f["ids"]);np.testing.assert_array_equal(dates,f["dates"])
            assert np.isfinite(f["wave"]).all() and np.all(f["wave"][f["dose"]==0]==0)
    np.testing.assert_array_equal(d["oracle"]["dose"],d["donor"]["dose"])
    assert len({str(v["fit_sha256"]) for v in d.values()})==1
    atoms=d["donor"]["atoms"];active=d["donor"]["dose"]>0
    n,t=len(ids),len(dates);rng=np.random.default_rng(22112397)
    sw=rng.multinomial(n,np.full(n,1/n),3000);tw=rng.multinomial(t,np.full(t,1/t),3000)
    def weighted(z):
        return np.einsum("bs,stf,bt->bf",sw,z.reshape(n,t,len(atoms)),tw,optimize=True)
    def div(a,b):
        return np.divide(a,b,out=np.full(a.shape,np.nan,dtype=float),where=b>1e-16)
    report=dict(stations=n,dates=t,features=atoms.tolist(),donor_fires=active.sum(0).tolist(),
                detector_sha256=str(d["donor"]["fit_sha256"]),full_replay_max_C={k:float(v["native_max_C"]) for k,v in d.items()})
    for subset in ("all","donor_firing"):
        for portion,sl in (("all",slice(None)),("early",slice(8)),("late",slice(8,16))):
            for mode in ("centered","full","centered_C"):
                def transform(arr):
                    z=arr[:,:,sl].astype(float)
                    if mode!="centered_C":z=z/x.std(1)[:,None,None]
                    if mode!="full":z=z-z.mean(2,keepdims=True)
                    return z
                ref=transform(d["donor"]["wave"]);e=(ref**2).sum(2);total=e.sum(0);eb=weighted(e)
                out={};boots={};per={}
                for arm in ("zero","natural","inferred","oracle"):
                    v=np.zeros_like(ref) if arm=="zero" else transform(d[arm]["wave"])
                    error=((v-ref)**2).sum(2)
                    if subset=="donor_firing":error*=active
                    scores=1-div(error.sum(0),total);boot=np.nanmean(1-div(weighted(error),eb),axis=1)
                    per_date=np.nanmean(1-div(error.reshape(n,t,-1).sum(0),e.reshape(n,t,-1).sum(0)),axis=1)
                    p=float(np.nanmean(scores));boots[arm]=boot;per[arm]=per_date
                    out[arm]=dict(value=p,crossed_station_date_95=np.nanquantile(boot,[.025,.975]).tolist(),per_atom=scores.tolist(),
                                  per_date=per_date.tolist(),leave_one_atom=[float(np.nanmean(np.delete(scores,i))) for i in range(len(atoms))],
                                  false_positive_energy_per_atom=div(((v*v).sum(2)*(~active)).sum(0),total).tolist())
                for a,b in (("inferred","natural"),("oracle","inferred"),("oracle","natural")):
                    out[a+"_minus_"+b]=dict(value=out[a]["value"]-out[b]["value"],
                        crossed_station_date_95=np.nanquantile(boots[a]-boots[b],[.025,.975]).tolist(),positive_dates=int((per[a]>per[b]).sum()))
                report[subset+"_"+portion+"_"+mode]=out
    report["detection"]={}
    for arm in ("inferred","natural"):
        pos=d[arm]["dose"]>0;tp=(pos&active).sum(0)
        report["detection"][arm]=dict(precision=div(tp,pos.sum(0)).tolist(),recall=div(tp,active.sum(0)).tolist(),
                                     tp=tp.tolist(),fp=(pos&~active).sum(0).tolist(),fn=(~pos&active).sum(0).tolist())
    def finite(value):
        if isinstance(value,dict):return {key:finite(v) for key,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    (OUT/"summary.json").write_text(json.dumps(finite(report),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","score",*ARMS));p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="prepare":capture(a.batch)
    elif a.stage=="score":run_scores()
    else:apply_arm(a.stage,a.batch)
