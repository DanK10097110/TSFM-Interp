"""Run pinned official M4 cutoffs with each model's native missing-history convention."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.predict import adapter, NAMES
from research.activation_transfer.moment.head_test.capture import scaling
from research.activation_transfer.moment.head_test.data import OUT as HEADS
from tsfm_lens.extraction.hooks import ActivationCatcher

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SOURCE = Path("/tmp/tsfm_m4_calendar")
GROUPS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12)}
REVISION = "6c1067e5a57161249b17289a565178dc7a3fb3ca"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def choose(group):
    horizon, season = GROUPS[group]
    train_path = SOURCE / f"{group}-train.csv"
    test_path = SOURCE / f"{group}-test.csv"
    # Keep only the needed rows in memory while parsing the source's wide empty cells.
    with train_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        series = {}
        for key, *row in reader:
            y = np.asarray([float(v) for v in row if v != ""], dtype=float)
            if len(y) > season and np.isfinite(y).all() and np.mean(abs(y[season:]-y[:-season])) > 0:
                series[key] = y
    ids = sorted(series, key=lambda k: hashlib.sha256(("moment-calendar-20261122|"+k).encode()).digest())[:256]
    with test_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        need = set(ids)
        futures = {key: [float(v) for v in row if v != ""] for key,*row in reader if key in need}
    x = np.full((len(ids), 480), np.nan)
    scales = []
    for i,key in enumerate(ids):
        y = series[key]
        scales.append(np.mean(abs(y[season:]-y[:-season])))
        past = y[-480:]
        x[i,-len(past):] = past
    future = np.array([futures[key] for key in ids])
    assert future.shape == (len(x),horizon) and np.isfinite(future).all()
    last = np.repeat(x[:,-1:], horizon, axis=1)
    seasonal = x[:,np.arange(horizon)%season-season]
    first = x[np.arange(len(x)),np.argmax(np.isfinite(x), axis=1)]
    drift = last + ((x[:,-1]-first)/(np.isfinite(x).sum(1)-1))[:,None]*np.arange(1,horizon+1)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/f"{group}_panel.npz",ids=ids,x=x,truth=future,scale=scales,last=last,seasonal=seasonal,drift=drift)
    record = {"revision":REVISION,"train_sha256":digest(train_path),"test_sha256":digest(test_path),
              "eligible":len(series),"holdout":len(ids),"median_history":float(np.median([len(series[k]) for k in ids]))}
    (OUT/f"{group}_manifest.json").write_text(json.dumps(record, indent=2)+"\n")


def forecast(model, group, enc, batch):
    with np.load(OUT/f"{group}_panel.npz") as f:
        ids,x = f["ids"],f["x"]
    horizon = GROUPS[group][0]
    outputs, states, shifts, scales = [], [], [], []
    for lo in range(0,len(x),batch):
        ctx = x[lo:lo+batch].astype("f4")
        with torch.inference_mode():
            if model.startswith("moment"):
                with ActivationCatcher(enc.module, ["encoder.final_layer_norm"]) as cap:
                    y = enc.predict(ctx,horizon,[.5])["point"]
                    states.append(cap.collect()["encoder.final_layer_norm"][:,-math.ceil(horizon/8):].numpy())
                mu,sd = scaling(enc.module.normalizer)
                shifts.extend(mu)
                scales.extend(sd)
            elif model=="timesfm":
                # Native TimesFM decode takes a True-on-padding mask and a multiple of 32 inputs.
                # Adapter.prepare is for rectangular, fully observed batches only.
                val = torch.from_numpy(np.nan_to_num(ctx))
                mask = torch.from_numpy(~np.isfinite(ctx))
                full,_,_ = enc.module.decode(horizon,val,mask)
                y = full[:,-1,:horizon,enc._point_idx].numpy()
            else:
                y = enc.predict(ctx,horizon,[.5])["quantiles"][...,0]
        outputs.append(y)
    ans = dict(ids=ids, native=np.concatenate(outputs))
    if states:
        ans.update(h=np.concatenate(states),mu=np.asarray(shifts),sd=np.asarray(scales))
    assert np.isfinite(ans["native"]).all()
    np.savez_compressed(OUT/f"{group}_{model}.npz",**ans)


def metrics(pred, truth, scale):
    den = np.abs(pred)+np.abs(truth)
    err = np.abs(pred-truth)
    return np.mean(np.divide(200*err,den,out=np.zeros_like(den),where=den!=0),axis=1),np.mean(err,axis=1)/scale


def score(group):
    with np.load(OUT/f"{group}_panel.npz") as f:
        ids, truth, scale = f["ids"], f["truth"],f["scale"]
        pred = {name:f[name] for name in ("last","seasonal","drift")}
    for model in NAMES:
        with np.load(OUT/f"{group}_{model}.npz") as f:
            np.testing.assert_array_equal(ids,f["ids"])
            pred[model] = f["native"].astype(float)
            if model.startswith("moment"):
                size=model.split("_")[-1]
                with np.load(HEADS/f"{size}_0_heads.npz") as weights:
                    for head in ("interior_published","interior_hidden","interior_orthogonal","terminal_hidden"):
                        v=f["h"]@weights[head+"_weight"] + weights[head+"_bias"]
                        y=(v*f["sd"][:,None,None]+f["mu"][:,None,None]).reshape(len(ids),-1)[:,:truth.shape[1]]
                        if head=="interior_published":
                            np.testing.assert_allclose((y-pred[model])/f["sd"][:,None],0,atol=4e-5)
                        else:
                            pred[size+"_"+head]=y
    vals={k:metrics(p, truth, scale) for k,p in pred.items()}
    rng=np.random.default_rng(22112218)
    boot=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=4000)/len(ids)
    def interval(v):
        return [float(np.mean(v)),*np.percentile(boot@v,[2.5,97.5]).tolist()]
    report={}
    for name,scores in vals.items():
        report[name]={}
        for i,metric in enumerate(("smape","mase")):
            v=scores[i]
            report[name][metric]=interval(v)
            for k in ("last","seasonal","bolt","moment_"+name.split("_")[0]):
                if k not in vals or k==name:continue
                d=v-vals[k][i]
                report[name][metric+"_minus_"+k]=interval(d)
                report[name][metric+"_wins_"+k]=float(np.mean((d<0)+.5*(d==0)))
    (OUT/f"{group}_scores.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("data","predict","score"))
    p.add_argument("--model",choices=NAMES)
    p.add_argument("--batch",type=int,default=16)
    a=p.parse_args()
    torch.set_num_threads(2)
    model=adapter(a.model) if a.stage=="predict" else None
    for g in GROUPS:
        if a.stage=="data":choose(g)
        elif a.stage=="predict":forecast(a.model,g,model,a.batch)
        else:score(g)
        print(a.stage,a.model,g,flush=True)
