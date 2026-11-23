"""Calibrate all representations on the same 768/256 items and atomically publish external native code doses."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.match import dictionary, encode
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import OUT as EXTERNAL, WORK, panel
from research.activation_transfer.moment.token_paths.geometry import CACHE

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
ART=Path("/tmp/tsfm_moment_crosscoder")
SEEDS=(22112401,22112402)
GRID=np.logspace(-6,1,8)
SOURCE_ARMS=("raw","sae","joint1","joint2","masked1","masked2","shuffled1","shuffled2")
ORACLE_ARMS=("oracle_joint1","oracle_joint2")


def write_npz(path,**data):
    tmp=path.with_suffix(f".{os.getpid()}.npz")
    np.savez_compressed(tmp,**data)
    os.replace(tmp,path)


def write_json(path,value):
    tmp=path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)


def raw_data():
    ids,dates,x=panel()
    with np.load(EXTERNAL/"donor.npz") as f:
        atoms=f["atoms"];np.testing.assert_array_equal(ids,f["ids"]);np.testing.assert_array_equal(dates,f["dates"])
    assert len(atoms)==4
    with np.load(WORK/"physical_ramp_confirmation/small_base.npz") as f:
        assert str(f["sig"])==hashlib.sha256(x.tobytes()).hexdigest()
        weather_small=f["arr"]
    with np.load(CACHE/"small/train.npz") as f:
        m4_ids=f["ids"];m4_small=f["middle"][:,0]
    with np.load(CACHE/"large/train.npz") as f:
        np.testing.assert_array_equal(m4_ids,f["ids"]);m4_large=f["middle"][:,0]
    with np.load(CACHE/"small/eval.npz") as f:
        blind_small=f["middle"][:,0];blind_ids=f["ids"];blind_domain=f["split"]
    assert len(m4_ids)==1024 and len(blind_small)==256 and not set(m4_ids)&set(blind_ids)
    y=encode(dictionary("large",22112201),m4_large)[:,atoms].astype(float)
    return dict(ids=ids,dates=np.array(dates),atoms=atoms,m4_ids=m4_ids,m4_small=m4_small,
        weather_small=weather_small,blind_small=blind_small,blind_ids=blind_ids,blind_domain=blind_domain,
        m4_large=m4_large,y=y)


def threshold(scores, y):
    """Exact gated validation SSE, with tied predictions kept or removed as a group."""
    positive=np.flatnonzero(scores>0)
    if not len(positive):
        return np.inf,0,float(y@y)
    order=positive[np.argsort(-scores[positive],kind="stable")]
    s=scores[order]
    boundaries=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    gains=np.r_[0.,np.cumsum(2*y[order]*s-s*s)[boundaries]]
    k=int(np.argmax(gains))
    retained=0 if k==0 else int(boundaries[k-1])+1
    if retained==0:
        cut=np.inf
    elif retained==len(s):
        cut=0.
    else:
        cut=float((s[retained-1]+s[retained])/2)
    dose=np.where(scores>cut,scores,0)
    return cut,retained,float(np.sum((dose-y)**2))


def calibrate(design,y,fit_n=768,grid=GRID):
    x=np.asarray(design,float);y=np.asarray(y,float)
    tr=x[:fit_n];mu=tr.mean(0);xc=tr-mu
    target_mean=y[:fit_n].mean(0)
    eig,basis=np.linalg.eigh(xc@xc.T)
    eig=np.maximum(eig,0);scale=max(eig[-1],1e-20)
    cross=basis.T@(y[:fit_n]-target_mean)
    val_cross=((x[fit_n:]-mu)@xc.T)@basis
    choices=[None]*y.shape[1]
    for relative in grid:
        projected=cross/(eig[:,None]+relative*scale)
        scores=val_cross@projected+target_mean
        for j in range(y.shape[1]):
            cut,n,loss=threshold(scores[:,j],y[fit_n:,j])
            priority=(loss,n,-float(relative))
            if choices[j] is None or priority<choices[j][0]:
                choices[j]=(priority,projected[:,j].copy(),cut)
    dual=basis@np.stack([v[1] for v in choices],axis=1)
    w=xc.T@dual;b=target_mean-mu@w;gate=np.array([v[2] for v in choices])
    return w,b,gate,dict(fit_count=fit_n,validation_count=len(x)-fit_n,feature_width=x.shape[1],
        lambda_relative=[-v[0][2] for v in choices],lambda_scale=float(scale),
        validation_sse=[v[0][0] for v in choices],validation_pred_fires=[v[0][1] for v in choices],
        thresholds=[float(t) if np.isfinite(t) else None for t in gate])


def apply(x,w,b,t):
    scores=np.asarray(x,float)@w+b
    return scores,np.where(scores>t,scores,0.)


def arm_parts(arm):
    oracle=arm.startswith("oracle_")
    clean=arm.removeprefix("oracle_")
    return clean[:-1],SEEDS[int(clean[-1])-1],oracle


def representation(arm,d):
    if arm in ("raw","sae"):
        args=(d["m4_small"],d["weather_small"],d["blind_small"])
        if arm=="raw":
            return args
        sae=dictionary("small",22112201)
        return tuple(encode(sae,a) for a in args)
    method,seed,oracle=arm_parts(arm)
    with np.load(ART/f"{method}_{seed}.npz") as f:
        for a,b in (("m4_ids","m4_ids"),("station_ids","ids"),("dates","dates")):
            np.testing.assert_array_equal(f[a],d[b])
        tag="both" if oracle else "source"
        train,test=f["m4_"+tag],f["noaa_"+tag]
        # New matrices may be supplied by the training job for this entirely read-only held-out check.
        if "blind_"+tag in f.files:
            np.testing.assert_array_equal(f["blind_ids"],d["blind_ids"])
            return train,test,f["blind_"+tag]
    from research.activation_transfer.moment.crosscoder.dictionaries import load_artifact, encode_views
    model,means=load_artifact(method,seed)
    large=None
    if oracle:
        with np.load(CACHE/"large/eval.npz") as f:
            np.testing.assert_array_equal(f["ids"],d["blind_ids"])
            large=f["middle"][:,0]
    with torch.inference_mode():
        blind=encode_views(model,means,d["blind_small"],large).numpy()
        sample_large=d["m4_large"][:7] if oracle else None
        replay=encode_views(model,means,d["m4_small"][:7],sample_large).numpy()
        np.testing.assert_allclose(replay,train[:7],atol=1e-5,rtol=1e-5)
    return train,test,blind


def fit_arm(arm,d):
    train,test,blind=representation(arm,d)
    w,b,cut,report=calibrate(train,d["y"])
    va,vd=apply(train[768:],w,b,cut);ts,td=apply(test,w,b,cut)
    assert np.allclose(np.sum((vd-d["y"][768:])**2,axis=0),report["validation_sse"])
    arrays=dict(ids=d["ids"],dates=d["dates"],atoms=d["atoms"],raw=ts,dose=td,
        validation_raw=va,validation_dose=vd,validation_ids=d["m4_ids"][768:])
    bs,bd=apply(blind,w,b,cut)
    arrays.update(blind_raw=bs,blind_dose=bd,blind_ids=d["blind_ids"],blind_domain=d["blind_domain"])
    report.update(arm=arm,access="small+large" if arm in ORACLE_ARMS else "small_only",
                  atom_ids=d["atoms"].tolist(),test_pred_fires=np.sum(td>0,axis=0).tolist(),
                  validation_residual_mean=[float(v) for v in (vd-d["y"][768:]).mean(0)])
    write_npz(OUT/(arm+"_doses.npz"),**arrays)
    write_json(OUT/(arm+"_fit.json"),report)
    print(arm,"NOAA predicted fire counts",report["test_pred_fires"],flush=True)


def publish():
    outfiles={}
    for tag,arms in (("predictions",SOURCE_ARMS),("oracle_predictions",ORACLE_ARMS)):
        records={};meta={}
        for name in arms:
            fpath=OUT/(name+"_doses.npz")
            if not fpath.exists():
                continue
            with np.load(fpath) as f:
                for key in ("ids","dates","atoms"):
                    if key in records:
                        np.testing.assert_array_equal(records[key],f[key])
                    else:
                        records[key]=f[key]
                records[name]=f["dose"]
            meta[name]=json.loads((OUT/(name+"_fit.json")).read_text())
        if meta:
            write_npz(OUT/(tag+".npz"),**records)
            write_json(OUT/(tag+".json"),dict(station_major=True,units="large native SAE seed 22112201 coefficient",
                atom_ids=records["atoms"].tolist(),arms=meta))
            outfiles[tag]=list(meta)
    print("published",outfiles,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("arms",nargs="+",choices=(*SOURCE_ARMS,*ORACLE_ARMS,"publish"))
    args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(exist_ok=True,parents=True)
    if args.arms==["publish"]:
        publish()
    else:
        d=raw_data()
        for a in args.arms:
            fit_arm(a,d);publish()
