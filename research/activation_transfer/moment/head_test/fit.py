"""Fit a common linear 8-value decoder, without using test targets for fitting or selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT as QUALITY
from research.activation_transfer.moment.head_test.data import OUT,SCRATCH

PENALTIES=np.logspace(-6,1,8)


def ridge(x,y,cal,truth):
    """Center features and targets on training only; select relative penalty by held-out MSE."""
    x=x.astype(float);y=y.astype(float)
    mx=x.mean(0);my=y.mean(0)
    xc=x-mx
    eig,basis=np.linalg.eigh(xc.T@xc)
    eig=eig.clip(min=0.)
    cross=basis.T@(xc.T@(y-my))
    errors=[];ws=[]
    for relative in PENALTIES:
        w=basis@(cross/(eig[:,None]+relative*max(eig[-1],1e-18)))
        val=(cal-mx)@w+my
        errors.append(float(np.mean((val-truth)**2)));ws.append(w)
    best=int(np.argmin(errors))
    w=ws[best];b=my-mx@w
    return w,b,dict(relative=float(PENALTIES[best]),mse_by_penalty=errors)


def learn(training,task,seed):
    h=training["h"].astype(float)
    y=training["y"].astype(float)
    is_train=training["split"]=="train"
    d=h.shape[-1]
    w0=training["weight"].astype(float).T; b0=training["bias"].astype(float)
    axes=np.linalg.qr(w0)[0]
    fits={"published":(w0,b0)}
    report={}
    representations=["hidden"] if seed else ["hidden","published_space","orthogonal"]
    for name in representations:
        if name=="hidden":features=h
        elif name=="published_space":features=h@w0+b0
        else:features=h-(h@axes)@axes.T
        train=features[is_train].reshape(-1,features.shape[-1])
        cal=features[~is_train].reshape(-1,features.shape[-1])
        w,b,record=ridge(train,y[is_train].reshape(-1,8),cal,y[~is_train].reshape(-1,8))
        if name=="published_space":
            b=b0@w+b;w=w0@w
        elif name=="orthogonal":
            w=w-axes@(axes.T@w)
            np.testing.assert_allclose(axes.T@w,0.,atol=1e-9)
        fits[name]=(w,b);report[name]=record
    for name,(w,b) in fits.items():
        pred=h[~is_train]@w+b
        mse=float(np.mean((pred-y[~is_train])**2))
        energy=float(np.mean((y[~is_train]-y[is_train].mean(axis=(0,1)))**2))
        report.setdefault(name,{})["val_r2"]=float(1-mse/energy)
        report[name]["val_mse"]=mse
    return fits,report


def train(size,seed,out):
    base=SCRATCH/f"{size}_{seed}"
    out.mkdir(parents=True,exist_ok=True)
    learned={};meta={}
    for task in ("interior","terminal"):
        with np.load(base/(task+".npz")) as f:train_data=dict(f)
        fits,meta[task]=learn(train_data,task,seed)
        for key,(w,b) in fits.items():
            learned[task+"_"+key+"_weight"]=w
            learned[task+"_"+key+"_bias"]=b
    np.savez_compressed(out/f"{size}_{seed}_heads.npz",**learned)
    (out/f"{size}_{seed}_fit.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")


def forecast(size,seed,out):
    base=SCRATCH/f"{size}_{seed}"
    with np.load(out/f"{size}_{seed}_heads.npz") as f:heads=dict(f)
    pred={}
    for H in (8,16):
        with np.load(base/f"test{H}.npz") as f:
            ids,h,sd,mu=f["ids"],f["h"],f["sd"],f["mu"]
            for name in heads:
                if not name.endswith("_weight"):continue
                key=name.removesuffix("_weight")
                normalized=h@heads[name]+heads[key+"_bias"]
                raw=normalized*sd[:,None,None]+mu[:,None,None]
                pred[f"{key}_{H}"]=raw.reshape(len(ids),2,H)
                if key=="interior_published":
                    np.testing.assert_allclose((raw-f["pub"])/sd[:,None,None],0.,atol=5e-5)
        if not seed:
            with np.load(QUALITY/f"moment_{size}.npz") as f:
                np.testing.assert_array_equal(ids,f["ids"])
                gap=(pred[f"interior_published_{H}"]-f[f"raw_{H}"]).reshape(-1,H)/sd[:,None]
                np.testing.assert_allclose(gap,0.,atol=2e-4)
    np.savez_compressed(out/f"{size}_{seed}_pred.npz",ids=ids,**pred)


def input_baseline(out):
    with np.load(SCRATCH/"train.npz") as f: x,split=f["context"],f["split"]
    with np.load(QUALITY/"panel.npz") as f: test,ids=f["context"],f["id"]
    with np.load(SCRATCH/"small_0/terminal.npz") as f:
        # Public terminal task, first 464 seen and last 16 withheld.
        mu,sd=f["mu"],f["sd"]
    past=(x[:,:-16]-mu[:,None])/sd[:,None]
    future=(x[:,-16:]-mu[:,None])/sd[:,None]
    tr=split=="train"
    z=test.reshape(-1,480)
    mean=z.mean(1);stdev=z.std(1)+1e-5
    norm=(z-mean[:,None])/stdev[:,None]
    report={};pred={"ids":ids}
    for n in (32,464):
        a=past[:,-n:]
        w,b,record=ridge(a[tr],future[tr]-past[tr,-1,None],a[~tr],future[~tr]-past[~tr,-1,None])
        y=(norm[:,-n:]@w+b+norm[:,-1,None])*stdev[:,None]+mean[:,None]
        pred[f"lag{n}_16"]=y.reshape(len(ids),2,16)
        pred[f"lag{n}_8"]=pred[f"lag{n}_16"][:,:,:8]
        report[f"lag{n}"]=record
    (out/"input_fit.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(out/"input_pred.npz",**pred)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("train","predict","input"))
    p.add_argument("size",nargs="?",default="small");p.add_argument("seed",nargs="?",type=int,default=0)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    if a.stage=="input":input_baseline(a.out)
    elif a.stage=="train":train(a.size,a.seed,a.out)
    else:forecast(a.size,a.seed,a.out)
