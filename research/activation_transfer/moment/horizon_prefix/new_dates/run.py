"""Confirm fixed-prefix forecast changes on new dates and geographically filtered held-out stations."""
import argparse
from datetime import date,timedelta
from hashlib import sha256,file_digest
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import torch
from research.activation_transfer.moment.forecast_quality.post_release import data as first
from research.activation_transfer.moment.forecast_quality.predict import adapter
from research.activation_transfer.moment.token_paths.geometry import CACHE

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
ORIGINS=[date(2026,m,1) for m in range(1,12)]
START=first.START
DAYS=[START+timedelta(days=i) for i in range((date(2026,11,9)-START).days)]
PHASE=np.array([(date(2000,d.month,d.day)-date(2000,1,1)).days for d in DAYS])
CUTS=np.array([(d-START).days for d in ORIGINS])


def read(z,files):
    v=np.full(len(DAYS),np.nan);coords=None
    for yr,f in sorted(files.items()):
        if yr<2008 or yr>2026:continue
        for line in z.read(f).splitlines():
            cols=line.split()
            if not cols:continue
            ds=cols[1].decode()
            k=(date(int(ds[:4]),int(ds[4:6]),int(ds[6:]))-START).days
            if not 0<=k<len(v):continue
            val=float(cols[8])
            # Verify NOAA's fixed-width specification with a second parsing convention.
            assert val==float(line[62:69])
            if val!=-9999:v[k]=val
            if yr==2026:coords=[float(cols[3]),float(cols[4])]
    return v,coords


def eligible(v):
    if sum(np.isfinite(v[:len(first.YEARS)][first.YEARS==y]).sum()>=300 for y in range(2008,2024))<8:return "short_climatology"
    for t in CUTS:
        if not np.all(np.isfinite(v[t:t+8])):return "missing_future"
        m=~np.isfinite(v[t-480:t])
        if m.sum()>9 or np.max(np.convolve(m,np.ones(8,int),mode="valid"))>=8:return "missing_past"
    return None


def prepare(path):
    m=json.loads((first.OUT/"manifest.json").read_text())
    with path.open("rb") as f:assert file_digest(f,"sha256").hexdigest()==m["archive_sha256"]
    old=set(m["accepted"])
    dates={d+timedelta(days=i) for d in ORIGINS for i in range(8)}
    previous={date.fromisoformat(d)+timedelta(days=i) for d in m["origins"] for i in range(16)}
    assert len(dates)==88 and dates.isdisjoint(previous)
    results={c:[] for c in ("primary","secondary")};log={}
    with ZipFile(path) as z:
        members={}
        for f in z.namelist():
            pat=re.fullmatch(r"(\d{4})/CRND0103-\1-([A-Z]{2}_.+)\.txt",f)
            if pat and pat[2][:2] in first.REGION:members.setdefault(pat[2],{})[int(pat[1])]=f
        candidates=sorted((s for s in members if 2026 in members[s]),key=lambda s:sha256(("frozen-new-temp-dates|"+s).encode()).digest())
        for s in candidates:
            key="primary" if s in old else "secondary"
            v,coords=read(z,members[s])
            reason=eligible(v)
            if not reason and key=="secondary" and any(first.distance(coords,r[1])<200 for r in results[key]):
                reason="closer_than_200km"
            log[s]=reason or key
            if reason:continue
            curve,rho=first.climate(v)
            inputs=[];truth=[];naive=[];missing=[]
            for t in CUTS:
                obs=v[t-480:t].copy();holes=np.flatnonzero(~np.isfinite(obs));missing.append(len(holes))
                for hole in holes:
                    k=t-480+hole
                    last=np.flatnonzero(np.isfinite(v[:k]))[-1]
                    obs[hole]=curve[PHASE[k]]+rho**(k-last)*(v[last]-curve[PHASE[last]])
                inputs.append(obs);truth.append(v[t:t+8])
                naive.append(curve[PHASE[t:t+8]]+(obs[-1]-curve[PHASE[t-1]])*rho**np.arange(1,9))
            results[key].append((s,coords,inputs,truth,naive,missing))
    OUT.mkdir(exist_ok=True,parents=True)
    allrows=results["primary"]+results["secondary"]
    if min(map(len,results.values()))<4:raise ValueError("Insufficient cohort")
    ids,coords,x,y,climate,missing=map(np.asarray,zip(*allrows))
    sets=np.repeat(list(results),[len(v) for v in results.values()])
    manifest=dict(archive_sha256=m["archive_sha256"],cohorts={k:len(q) for k,q in results.items()},
                  ids=ids.tolist(),origins=[str(o) for o in ORIGINS],disposition=log,
                  context_sha256=sha256(x.tobytes()).hexdigest(),future_sha256=sha256(y.tobytes()).hexdigest())
    mf=OUT/"manifest.json"
    if mf.exists():assert json.loads(mf.read_text())==manifest
    mf.write_text(json.dumps(manifest,indent=2)+"\n")
    np.savez_compressed(OUT/"panel.npz",ids=ids,coords=coords,x=x,y=y,climate=climate,missing=missing,cohort=sets)
    print(manifest["cohorts"],flush=True)


def pred(size,H,batch):
    with np.load(OUT/"panel.npz") as f:x=f["x"].reshape(-1,480);ids=f["ids"]
    a=adapter("moment_"+size)
    folder=CACHE/"prefix_new_dates"/f"{size}_{H}";folder.mkdir(exist_ok=True,parents=True)
    vs=[]
    for lo in range(0,len(x),112):
        dest=folder/f"{lo}.npy"
        if not dest.exists():
            arr=np.concatenate([a.predict(x[i:min(i+batch,lo+112)],H,[.5])["point"][:,:8]
                                for i in range(lo,min(lo+112,len(x)),batch)])
            assert np.isfinite(arr).all()
            temp=dest.with_suffix(".tmp.npy");np.save(temp,arr);temp.replace(dest)
        vs.append(np.load(dest))
        print(size,H,lo,len(x),flush=True)
    np.savez_compressed(OUT/f"{size}_{H}.npz",ids=ids,y=np.concatenate(vs).reshape(len(ids),len(ORIGINS),8))


def score():
    with np.load(OUT/"panel.npz") as f:ids,x,y,clim,cs=f["ids"],f["x"],f["y"],f["climate"],f["cohort"]
    models={"climate_anomaly":clim,"last":np.broadcast_to(x[:,:,-1:],y.shape)}
    for key in ("small_8","small_16","large_8","large_16","large_32"):
        with np.load(OUT/(key+".npz")) as f:np.testing.assert_array_equal(ids,f["ids"]);models[key]=f["y"]
    rng=np.random.default_rng(22112303);out={}
    for name in np.unique(cs):
        m=cs==name;n=int(m.sum());t=len(ORIGINS)
        sw=rng.multinomial(n,np.full(n,1/n),size=5000)/n
        dw=rng.multinomial(t,np.full(t,1/t),size=5000)/t
        def summary(v):
            return dict(mean=float(v.mean()),station95=np.quantile(sw@v.mean(1),[.025,.975]).tolist(),
                        date95=np.quantile(dw@v.mean(0),[.025,.975]).tolist(),
                        crossed95=np.quantile(np.sum((sw@v)*dw,axis=1),[.025,.975]).tolist(),
                        per_date=v.mean(0).tolist(),positive_dates=int(np.sum(v.mean(0)>0)),
                        leave_one_date_out=((v.sum()-v.sum(0))/(v.size-n)).tolist())
        errors={k:np.abs(v[m]-y[m]).mean(2) for k,v in models.items()}
        cell=dict(stations=n,dates=t,MAE_C={k:summary(e) for k,e in errors.items()})
        for prefix,other in (("small",16),("large",16),("large",32)):
            k=prefix+"_"+str(other);base=prefix+"_8"
            cell[k+"_minus_direct_MAE_C"]=summary(errors[k]-errors[base])
            cell[k+"_minus_direct_RMS_C"]=summary(np.sqrt(np.mean((models[k][m]-models[base][m])**2,axis=2)))
        out[str(name)]=cell
    (OUT/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("data","predict","score"))
    p.add_argument("--archive",type=Path,default=Path("/tmp/tsfm_moment_quality_20261122/noaa")/first.SNAPSHOT)
    p.add_argument("--size",choices=("small","large"));p.add_argument("--horizon",type=int,choices=(8,16,32))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="data":prepare(a.archive)
    elif a.stage=="predict":pred(a.size,a.horizon,a.batch)
    else:score()
