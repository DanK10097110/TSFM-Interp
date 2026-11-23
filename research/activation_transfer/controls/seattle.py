"""Recalibrate across Seattle sensors and dates with a two-way held-out design."""

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.controls.analyze import by_horizon, fit_clean
from research.activation_transfer.controls.collect import HERE, REV, MODELS, edit_histories, forecast_responses


def sensor_grid(table, ids, ends, count):
    """Return only sensors with complete eligible histories on the prespecified date grid."""
    table_ids = [str(i) for i in table["item_id"].to_pylist()]
    index = {name:i for i,name in enumerate(table_ids)}
    histories, items, used = [], [], set()
    for name in ids:
        a = table["target"][index[name]].values.to_numpy(zero_copy_only=False)
        w = np.stack([a[e-128:e] for e in ends]).astype(np.float64)
        sd = w.std(axis=1,keepdims=True)
        if not np.isfinite(w).all() or np.any(sd<1e-8):
            continue
        x = ((w-w.mean(axis=1,keepdims=True))/sd).astype(np.float32)
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        if digest in used:
            continue
        histories.append(x)
        items.append({"item":"LOOP_SEATTLE/"+name,"grid_sha256":digest})
        used.add(digest)
        if len(items) == count:
            break
    if len(items) != count:
        raise ValueError(f"Only {len(items)} valid sensors for {count} requested")
    return items, np.stack(histories)


def collect(args):
    prior = json.loads((args.out/"items.json").read_text())
    cal_pool = [r["item"].split("/")[1] for r in prior["rows"] if r["split"]=="LOOP_SEATTLE"]
    excluded = set(cal_pool)
    for line in (HERE.parents[2]/"benchmark_v3/external_real/corpus.jsonl").read_text().splitlines():
        p = json.loads(line)["provenance"]["generator_params"]
        if p["dataset"]=="LOOP_SEATTLE":
            excluded.add(str(p["item_id"]))
    shard = args.data_root/"LOOP_SEATTLE/H/data-00000-of-00001.arrow"
    with pa.memory_map(str(shard)) as f:
        t = pa.ipc.open_stream(f).read_all()
        n = 8760
        if {len(row.values) for row in t["target"]}!={n} or len(set(t["start"].to_pylist()))!=1:
            raise ValueError("Sensor lengths or time origins differ")
        ends = np.rint(np.linspace(216,n-216,24)).astype(int)+np.random.default_rng(975515).integers(-36,37,24)
        if min(np.diff(ends)) < 144 or ends[-1]>n-144:
            raise ValueError("Overlapping windows or overlap with the earlier experiment")
        ids = [str(v) for v in t["item_id"].to_pylist()]
        trial = [ids[i] for i in np.random.default_rng(975514).permutation(len(ids)) if ids[i] not in excluded]
        c, xcal = sensor_grid(t, cal_pool,ends[:8],80)
        e, xe = sensor_grid(t, trial,ends[8:],48)
        utc = [str(t["start"][0].as_py()+timedelta(hours=int(z))) for z in ends]
    manifest={"source_sha256":hashlib.sha256(shard.read_bytes()).hexdigest(),
              "calibration_sensors":c,"test_sensors":e,"end_indices":ends.tolist(),"end_exclusive_shard_time":utc}
    mf = args.out/"seattle_grid_items.json"
    if mf.exists() and json.loads(mf.read_text())!=manifest:
        raise ValueError("Sensor grid changed")
    if not mf.exists():
        mf.write_text(json.dumps(manifest,indent=2)+"\n")
    xs = np.concatenate([xcal.reshape(-1,128), xe.reshape(-1,128)])
    pairs, _, stats = edit_histories(xs)
    output = {"stats": stats, "cal_dims": xcal.shape[:2], "test_dims": xe.shape[:2]}
    for m in MODELS:
        print("Collecting Seattle grid on",m,flush=True)
        output[m]=forecast_responses(m,pairs[:,:4],args)
    np.savez_compressed(args.out/"seattle_responses.npz",**output)


def train_scalar(s,t):
    numerator = (s*t).sum(axis=(0,2))
    denominator = np.square(s).sum(axis=(0,2))
    return np.maximum(0,np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12))


def bootstrap_weights(sensors, dates):
    rng=np.random.default_rng(88910)
    return (rng.multinomial(sensors,np.full(sensors,1/sensors),size=2000),
            rng.multinomial(dates,np.full(dates,1/dates),size=2000))


def crossed_gain(base,new,weights):
    """Aggregate horizon errors then resample all 4 stimuli jointly on both axes."""
    b=base.mean(axis=(0,1))
    gain=1-new.mean(axis=(0,1))/b
    sw,tw=weights
    # Contract sequentially so runtime and memory scale with the sensor/date grid.
    bdraw=np.einsum("bs,ste->bte",sw,base,optimize=True)
    bdraw=np.einsum("bt,bte->be",tw,bdraw)
    ndraw=np.einsum("bs,ste->bte",sw,new,optimize=True)
    ndraw=np.einsum("bt,bte->be",tw,ndraw)
    draws=1-ndraw/bdraw
    return {"gain":float(gain.mean()),"ci95":np.quantile(draws.mean(axis=1),[.025,.975]).tolist(),
            "per_edit_gain":gain.tolist(),"per_edit_ci95":np.quantile(draws,[.025,.975],axis=0).T.tolist()}


def donor_test(residual, correction, same_error):
    """A sensor shuffle holds dates fixed; a date shuffle holds sensors fixed."""
    rng=np.random.default_rng(88911)
    expected=np.mean(residual**2, axis=(0,1,3))
    observed=np.mean(same_error.mean((0,1))/expected)
    s,t=correction.shape[:2]
    out={}
    for axis,n in (("sensor",s),("date",t)):
        null=np.empty(999)
        for i in range(len(null)):
            p=rng.permutation(n)
            donor=correction[p] if axis=="sensor" else correction[:,p]
            null[i]=np.mean(np.square(residual-donor).mean((0,1,3))/expected)
        out[axis]={"p":float((1+(null<=observed).sum())/(1+len(null))),
                   "paired_gain_vs_mean_donor":float(null.mean()-observed),
                   "donor_error_interval95":np.quantile(null,[.025,.975]).tolist()}
    return out


def analyze(args):
    with np.load(args.out/"responses.npz",allow_pickle=False) as f:
        daily={k:f[k][f["split"]=="train"] for k in [*MODELS,"summaries"]}
    with np.load(args.out/"seattle_responses.npz",allow_pickle=False) as f:
        new={k:f[k] for k in f.files}
    ns,nt=map(int,new["cal_dims"])
    es,et=map(int,new["test_dims"])
    train=np.arange(ns*nt)
    test=np.arange(ns*nt,len(new["stats"]))
    weeksingle = train.reshape(ns,nt)[:,-1]
    wx=bootstrap_weights(es,et)
    results={}
    for component in ("full","shape","level"):
        ys={m:by_horizon(new[m].astype(np.float64),component) for m in MODELS}
        day={m:by_horizon(daily[m][:,:4].astype(np.float64),component) for m in MODELS}
        for source,dest in [("bolt","timesfm"),("timesfm","bolt")]:
            stats=new["stats"]
            key=f"{source}_to_{dest}"
            comp_results=results.setdefault(key,{}).setdefault(component,{})
            daily_error=None
            for scheme in ("daily","recenter","local","one_date"):
                train_x=daily["summaries"] if scheme in ("daily","recenter") else stats[train if scheme=="local" else weeksingle]
                errs_fit,errs_test={},{}
                for m in MODELS:
                    if scheme in ("daily","recenter"):
                        a,b=fit_clean(train_x,day[m],stats)
                        errs_fit[m]=day[m]-a
                        all_errs=ys[m]-b
                        if scheme=="recenter":
                            intercept=all_errs[train].mean(axis=0)
                            all_errs-=intercept
                            errs_fit[m]=all_errs[train]
                        errs_test[m]=all_errs[test]
                    else:
                        ix=train if scheme=="local" else weeksingle
                        a,b=fit_clean(train_x,ys[m][ix],stats[test])
                        errs_fit[m]=ys[m][ix]-a
                        errs_test[m]=ys[m][test]-b
                coef=train_scalar(errs_fit[source],errs_fit[dest])
                residual=errs_test[dest].reshape(es,et,4,-1)
                correction=(errs_test[source]*coef[None,:,None]).reshape(es,et,4,-1)
                base=(residual**2).mean(-1)
                error=((residual-correction)**2).mean(-1)
                if daily_error is None:
                    daily_error=base
                row={"residual_scalar":coef.tolist(),"over_own_clean":crossed_gain(base,error,wx),
                     "over_common_daily_clean":crossed_gain(daily_error,error,wx),
                     "clean_over_common_daily_clean":crossed_gain(daily_error,base,wx)}
                if scheme=="local" and component=="full":
                    row["donors"]=donor_test(residual,correction,error)
                comp_results[scheme]=row
    (args.out/"seattle_results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    for direction,data in results.items():
        for name,row in data["full"].items():
            print(direction,name,row["over_own_clean"]["gain"],row["over_own_clean"]["ci95"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub",type=Path,default=Path.home()/".cache/huggingface/hub")
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--out",type=Path,default=HERE)
    parser.add_argument("--batch-size",type=int,default=48)
    parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    if not args.data_root:
        args.data_root=args.hub/"datasets--Salesforce--GiftEval/snapshots"/REV
    if not args.analyze_only:
        collect(args)
    analyze(args)
