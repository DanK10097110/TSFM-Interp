"""Score frozen physical edits against the eight realized observations following their archived origins."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND, HERE as BF
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
SELF = STUDIES/"self_transfer/results"
OUT = HERE/"results"
TOLERANCE = 1e-6


def future(x, ids):
    sources = json.loads((BF/"results/manifest.json").read_text())["domains"]
    target = np.full((len(ids),8),np.nan)
    for domain,meta in sources.items():
        with pa.memory_map(meta["source"]) as stream:
            table = pa.ipc.open_stream(stream).read_all()
        field = "item_id" if "item_id" in table.column_names else "id"
        index = {str(k):i for i,k in enumerate(table[field].to_pylist())}
        for i,key in enumerate(ids):
            prefix,item = str(key).split("/",1)
            if prefix != domain:
                continue
            series = table["target"][index[item]].values
            raw = series.slice(len(series)-488,488).to_numpy(zero_copy_only=False).astype(float)
            past = raw[:480]
            mu,sd = past.mean(), past.std()
            np.testing.assert_array_equal(((past-mu)/sd).astype("f4"),x[i])
            target[i] = (raw[480:]-mu)/sd
    return target


def load(panel,source,target,ids):
    if panel=="blind":
        with np.load(BF/"results"/(source+"_fresh.npz")) as f:
            np.testing.assert_array_equal(f["ids"],ids)
            atoms,active,ds,dt = f["atoms"],f["active"],f["source"][...,:8],f["minimum"][...,:8]
        base = []
        for model in (source,target):
            with np.load(BLIND/(model+"_early.npz")) as f:
                np.testing.assert_array_equal(f["ids"],ids)
                base.append(f["forecast"][:,:8])
    else:
        def get(path,key):
            with np.load(path) as f:
                keep = np.char.startswith(f["split"],"extra_")
                np.testing.assert_array_equal(f["ids"][keep],ids)
                return f[key][keep]
        self_panel = SELF/f"{source}_to_{source}.npz"
        with np.load(self_panel) as f:
            atoms = f["atoms"]
        active = get(self_panel,"active")
        ds = get(self_panel,"native")[...,:8]
        dt = get(SELF/f"{source}_to_{target}.npz","adaptive")[...,:8]
        base = [get(DEPTH/f"{model}_{3 if model=='bolt' else 10}.npz","base")[:,:8] for model in (source,target)]
    return atoms, active, ds.astype(float), dt.astype(float), *(v.astype(float) for v in base)


def loss_change(error, delta, squared=False):
    if squared:
        return np.mean((2*error+delta)*delta,axis=-1)
    return np.mean(np.abs(error+delta)-np.abs(error),axis=-1)


def avg(values,mask,weights):
    numbers = weights @ mask.astype(float)
    sums = weights @ np.where(mask,values,0.)
    return np.nanmean(np.divide(sums,numbers,out=np.full_like(sums,np.nan),where=numbers>0),axis=1)


def entry(v):
    return dict(value=float(v[0]),ci95=np.nanquantile(v[1:],[.025,.975]).tolist())


def assess(ds,dt,bs,bt,y,active):
    valid = np.isfinite(y).all(axis=-1)
    keep = (active & valid[:,None]).sum(axis=0)>=8
    m = active[:,keep] & valid[:,None]
    ds,dt = ds[:,keep],dt[:,keep]
    w = np.vstack((np.ones(len(y)),resampling_counts(len(y),draws=3000))).astype(float)
    metrics = dict(items=len(y),available_outcomes=int(valid.sum()),features=int(keep.sum()),cases=int(m.sum()),
                   normalized_clean_mae_source=float(np.abs(bs[valid]-y[valid]).mean()),
                   normalized_clean_mae_target=float(np.abs(bt[valid]-y[valid]).mean()))
    for squared in (False,True):
        e1,e2 = (bs-y)[:,None,:],(bt-y)[:,None,:]
        donor = loss_change(e1,ds,squared)
        block = dict(donor_mean_change=entry(avg(donor,m,w)),donor_mean_abs_change=entry(avg(np.abs(donor),m,w)),
                     donor_harm_rate=entry(avg(donor>TOLERANCE,m,w)))
        sign_donor = donor>0
        draws = {}
        for label,candidate in (("physical",loss_change(e2,dt,squared)),("same_edit_other_base",loss_change(e2,ds,squared)),
                                ("other_edit_same_base",loss_change(e1,dt,squared))):
            ok = m & (np.abs(donor)>TOLERANCE) & (np.abs(candidate)>TOLERANCE)
            sign_target = candidate>0
            agree = avg(sign_target==sign_donor,ok,w)
            counts = w @ ok.astype(float)
            def rate(sign):
                return np.divide(w @ (ok&sign).astype(float),counts,out=np.full_like(counts,np.nan),where=counts>0)
            p,q = rate(sign_donor),rate(sign_target)
            expected = np.nanmean(p*q+(1-p)*(1-q),axis=1)
            draws[label] = agree
            block[label] = dict(agree=entry(agree),marginal_expectation=entry(expected),above_marginals=entry(agree-expected),
                                mean_change=entry(avg(candidate,m,w)),mean_abs_change=entry(avg(np.abs(candidate),m,w)),
                                target_harm_rate=entry(avg(candidate>TOLERANCE,m,w)),non_tied_events=int(ok.sum()))
        block["perfect_target_displacement_minus_physical"] = entry(draws["same_edit_other_base"]-draws["physical"])
        metrics["mse" if squared else "mae"] = block
    return metrics


def run(out):
    result = {}
    out.mkdir(exist_ok=True,parents=True)
    for panel,root in (("blind",BLIND),("extra",FRESH)):
        with np.load(root/"data.npz") as f:
            x,ids,split = f["x"],f["ids"],f["split"]
        if set(split) != {"daily","monash_traffic"}:
            raise ValueError("Unexpected data sources")
        y = future(x,ids)
        panel_result = dict(future_sha256=sha256(y.tobytes()).hexdigest(),input_sha256=sha256(x.tobytes()).hexdigest())
        save = dict(ids=ids,split=split,future=y)
        for source,target in (("bolt","timesfm"),("timesfm","bolt")):
            atoms,mask,ds,dt,bs,bt = load(panel,source,target,ids)
            save[source+"_base"] = bs
            panel_result[source] = {}
            for domain in np.unique(split):
                use = split==domain
                panel_result[source][str(domain)] = assess(ds[use],dt[use],bs[use],bt[use],y[use],mask[use])
        np.savez_compressed(out/(panel+".npz"),**save)
        result[panel] = panel_result
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
