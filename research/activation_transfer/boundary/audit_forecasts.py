"""Bootstrap forecast direction, magnitude and error together by independent evaluation item."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.boundary.analyze import BOOT, ci, metric_vectors


def format_interval(value, samples):
    return {"value": float(value), "ci95": np.quantile(samples, [.025,.975]).tolist()}


def sufficient_stats(p, r):
    return np.stack([np.sum(p*r, axis=(1,2)), np.sum(p*p, axis=(1,2)),
                     np.sum(r*r, axis=(1,2)), np.sum((p-r)**2, axis=(1,2))],axis=1)


def pooled(s):
    cross, pred_e, true_e, e = np.moveaxis(s, -1,0)
    return {
        "gain": cross/np.maximum(true_e,1e-24),
        "relative_rmse": np.sqrt(e/np.maximum(true_e,1e-24)),
        "energy_ratio": pred_e/np.maximum(true_e,1e-24),
        "cosine": cross/np.sqrt(np.maximum(pred_e*true_e,1e-48)),
    }


def pool_bootstrap(p, r, draws):
    stats = sufficient_stats(p, r)
    return pooled(stats.sum(axis=0)), pooled(stats[draws].sum(axis=1))


def audit(path):
    report = json.loads(path.read_text())
    values = {k:np.asarray(v,dtype=np.float64) for k,v in report["forecast_changes"].items()}
    r = values["native_dense"]
    draws = np.random.default_rng(20261120).integers(len(r), size=(BOOT,len(r)))
    arms, estimates = {}, {}
    for arm, pred in values.items():
        point, boot = pool_bootstrap(pred, r, draws)
        stats = {k:format_interval(x,boot[k]) for k,x in point.items()}
        stats["mean_item_cosine"] = ci(metric_vectors(pred,r)["cosine"])
        stats["mean_item_shape_cosine"] = ci(metric_vectors(
            pred-pred.mean(-1,keepdims=True), r-r.mean(-1,keepdims=True))["cosine"])
        rms = np.sqrt(np.mean(pred**2,axis=-1))
        stats["mean_rms"] = ci(rms)
        arms[arm] = stats
        estimates[arm] = (point,boot)
    contrasts = {}
    for a,b in [("cross_dense","cross_code"),("cross_fixed_code","cross_code"),
                ("cross_shoulder","cross_code"),("cross_subtract","cross_code"),
                ("native_fixed_decoded","native_decoded"),("cross_dense","cross_input"),
                ("cross_shoulder","cross_dense"),("cross_subtract","cross_dense")]:
        if a not in estimates:
            continue
        pt_a,bt_a = estimates[a]
        pt_b,bt_b = estimates[b]
        diff = {k:format_interval(pt_a[k]-pt_b[k],bt_a[k]-bt_b[k]) for k in pt_a}
        diff["mean_item_cosine"] = ci(metric_vectors(values[a],r)["cosine"]-
                                      metric_vectors(values[b],r)["cosine"])
        contrasts[a+"_minus_"+b] = diff
    return {"input":path.name,"n_items":len(r),"near_floor":report["reference_near_floor_count"],
            "replay_max":report["self_patch_max"],"arms":arms,"contrasts":contrasts}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",nargs="+",type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    args.out.write_text(json.dumps([audit(f) for f in args.inputs],indent=2,allow_nan=False)+"\n")
