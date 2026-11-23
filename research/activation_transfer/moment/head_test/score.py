"""Paired real-future error under published, calibrated-output, and frozen hidden readouts."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT as QUALITY
from research.activation_transfer.moment.forecast_quality.score import boot_weights,item_metrics,percentile
from research.activation_transfer.moment.head_test.data import OUT


def report(out):
    models={}
    for p in sorted(out.glob("*_pred.npz")):
        with np.load(p) as f:
            models[p.name.removesuffix("_pred.npz")]=dict(f)
    if not models:raise ValueError("No fitted heads")
    with np.load(QUALITY/"panel.npz") as f:x,y,ids,split,domains=f["context"],f["future"],f["id"],f["split"],f["domain"]
    for data in models.values():np.testing.assert_array_equal(data["ids"],ids)
    with np.load(QUALITY/"classical.npz") as f:baselines={k:f[k] for k in f if k.startswith(("last_","history_select_","season1_"))}
    for m in ("timesfm","bolt","moment_large","moment_base","moment_small"):
        if (QUALITY/(m+".npz")).exists():
            with np.load(QUALITY/(m+".npz")) as f:
                np.testing.assert_array_equal(f["ids"],ids)
                for H in (8,16):baselines[m+"_"+str(H)]=f["raw_"+str(H)]
    headlines={}
    for H,segment,end in ((16,slice(16),"h16"),(16,slice(8),"prefix8"),(16,slice(8,16),"late8"),(8,slice(8),"h8")):
        preds={k.removesuffix("_"+str(H)):v for k,v in baselines.items() if k.endswith("_"+str(H))}
        for model,archive in models.items():
            for k,v in archive.items():
                if not k.endswith("_"+str(H)) or "_published_" in k and "_published_space_" not in k:continue
                preds[model+"_"+k.removesuffix("_"+str(H))]=v
        headlines[end]={}
        for domain in np.unique(domains):
            use=(domains==domain)&(split=="holdout")
            bootstrap=boot_weights("head-test/"+str(domain),int(use.sum()))
            stats={}
            errors={}
            for k,pred in preds.items():
                valid=pred[use,:,segment]
                if not np.isfinite(valid).all():continue
                vals=item_metrics(valid,y[use,:,segment],x[use])
                stats[k]={q:percentile(bootstrap@vals[q]) for q in ("mase1","nmae_sd","mae")}
                stats[k]["median_item_mase1"]=float(np.median(vals["mase1"]))
                errors[k]=vals["mase1"]
            paired={}
            for target in ("small_0_interior_hidden","small_0_terminal_hidden","base_0_interior_hidden",
                           "large_0_interior_hidden","large_0_terminal_hidden"):
                if target not in errors:continue
                size=target.split("_")[0];task=target.split("_")[2]
                for control in ("last","history_select","season1","bolt","timesfm","moment_"+size,
                                size+"_0_"+task+"_published_space",size+"_0_"+task+"_orthogonal",
                                size+"_619_"+task+"_hidden",size+"_947_"+task+"_hidden","input_lag464","input_lag32"):
                    if control not in errors:continue
                    a,b=errors[target],errors[control]
                    paired[target+"_vs_"+control]=dict(ratio=percentile((bootstrap@a)/(bootstrap@b)),
                        delta=percentile(bootstrap@(a-b)),win_fraction=float(np.mean(np.where(a==b,.5,a<b))))
            headlines[end][str(domain)]=dict(n=int(use.sum()),scores=stats,comparisons=paired)
    (out/"summary.json").write_text(json.dumps(headlines,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=OUT)
    report(p.parse_args().out)
