"""Score the three fixed reconstruction readouts on the independent post-checkpoint weather panel."""
import argparse
import json
import numpy as np

from research.activation_transfer.moment.head_test.data import QUALITY,OUT,SCRATCH
from research.activation_transfer.moment.forecast_quality.score import boot_weights,percentile


def run(out):
    weather=QUALITY.parent/"post_release/results"
    with np.load(weather/"panel.npz") as f:
        ids,x,y,clim,anomaly,region=f["ids"],f["context"],f["future"],f["clim"],f["anomaly"],f["region"]
    models={}
    for path in sorted(out.glob("*_heads.npz")):
        prefix=path.name.replace("_heads.npz","")
        fpath=SCRATCH/prefix/"external16.npz"
        if not fpath.exists():continue
        with np.load(path) as f:heads=dict(f)
        with np.load(fpath) as f:
            np.testing.assert_array_equal(ids,f["ids"])
            h,mu,sd=f["h"],f["mu"],f["sd"]
        for key in ("interior_published","interior_hidden","terminal_hidden","interior_orthogonal","interior_published_space"):
            if key+"_weight" not in heads:continue
            w,b=heads[key+"_weight"],heads[key+"_bias"]
            z=(h@w+b)*sd[:,None,None]+mu[:,None,None]
            models[prefix+"_"+key]=z.reshape(y.shape)
    models.update(last=np.broadcast_to(x[:,:,-1,None],y.shape),climatology=clim,climate_anomaly=anomaly)
    per_station={key:np.mean(np.abs(forecast-y),axis=(1,2)) for key,forecast in models.items()}
    w=boot_weights("future-stations-2026",len(ids))
    overall={key:percentile(w@mae) for key,mae in per_station.items()}
    paired={}
    for label in [k for k in per_station if k.endswith("_0_interior_hidden")]:
        prefix=label.removesuffix("_interior_hidden")
        for comparator in ("last","climatology","climate_anomaly",prefix+"_interior_published",
                           prefix+"_interior_orthogonal",prefix+"_interior_published_space",prefix.replace("_0","_619")+"_interior_hidden"):
            if comparator in per_station:
                diff=per_station[label]-per_station[comparator]
                paired[label+"_minus_"+comparator]=dict(**percentile(w@diff),
                    regions={r:float(diff[region==r].mean()) for r in np.unique(region)})
    result=dict(stations=len(ids),origins=x.shape[1],measure="station mean absolute error in degrees C",
                overall=overall,paired=paired)
    (out/"post_release.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=type(OUT),default=OUT);run(p.parse_args().out)
