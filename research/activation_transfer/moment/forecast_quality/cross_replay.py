"""Compare all independent native head-test captures with this study's raw predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER
from research.activation_transfer.moment.forecast_quality.score import boot_weights, item_metrics, percentile

DEFAULT = Path("/tmp/tsfm_moment_nov22_heads")


def run(captured, provisional):
    record={}
    for dest,prefix in ((OUT,"test"),(WEATHER,"external")):
        with np.load(dest/"panel.npz") as p:
            x,y=p["context"],p["future"]
            ids=p["id"] if dest==OUT else p["ids"]
            domain=p["domain"] if dest==OUT else np.repeat("noaa",len(x))
            subset=p["split"]=="holdout" if dest==OUT else np.ones(len(x),bool)
            if dest==WEATHER:
                weather_baseline=p["anomaly"]
        for size in ("small","base","large"):
            model="moment_"+size
            for h in (8,16):
                with np.load(captured/f"{size}_0/{prefix}{h}.npz") as independent:
                    np.testing.assert_array_equal(independent["ids"],ids)
                    pred=independent["pub"].reshape(*x.shape[:2],h)
                if dest==OUT:
                    with np.load(OUT/"classical.npz") as c:
                        baseline=c[f"history_select_{h}"]
                else:
                    baseline=weather_baseline[:,:,:h]
                key=f"{prefix}_{model}_h{h}"
                out={}
                ratios=[]
                for d in np.unique(domain):
                    select=(domain==d)&subset
                    w=boot_weights(d+"/holdout" if dest==OUT else "noaa_stations",int(select.sum()))
                    field="mase1" if dest==OUT else "mae"
                    actual=y[select,:,:h]
                    a=item_metrics(pred[select],actual,x[select])[field]
                    b=item_metrics(baseline[select],actual,x[select])[field]
                    ratio=(w@a)/(w@b)
                    ratios.append(ratio)
                    out[d]={"value":percentile(w@a),"vs_history_select_or_climate_anomaly":percentile(ratio)}
                out["equal_domain_comparator_ratio"]=percentile(np.mean(ratios,axis=0))
                ours=dest/(model+".npz")
                if ours.exists():
                    with np.load(ours) as f:
                        np.testing.assert_array_equal(f["ids"],ids)
                        ours_native=f[f"raw_{h}"]
                        raw_difference=np.abs(ours_native - pred)
                        same_float32_bits=np.array_equal(ours_native.astype("f4").view("u4"),pred.astype("f4").view("u4"))
                        gap=raw_difference/x.std(2,keepdims=True)
                    if gap.max()>2e-4:
                        raise ValueError((key,gap.max()))
                    out["independent_max_error_in_history_sd"]=float(gap.max())
                    out["independent_max_error_native_units"]=float(raw_difference.max())
                    out["number_of_forecast_values_compared"]=int(pred.size)
                    out["numerically_identical_every_value"]=bool(np.count_nonzero(raw_difference)==0)
                    out["bit_identical_at_native_float32_precision"]=bool(same_float32_bits)
                elif not provisional:
                    raise FileNotFoundError(ours)
                record[key]=out
    where=Path("/tmp/tsfm_moment_quality_20261122/partial_cross_replay.json") if provisional else OUT/"cross_replay.json"
    where.write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured",type=Path,default=DEFAULT)
    parser.add_argument("--provisional",action="store_true")
    args=parser.parse_args()
    run(args.captured,args.provisional)
