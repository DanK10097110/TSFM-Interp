"""Paired station-block and region-block forecast comparisons using unscaled degrees Celsius."""
import json
import numpy as np

from research.activation_transfer.moment.forecast_quality.data import hash_array
from research.activation_transfer.moment.forecast_quality.post_release.data import ORIGINS, OUT
from research.activation_transfer.moment.forecast_quality.predict import NAMES
from research.activation_transfer.moment.forecast_quality.score import boot_weights, item_metrics, percentile


def run():
    with np.load(OUT / "panel.npz") as panel:
        x, y = panel["context"], panel["future"]
        ids, region = panel["ids"], panel["region"]
        observed = {"climatology":panel["clim"], "climate_anomaly":panel["anomaly"],
                    "lag365":x[:,:,-365:-349], "last":np.repeat(x[:,:,-1:],16,axis=2),
                    "drift":x[:,:,-1:] + np.arange(1,17)*(x[:,:,-1:]-x[:,:,:1])/479}
    registry = json.loads((OUT / "manifest.json").read_text())
    np.testing.assert_array_equal(ids, registry["accepted"])
    if hash_array(x) != registry["sha256_context"] or hash_array(y) != registry["sha256_future"]:
        raise ValueError("Weather evaluation arrays no longer match the frozen registry")
    models = {}
    for model in NAMES:
        with np.load(OUT / (model+".npz")) as f:
            np.testing.assert_array_equal(ids, f["ids"])
            assert hash_array(x) == str(f["context_sha256"])
            models[model] = {h:f["raw_"+str(h)] for h in (8,16)}
    weights = boot_weights("noaa_stations", len(ids))
    regions = sorted(set(region))
    # Sample regions as complete blocks, preserving each region's station count and
    # the station-weighted overall point estimate. Four blocks limit precision.
    regional_draws = boot_weights("noaa_regions", len(regions))
    reg_weights = regional_draws[:, np.searchsorted(regions, region)]
    reg_weights /= reg_weights.sum(1, keepdims=True)
    summary = {}
    for endpoint, h, sl in (("h8",8,slice(8)), ("h16",16,slice(16)),
                            ("h16_first8",16,slice(8)),("h16_last8",16,slice(8,16))):
        ref = y[:,:,sl]
        forecasts = {m:v[:,:,sl] for m,v in observed.items()}
        forecasts.update({m:f[h][:,:,sl] for m,f in models.items()})
        evaluated = {m:item_metrics(v,ref,x) for m,v in forecasts.items()}
        metrics = {m:{k:percentile(weights@v[k]) for k in ("mae","mase1","bias")} for m,v in evaluated.items()}
        for method, values in evaluated.items():
            metrics[method]["rmse"] = percentile(np.sqrt(weights@values["mse"]))
        by_region = {r:{m:float(np.mean(v["mae"][region==r])) for m,v in evaluated.items()} for r in regions}
        by_origin = {str(d):{m:float(np.abs(v[:,j]-ref[:,j]).mean()) for m,v in forecasts.items()}
                     for j,d in enumerate(ORIGINS)}
        by_origin_bias = {str(d):{m:float((v[:,j]-ref[:,j]).mean()) for m,v in forecasts.items()}
                          for j,d in enumerate(ORIGINS)}
        comparisons = {}
        for m in models:
            errors = evaluated[m]["mae"]
            for baseline in ("last","lag365","climatology","climate_anomaly","bolt","timesfm"):
                if m == baseline:
                    continue
                base = evaluated[baseline]["mae"]
                comparisons[m + "_vs_" + baseline] = {
                    "mae_ratio":percentile((weights@errors)/(weights@base)),
                    "mae_difference_c":percentile(weights@(errors-base)),
                    "region_block_mae_difference_c":percentile(reg_weights@(errors-base)),
                    "station_win_rate":percentile(weights@(errors<base).astype(float)),
                    "leave_one_region_out_mae_difference_c":{
                        r:float((errors[region!=r]-base[region!=r]).mean()) for r in regions
                    },
                    "leave_one_origin_out_mae_difference_c":{
                        str(day):float(np.delete(np.abs(forecasts[m]-ref)-np.abs(forecasts[baseline]-ref),j,axis=1).mean())
                        for j,day in enumerate(ORIGINS)
                    },
                }
        summary[endpoint] = dict(metrics=metrics, paired=comparisons, by_region_mae_c=by_region,
                                 by_origin_mae_c=by_origin,by_origin_mean_signed_error_c=by_origin_bias)
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    run()
