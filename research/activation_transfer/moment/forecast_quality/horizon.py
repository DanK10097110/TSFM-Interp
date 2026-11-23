"""Score the same eight observed days/steps using direct H8 and the first eight outputs of H16."""
import json
import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER
from research.activation_transfer.moment.forecast_quality.predict import NAMES
from research.activation_transfer.moment.forecast_quality.score import boot_weights, item_metrics, percentile


def main():
    output={}
    for path in (OUT, WEATHER):
        with np.load(path/"panel.npz") as p:
            x,y=p["context"],p["future"][:,:,:8]
            domain=p["domain"] if path==OUT else np.repeat("noaa",len(x))
            split=p["split"]=="holdout" if path==OUT else np.ones(len(x),bool)
            ids=p["id"] if path==OUT else p["ids"]
        for model in NAMES:
            with np.load(path/(model+".npz")) as f:
                direct=f["raw_8"]
                prefix=f["raw_16"][:,:,:8]
                external={}
                if path==OUT:
                    for h in (8,16):
                        external[h]=(np.abs(f[f"standardized_{h}"]-f[f"raw_{h}"]).mean(2)/x.std(2))
            for d in np.unique(domain):
                idx=(domain==d)&split
                w=boot_weights("horizon|" + d,int(idx.sum()))
                a=item_metrics(direct[idx],y[idx],x[idx])["mase1"]
                b=item_metrics(prefix[idx],y[idx],x[idx])["mase1"]
                gap=(np.abs(prefix[idx]-direct[idx]).mean(2)/x[idx].std(2)).mean(1)
                output.setdefault(d,{})[model]=dict(first8_from_h16_vs_direct_h8_mase1_ratio=percentile((w@b)/(w@a)),
                    mean_absolute_forecast_shift_in_past_sd=percentile(w@gap),
                    raw_h8_better_frequency=percentile(w@np.where(a==b,.5,a<b)))
                for h, errors in external.items():
                    magnitude=errors[idx]
                    worst,origin=np.unravel_index(magnitude.argmax(),magnitude.shape)
                    output[d][model][f"h{h}_raw_vs_external_z_in_past_sd"]=dict(
                        median_origin=float(np.median(magnitude)),p95_origin=float(np.quantile(magnitude,.95)),
                        max_origin=float(magnitude.max()),id=ids[idx][worst],origin=int(origin),
                        worst_history_native_sd=float(x[idx][worst,origin].std()))
    (OUT/"horizon.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
