"""Describe forecast level, variability, and squared error components on fixed items."""
import json
import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER
from research.activation_transfer.moment.forecast_quality.predict import NAMES


def stats(p, y, x):
    """All forecast and target coordinates are normalized to this origin's own past."""
    mu=x.mean(-1,keepdims=True)
    sd=x.std(-1,keepdims=True)
    pred=(p-mu)/sd
    actual=(y-mu)/sd
    a,pred_level=actual.mean(-1),pred.mean(-1)
    last=((x[...,-1:]-mu)/sd).squeeze(-1)
    err=pred-actual
    total=np.square(err).mean()
    def corr(u,v):
        return float(np.corrcoef(u.ravel(),v.ravel())[0,1])
    return dict(corr_forecast_actual_level=corr(pred_level,a),
                corr_forecast_last_level=corr(pred_level,last),
                corr_last_actual_level=corr(last,a),
                forecast_to_actual_level_sd=pred_level.std()/a.std(),
                normalized_level_bias=float((pred_level-a).mean()),
                fraction_forecast_error_energy_in_level=float(np.square(err.mean(-1)).mean()/total),
                normalized_waveform_mae=float(abs(err-err.mean(-1,keepdims=True)).mean()),
                normalized_level_mae=float(abs(err.mean(-1)).mean()))


def run():
    output={}
    for path in (OUT,WEATHER):
        with np.load(path/"panel.npz") as f:
            x,y=f["context"],f["future"]
            if path==WEATHER:
                domain=np.repeat("noaa",len(x))
                choose=np.ones(len(x),bool)
            else:
                domain=f["domain"]
                choose=f["split"]=="holdout"
        for model in NAMES:
            with np.load(path/(model+".npz")) as f:
                for h in (8,16):
                    pred=f["raw_"+str(h)]
                    for d in np.unique(domain):
                        idx=choose & (domain==d)
                        output.setdefault(d,{}).setdefault(model,{})[h]=stats(pred[idx],y[idx,:,:h],x[idx])
    (OUT/"level.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
