"""Check both registered held-out ramp/step contrasts across calendar and geographic blocks."""
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.moment.forecast_quality.post_release.data import REGION

HERE=Path(__file__).resolve().parent
MOMENT=HERE/"dictionary_transfer/input_deltas/new_dates/results"
BOLT=HERE.parent/"shape_transfer/physical_ramp/new_dates/results"


def cell_error_and_reference(ref, correct, wrong, n, t):
    def center(z):
        a=z[:,:8].astype(float)
        return a-a.mean(1,keepdims=True)
    y=center(ref)
    error_delta=((center(wrong)-y)**2-(center(correct)-y)**2).sum(1)
    return error_delta.reshape(n,t), (y*y).sum(1).reshape(n,t)


def read_panels():
    with np.load(BOLT/"transfer.npz") as f:
        ids,dates=f["stations"],f["dates"]
        n,t=len(ids),len(dates)
        np.testing.assert_array_equal(f["ids"],(np.char.add(np.char.add(ids[:,None],"/"),dates[None,:])).ravel())
        bolt=cell_error_and_reference(f["truth"],f["mapped"],f["wrong_step"],n,t)
    moment=[]
    for name in ("reference","mapped","wrong_edit"):
        with np.load(MOMENT/(name+".npz")) as f:
            np.testing.assert_array_equal(ids,f["ids"])
            np.testing.assert_array_equal(dates,f["dates"])
            moment.append(f["wave"])
    return ids, dates, {"small_to_large":cell_error_and_reference(*moment,n,t),"timesfm_to_bolt":bolt}


def run():
    ids,dates,panels=read_panels()
    n,t=len(ids),len(dates)
    regions=np.array([REGION[str(i).split("_")[0]] for i in ids])
    unique=np.unique(regions)
    assert len(unique)==4
    rng=np.random.default_rng(22112452)
    sw=rng.multinomial(n,np.full(n,1/n),10000)
    tw=rng.multinomial(t,np.full(t,1/t),10000)
    def total(v):
        return ((sw@v)*tw).sum(1)
    out=dict(stations=n,dates=dates.tolist(),region_counts={k:int((regions==k).sum()) for k in unique},comparisons={})
    for name,(num,den) in panels.items():
        across=total(num)/total(den)
        def subset(mask):
            return float(num[mask].sum()/den[mask].sum())
        regional={k:subset(regions==k) for k in unique}
        out["comparisons"][name]=dict(primary=float(num.sum()/den.sum()),
            geographic_region=regional,equally_weighted_region=float(np.mean(list(regional.values()))),
            leave_out_region={k:subset(regions!=k) for k in unique},
            by_date=(num.sum(0)/den.sum(0)).tolist(),leave_one_date=((num.sum()-num.sum(0))/(den.sum()-den.sum(0))).tolist(),
            crossed_station_date_95=np.quantile(across,[.025,.975]).tolist(),
            two_contrast_bonferroni_97_5=np.quantile(across,[.0125,.9875]).tolist())
    (HERE/"confirmation.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
