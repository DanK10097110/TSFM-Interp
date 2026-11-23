"""Measure the withheld temporal curvature beyond each waveform's own eight anchors."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.lead_budget.run import AUDIT, OUT
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals


def detail(values, cols):
    left = np.searchsorted(cols,AUDIT)-1
    alpha = (AUDIT-cols[left])/(cols[left+1]-cols[left])
    chord = (1-alpha)*values[...,cols[left]] + alpha*values[...,cols[left+1]]
    v = values[...,AUDIT] - chord
    return v - v.mean(axis=-1,keepdims=True)


def run(out):
    result = {}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+".npz")) as f:
            d = dict(f)
        result[source] = {}
        for cohort in np.unique(d["split"]):
            use = d["split"]==cohort
            active = d["active"][use]
            keep = active.sum(axis=0)>=8
            m = active[:,keep]
            series = resampling_counts(len(m),draws=3000)
            fweights = np.random.default_rng(6060191).multinomial(m.shape[1],np.full(m.shape[1],1/m.shape[1]),size=len(series))
            cohort_result = {}
            for method in ("spread","selected"):
                cols = d[method+"_columns"]
                observed = d[method+"_interp"][use][:,keep].astype(float)
                cubic = CubicSpline(cols,observed[...,cols],bc_type="natural",axis=-1)(np.arange(32))
                truth = detail(d["source"][use][:,keep].astype(float),cols)
                base = np.linalg.norm(truth[m])
                raw = d["source"][use][:,keep][...,AUDIT].astype(float)
                row = dict(residual_over_total_centered_rms=float(base/np.linalg.norm((raw-raw.mean(axis=-1,keepdims=True))[m])))
                draws = {}
                for label,pred in (("physical",d[method][use][:,keep]),("cubic",cubic),
                                   ("historical",d[method+"_regression"][use][:,keep])):
                    p,b,c,ok = intervals(m,detail(pred.astype(float),cols),truth,series,fweights)
                    draws[label] = p,b,c
                    row[label] = dict(value=float(p),ci95=np.nanquantile(b,[.025,.975]).tolist(),
                                      ci95_histories_features=np.nanquantile(c,[.025,.975]).tolist(),features=int(ok.sum()))
                for comparator in ("cubic","historical"):
                    p,b,c = [a-z for a,z in zip(draws["physical"],draws[comparator])]
                    row["physical_minus_"+comparator] = dict(value=float(p),ci95=np.nanquantile(b,[.025,.975]).tolist(),
                                                             ci95_histories_features=np.nanquantile(c,[.025,.975]).tolist())
                cohort_result[method] = row
            result[source][str(cohort)] = cohort_result
    (out/"residual.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
