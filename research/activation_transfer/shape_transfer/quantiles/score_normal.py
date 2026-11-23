"""Score decile detail against both linear and Gaussian interpolation on separate histories."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.quantiles.geometry import HERE
from research.activation_transfer.shape_transfer.quantiles.interior import ANCHOR, INNER
from research.activation_transfer.shape_transfer.quantiles.normal_score import NORMAL
from research.activation_transfer.shape_transfer.quantiles.score_interior import FRACTIONS, displacements
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from research.activation_transfer.atoms.scores import resampling_counts


def run(out):
    learned = json.loads((out/"interior.json").read_text(encoding="utf-8"))
    report = {}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+"_normal.npz")) as f:
            d = dict(f)
        rows = {}
        for domain in np.unique(d["split"]):
            use = d["split"]==domain
            active = d["active"][use]
            keep = active.sum(axis=0)>=8
            mask = active[:,keep]
            nf = int(keep.sum())
            row = dict(n=len(mask), features=d["atoms"][keep].tolist(), active_cases=int(mask.sum()),
                       caps={a:int(d[a+"_capped"][use][:,keep].sum()) for a in ("three","five")}, segments={})
            draws = resampling_counts(len(mask),draws=3000)
            fdraws = np.random.default_rng(870012).multinomial(nf,np.full(nf,1/nf),size=len(draws))
            for segment,leads in (("early",slice(0,8)),("late",slice(8,16))):
                arcs = {a:d[a][use][:,keep,leads].astype(float) for a in ("source","three","five")}
                # Represent each candidate with six actual effects relative to median and its own two tail displacements.
                predictions = {a:(v[...,INNER]-v[...,4,None],displacements(v[...,ANCHOR])) for a,v in arcs.items()}
                anchor_inputs = {"observed":predictions["source"][1]}
                if segment=="early":
                    med,width,asym = np.moveaxis(d["linear"][use][:,keep,:,:3].astype(float),-1,0)
                    anchor_inputs["linear"] = displacements(np.stack((med+asym-width,med,med+asym+width),axis=-1))
                slope = np.asarray(learned[source]["per_lead_weights"])[leads]
                for input_name,tails in anchor_inputs.items():
                    for mode,w in (("uniform",FRACTIONS),("normal",NORMAL),("cal_learned",slope)):
                        predictions[input_name+"_"+mode] = (w*tails,tails)
                endpoints = {}
                for name, subtract, cols in (
                    ("deviation",0.,np.arange(6)),("linear_residual",FRACTIONS,np.arange(6)),
                    ("normal_residual",NORMAL,np.arange(6)),("exposed_30_70",NORMAL,[1,4]),
                    ("withheld_20_40_60_80",NORMAL,[0,2,3,5])):
                    ref = predictions["source"]
                    truth = (ref[0]-subtract*ref[1])[...,cols].reshape(*mask.shape,-1)
                    estimates = {a:(v-subtract*t)[...,cols].reshape(*mask.shape,-1) for a,(v,t) in predictions.items() if a!="source"}
                    values = {}
                    def combined(p,bs,both):
                        return dict(value=float(p),ci95=np.nanquantile(bs,[.025,.975]).tolist(),
                                    ci95_features_and_histories=np.nanquantile(both,[.025,.975]).tolist())
                    boot = {}
                    for a,v in estimates.items():
                        p,b,both,valid=intervals(mask,v,truth,draws,fdraws)
                        values[a]=dict(**combined(p,b,both),n_measurable=int(valid.sum()))
                        boot[a]=(p,b,both)
                    contrasts=[("five","three")]
                    if segment=="early":
                        contrasts.extend([("three","linear_normal"),("three","linear_cal_learned")])
                    for a,b in contrasts:
                        delta=[left-right for left,right in zip(boot[a],boot[b])]
                        values[a+"_minus_"+b] = combined(*delta)
                    values["reference_rms_pooled"]=float(np.sqrt(np.mean(truth[mask]**2)))
                    endpoints[name]=values
                # Count any price paid in previously matched anchor responses.
                for metric in ("median_shape","width","asymmetry"):
                    def anchor_metric(v):
                        med=v[...,4]
                        if metric=="median_shape":
                            return med-med.mean(axis=-1,keepdims=True)
                        return (v[...,8]-v[...,0])/2 if metric=="width" else (v[...,8]+v[...,0])/2-med
                    rr=anchor_metric(arcs["source"])
                    stats={}
                    for a in ("three","five"):
                        p,b,both,valid=intervals(mask,anchor_metric(arcs[a]),rr,draws,fdraws)
                        stats[a]=combined(p,b,both)
                    endpoints[metric]=stats
                row["segments"][segment]=endpoints
            rows[str(domain)]=row
        report[source]=dict(normal_weights=NORMAL.tolist(),fd_source=d["fd_source"].tolist(),
                            fd_target=d["fd_target"].tolist(),domains=rows)
    (out/"normal.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    run(p.parse_args().out)
