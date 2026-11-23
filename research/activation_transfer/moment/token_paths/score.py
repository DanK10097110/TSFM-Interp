"""Summarize physical cross-model interventions at the history and output patch positions."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import metrics,terms
from research.activation_transfer.moment.token_paths.geometry import HERE,SLOTS
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,difference


def score(size,out):
    result={}
    for source in ("bolt","timesfm"):
        data={}
        for site in ("middle","readout"):
            with np.load(out/f"{size}_{source}_{site}.npz") as z:data[site]=dict(z)
        d=data["middle"]
        for k in ("row","col","active","atoms","ids","donor","request"):
            np.testing.assert_array_equal(d[k],data["readout"][k])
        result[source]={}
        for dom in np.unique(d["split"]):
            p=Panel(d["active"],d["row"],d["col"],d["split"]==dom)
            cell=dict(n_features=int(p.keep.sum()),events=int(p.events.sum()),atoms=d["atoms"][p.keep].tolist(),
                      counts=p.mask.sum(0).tolist())
            clipped={}; sizes={}; rms={}
            for site,vals in data.items():
                for slot in SLOTS:
                    k=site+"_"+slot
                    clipped[k]=int(np.sum(vals[slot+"_scale"][p.events]<1))
                    sizes[k]=float(np.median(vals[slot+"_length"][p.events]))
                    rms[k]=[float(np.sqrt(np.mean(vals[slot][p.events,sl].astype(float)**2))) for sl in (slice(8),slice(8,16))]
            cell.update(clipped=clipped,median_write_length=sizes,rms_early_late=rms)
            for tag,sl in (("early",slice(8)),("late",slice(8,16))):
                for kind in ("shape","full","mean"):
                    ref=component(d["donor"][:,sl],kind)
                    estimates={}
                    per={}
                    for site,vals in data.items():
                        for slot in SLOTS:
                            key=site+"_"+slot
                            y=component(vals[slot][:,sl],kind)
                            estimates[key]=p.estimate(y,ref)
                            point=metrics(terms(p.mask,p.cube(y),p.cube(ref)).sum(0))["skill"]
                            per[key]=[float(q) if np.isfinite(q) else None for q in point]
                    est={key:entry(v) for key,v in estimates.items()}
                    est["middle_history_minus_first"]=difference(estimates["middle_history"],estimates["middle_first"])
                    est["middle_history_minus_readout_first"]=difference(estimates["middle_history"],estimates["readout_first"])
                    cell[tag+"_"+kind]=est
                    cell[tag+"_"+kind+"_per_feature"]=per
            result[source][str(dom)]=cell
    (out/f"{size}_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("size",choices=("small","large"));p.add_argument("--out",type=Path,default=HERE/"results")
    a=p.parse_args();score(a.size,a.out)
