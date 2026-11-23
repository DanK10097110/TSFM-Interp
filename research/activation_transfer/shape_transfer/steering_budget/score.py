"""Report realized and local shape at a fixed length, retaining donor and dense references."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from research.activation_transfer.shape_transfer.steering_budget.run import HERE,BASES,SOLVERS


class Panel:
    """Place firing-only effects back on their history × feature axes for paired resampling."""
    def __init__(self,active,row,col,domain,draws=3000):
        self.rows=row
        self.cols=col
        self.active=active
        self.domain=domain
        self.keep=active[domain].sum(0)>=8
        self.mask=active[domain][:,self.keep]
        self.events=domain[row]&self.keep[col]
        self.hdraw=resampling_counts(len(self.mask),draws=draws)
        n=int(self.keep.sum())
        self.fdraw=np.random.default_rng(221106).multinomial(n,np.full(n,1/n),size=draws) if n else None

    def cube(self,wave):
        x=np.zeros((*self.active.shape,wave.shape[-1]),dtype=float)
        x[self.rows,self.cols]=wave
        return x[self.domain][:,self.keep]

    def estimate(self,pred,truth):
        return intervals(self.mask,self.cube(pred),self.cube(truth),self.hdraw,self.fdraw)[:3]


def component(y,kind):
    y=y.astype(float)
    if kind=="mean":return y.mean(-1,keepdims=True)
    if kind=="shape":return y-y.mean(-1,keepdims=True)
    return y


def difference(a,b):
    return entry(tuple(x-y for x,y in zip(a,b)))


def score(out):
    result={}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+".npz")) as z:d=dict(z)
        result[source]={}
        for domain in np.unique(d["split"]):
            p=Panel(d["active"],d["row"],d["col"],d["split"]==domain)
            summary=dict(features=int(p.keep.sum()),events=int(p.events.sum()))
            arms=[b+"_"+s for b in BASES for s in SOLVERS]
            summary["median_budget_fraction"]={a:float(np.median(d[a+"_length"][p.events]/d["length"][p.events])) for a in arms}
            for part,sl,ref in (("early_dense",slice(8),"dense"),("early_donor",slice(8),"donor"),
                                ("late_donor",slice(8,16),"donor"),("linear_goal",slice(8),"goal")):
                for kind in ("shape","full","mean"):
                    truth=component(d[ref][:,sl],kind)
                    vals={a:p.estimate(component(d[a+("_linear" if ref=="goal" else "")][:,sl],kind),truth) for a in arms}
                    block={a:entry(e) for a,e in vals.items()}
                    for b in BASES:
                        block[b+"_optimal_minus_radial"]=difference(vals[b+"_optimal"],vals[b+"_radial"])
                    for other in ("response","pca","random"):
                        block["sae_minus_"+other+"_optimal"]=difference(vals["sae_optimal"],vals[other+"_optimal"])
                    summary[part+"_"+kind]=block
            result[source][str(domain)]=summary
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=HERE/"results")
    score(p.parse_args().out)
