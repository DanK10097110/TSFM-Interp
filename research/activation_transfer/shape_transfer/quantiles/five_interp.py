"""Compare five-anchor transfer with a five-anchor forecast-only estimate at the same information budget."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.quantiles.geometry import HERE
from research.activation_transfer.shape_transfer.quantiles.interior import DECILES
from research.activation_transfer.shape_transfer.quantiles.normal_score import NORMAL
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals

SLOTS=[0,2,4,6,8]
HIDDEN=[1,3,5,7]
QUANTILE_X=ndtri(DECILES)
W=(QUANTILE_X[HIDDEN]-QUANTILE_X[SLOTS[:-1]])/(QUANTILE_X[SLOTS[1:]]-QUANTILE_X[SLOTS[:-1]])


def between(v,w):
    return v[...,:-1]*(1-w)+v[...,1:]*w


def from_linear(ch):
    med,width,asym,left,right=np.moveaxis(ch,-1,0)
    lo,hi=asym-width,asym+width
    return np.stack((med+lo,med+NORMAL[1]*lo+left,med,med+NORMAL[4]*hi+right,med+hi),axis=-1)


def run(out):
    report={}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+"_normal.npz")) as f:
            d=dict(f)
        model={}
        for dom in np.unique(d["split"]):
            use=d["split"]==dom
            active=d["active"][use]
            keep=active.sum(axis=0)>=8
            mask=active[:,keep]
            counts=resampling_counts(len(mask),draws=3000)
            nf=int(keep.sum())
            fcounts=np.random.default_rng(870012).multinomial(nf,np.full(nf,1/nf),size=len(counts))
            vals={k:d[k][use][:,keep,:8].astype(float) for k in ("source","three","five")}
            inputs={k:v[...,SLOTS] for k,v in vals.items()}
            predicted={k:v[...,HIDDEN] for k,v in vals.items()}
            for name,anc in (("linear",from_linear(d["linear"][use][:,keep].astype(float))),("observed",inputs["source"])):
                for tag,w in (("probability",.5),("normal",W)):
                    key=name+"_"+tag
                    inputs[key]=anc
                    predicted[key]=between(anc,w)
            domains={}
            for metric in ("deviation","three_normal_residual","five_normal_residual"):
                endpoints={}
                estimates={}
                for name,v in predicted.items():
                    anc=inputs[name]
                    # Subtract the model's own baseline forecast displacement before comparing shape.
                    if metric=="deviation":
                        null=anc[...,2,None]
                    elif metric=="three_normal_residual":
                        tails=anc[...,[0,0,4,4]]-anc[...,2,None]
                        null=anc[...,2,None]+NORMAL[[0,2,3,5]]*tails
                    else:
                        null=between(anc,W)
                    estimates[name]=(v-null).reshape(*mask.shape,-1)
                truth=estimates.pop("source")
                bs={}
                for name,y in estimates.items():
                    p,b,both,valid=intervals(mask,y,truth,counts,fcounts)
                    bs[name]=p,b,both
                    endpoints[name]=dict(value=float(p),ci95=np.nanquantile(b,[.025,.975]).tolist(),
                                         ci95_features_and_histories=np.nanquantile(both,[.025,.975]).tolist(),
                                         measurable=int(valid.sum()))
                for ctrl in ("linear_normal","observed_normal","linear_probability"):
                    p,b,c=(a-z for a,z in zip(bs["five"],bs[ctrl]))
                    endpoints["five_minus_"+ctrl]=dict(value=float(p),ci95=np.nanquantile(b,[.025,.975]).tolist(),
                                                       ci95_features_and_histories=np.nanquantile(c,[.025,.975]).tolist())
                endpoints["reference_rms"]=float(np.sqrt(np.mean(truth[mask]**2)))
                domains[metric]=endpoints
            model[str(dom)]=domains
        report[source]=model
    (out/"five_interp.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    run(p.parse_args().out)
