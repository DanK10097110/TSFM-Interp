"""Series-paired uncertainty for actual forecast interventions and their clean-state scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.steering.amplify.data import OUT

AMPLIFIED = ("naive","scalar","read","shuffle_read","shuffle_both","oracle")


def interval(a):
    return {"value":float(a[0]),"ci95":np.quantile(a[1:],[.025,.975]).tolist()}


def effects_recovery(pred,target,weights,shape=False):
    if shape:
        pred=pred-pred.mean(-1,keepdims=True)
        target=target-target.mean(-1,keepdims=True)
    error=weights @ ((pred-target)**2).sum(-1)
    energy=weights @ (target**2).sum(-1)
    return 1-error/np.maximum(energy,1e-25)


def score_metrics(p,t,mean,weights):
    w=weights.sum(1)[:,None]
    xy=(weights@(p*t))-((weights@p)*(weights@t))/w
    xx=weights@(p*p)-(weights@p)**2/w
    yy=weights@(t*t)-(weights@t)**2/w
    return {"correlation":xy/np.sqrt(np.maximum(xx*yy,1e-25)),
            "r2":1-(weights@((p-t)**2))/np.maximum(weights@((t-mean)**2),1e-25)}


def summarize(est,names):
    """Include the equal-cue primary and individual endpoints from the same resamples."""
    return {"macro":interval(est.mean(1)),**{str(cue):interval(est[:,j]) for j,cue in enumerate(names)}}


def report(z):
    rng=np.random.default_rng(2026112204)
    summary={}
    names=z["cue_names"]
    for domain in ("daily","weekly"):
        sel=z["domain"]==domain
        n=int(sel.sum())
        w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=2000)])
        reads={"native_axis":z["native_score"][sel],**{k:z["score_"+k][sel] for k in ("naive","scalar","read","shuffle_read")}}
        scores={k:score_metrics(p,z["true_score"][sel],z["train_score_mean"],w) for k,p in reads.items()}
        row={"n":n,"score":{k:{m:summarize(v,names) for m,v in detail.items()} for k,detail in scores.items()}}
        row["score_read_minus"]={k:summarize(scores["read"]["r2"]-detail["r2"],names)
                                 for k,detail in scores.items() if k!="read"}
        for ref,methods in [("source_amplify",AMPLIFIED),("target_amplify",AMPLIFIED),("oracle",AMPLIFIED),
                            ("source_add",("fixed",)),("target_add",("fixed",))]:
            row[ref]={}
            for part in ("full","shape"):
                vals={k:effects_recovery(z[k][sel],z[ref][sel],w,part=="shape") for k in methods}
                detail={k:summarize(v,names) for k,v in vals.items()}
                if "read" in vals:
                    detail["read_minus"]={k:summarize(vals["read"]-vals[k],names) for k in methods if k!="read"}
                row[ref][part]=detail
            row[ref]["rms_by_cue"]=np.sqrt((z[ref][sel]**2).mean(axis=(0,2))).tolist()
        # Decompose large source-response errors into amplitude and signed direction.
        t=z["source_amplify"][sel]
        t2=w@(t*t).sum(-1)
        row["source_amplitude"]={}
        for key in ("read","oracle"):
            pred=z[key][sel]
            p2=w@(pred*pred).sum(-1)
            inner=w@(pred*t).sum(-1)
            row["source_amplitude"][key]={
                "rms_ratio":summarize(np.sqrt(p2/np.maximum(t2,1e-25)),names),
                "signed_cosine":summarize(inner/np.sqrt(np.maximum(p2*t2,1e-25)),names)}
        summary[domain]=row
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    for stem in ("timesfm_to_bolt","bolt_to_timesfm"):
        with np.load(a.out/(stem+".npz")) as data:
            result=report(dict(data))
        (a.out/(stem+"_results.json")).write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n")
        for domain,r in result.items():
            print(stem,domain,"clean score",{k:round(v["r2"]["macro"]["value"],3) for k,v in r["score"].items()})
            print("oracle-read effect",{k:round(r["oracle"]["full"][k]["macro"]["value"],3) for k in AMPLIFIED})
