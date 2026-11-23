"""Evaluate feedback against natural source removals and expose machine-level influence."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import metrics,resampling_counts,terms
from research.activation_transfer.shape_transfer.depth.feedback.data import OUT
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals


def interval(point,series,both):
    return dict(value=float(point),ci95=np.nanquantile(series,[.025,.975]).tolist(),
                ci95_features_and_histories=np.nanquantile(both,[.025,.975]).tolist())


def run(out):
    report={}
    for label in ("original","new"):
        first=None
        layers={}
        all_arms={}
        for depth in (0,10,19):
            with np.load(out/f"block{depth}.npz") as f:
                v=dict(f)
            sel=v["split"]==label
            a=v["active"][sel]
            ids=v["ids"][sel]
            k=a.sum(axis=0)>=8
            m=a[:,k]
            if first is None:
                first=m
            else:
                np.testing.assert_array_equal(first,m)
            nf=int(k.sum())
            draws=resampling_counts(len(m),draws=4000)
            fd=np.random.default_rng(811401).multinomial(nf,np.full(nf,1/nf),size=len(draws))
            row=dict(n=len(m),atoms=v["atoms"][k].tolist(),firings=m.sum(axis=0).tolist(),
                     scalar_step_counts={str(s):int(np.sum((v["scalar_step"][sel][:,k]==s)&m)) for s in (0,.25,.5,.75,1.)},
                     corrected_step_counts={str(s):int(np.sum((v["corrected_step"][sel][:,k]==s)&m)) for s in (0,.25,.5,1.)},
                     scores={})
            for period,h in (("early",slice(0,8)),("late",slice(8,16))):
                for centered in (True,False):
                    def extract(key):
                        y=v[key][sel][:,k,h].astype(float)
                        return y-y.mean(axis=-1,keepdims=True) if centered else y
                    truth=extract("source")
                    methods=["open","scalar","corrected"]+(["goal"] if period=="early" else [])
                    scores={}
                    key=period+("_shape" if centered else "_full")
                    for arm in methods:
                        pred=extract(arm)
                        point,bs,cross,valid=intervals(m,pred,truth,draws,fd)
                        all_arms[depth,key,arm]=(point,bs,cross)
                        entry=interval(point,bs,cross)
                        if key=="early_shape":
                            stat=terms(m,pred,truth)
                            total=stat.sum(axis=0)
                            per_atom=metrics(total)["skill"]
                            leave=metrics(total[None]-stat)["skill"]
                            loo=np.nanmean(leave,axis=1)
                            entry.update(per_atom=[float(s) if np.isfinite(s) else None for s in per_atom],
                                         leave_one_machine_out=[float(loo.min()),float(loo.max())],
                                         machine_giving_max_when_removed=str(ids[np.argmax(loo)]))
                        scores[arm]=entry
                    for arm,ctrl in (("scalar","open"),("corrected","scalar"),("corrected","open")):
                        scores[arm+"_minus_"+ctrl]=interval(*(a-b for a,b in zip(all_arms[depth,key,arm],all_arms[depth,key,ctrl])))
                    row["scores"][key]=scores
            layers[str(depth)]=row
        diff={k:interval(*(a-b for a,b in zip(all_arms[19,k,"open"],all_arms[10,k,"open"])))
              for k in ("early_shape","early_full","late_shape","late_full")}
        report[label]=dict(layers=layers,final_minus_middle=diff)
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
