"""Separate calibration error, angle, and closed gates in measured physical atom substitutions."""
import argparse
from hashlib import sha256
import json
import numpy as np
from research.activation_transfer.moment.dictionary_transfer.match import OUT
from research.activation_transfer.shape_transfer.steering_budget.score import component


def gain(x,y):
    return max(0.,float(np.sum(x*y)/max(1e-30,np.sum(x*x))))


def evaluate(x,y,fold):
    energy=np.sum(y*y)
    if energy<=1e-20:return None
    best=gain(x,y)
    event=np.maximum(0.,np.sum(x*y,axis=1)/np.maximum(1e-30,np.sum(x*x,axis=1)))[:,None]
    fitted=np.zeros_like(x)
    cv=bool(fold.any() and (~fold).any())
    if cv:
        for mask in (fold,~fold):fitted[mask]=x[mask]*gain(x[~mask],y[~mask])
    def skill(est):return float(1-np.sum((est-y)**2)/energy)
    return dict(n=len(x),energy=float(energy),source_rms=float(np.sqrt(np.mean(y*y))),raw=skill(x),
                pooled_signed_cos=float(np.sum(x*y)/max(1e-30,np.sqrt(energy*np.sum(x*x)))),
                best_positive_gain=best,optimistic_atom=skill(best*x),optimistic_event=skill(event*x),
                crossfit=skill(fitted) if cv else None)


def run(stem):
    with np.load(OUT/(stem+"_forecast.npz")) as f:d=dict(f)
    groupfile=OUT/(stem+"_group_waves.npz")
    if groupfile.exists():
        with np.load(groupfile) as f:
            for head in ("native","interior"):
                for k in ("one","four","sixteen","gated","dose"):d[head+"_"+k]=f[k+"_"+head]
    fold=np.array([bool(sha256(("dose-choice|"+key).encode()).digest()[0]%2) for key in d["ids"]])
    r,c=d["row"],d["col"]
    report={}
    for dom in np.unique(d["split"]):
        active=d["z"][d["split"]==dom]>0
        keep=np.flatnonzero(active.sum(0)>=8)
        rec={}
        for h in ("native","interior"):
            for name in ("corr_own","corr_forced","geom_forced","dense","random","response","one","four","sixteen","gated","dose"):
                if h+"_"+name not in d:continue
                for label,sl in (("early",slice(8)),("late",slice(8,16))):
                    pred=component(d[h+"_"+name][:,sl],"shape")
                    truth=component(d["truth_"+h][:,sl],"shape")
                    values=[]
                    for idx in keep:
                        m=(c==idx)&(d["split"][r]==dom)
                        z=evaluate(pred[m],truth[m],fold[r[m]])
                        if z is not None:
                            z["atom"]=int(d["atoms"][idx]);values.append(z)
                    asum={}
                    for key in ("raw","pooled_signed_cos","optimistic_atom","optimistic_event","crossfit"):
                        finite=[(q[key],q["energy"]) for q in values if q[key] is not None]
                        if finite:
                            v,e=np.array(finite).T
                            asum[key]=dict(mean=float(v.mean()),median=float(np.median(v)),n=len(v),
                                           positive=int(np.sum(v>0)),weighted_by_source_energy=float(v@e/e.sum()))
                    rec[h+"_"+name+"_"+label]=dict(macro=asum,features=values)
        report[str(dom)]=rec
    (OUT/(stem+"_gain.json")).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stem",nargs="?");a=p.parse_args()
    for stem in ([a.stem] if a.stem else sorted(x.name.removesuffix("_forecast.npz") for x in OUT.glob("*_forecast.npz"))):run(stem)
