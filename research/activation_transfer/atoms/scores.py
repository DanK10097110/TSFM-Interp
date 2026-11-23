"""Measure atomic transfer in common forecast space, conditional on source feature firing."""
import argparse
import json
from pathlib import Path

import numpy as np


def terms(active,estimate,reference):
    """Per-item moments for each atom, with an explicit firing count."""
    e,r=np.asarray(estimate,dtype=float),np.asarray(reference,dtype=float)
    m=active.astype(float)
    return np.stack([m*np.mean(r*r,-1),m*np.mean(e*e,-1),m*np.mean(e*r,-1),m],axis=-1)


def metrics(sums):
    yy,xx,xy,n=np.moveaxis(sums,-1,0)
    live=(n>0)&(yy>1e-10*n)
    def mask(x):
        return np.where(live,x,np.nan)
    den=np.maximum(yy,1e-20)
    cos=xy/np.maximum(np.sqrt(yy*xx),1e-20)
    return dict(cosine=mask(cos),skill=mask((2*xy-xx)/den),gain=mask(xy/den),
                source_rms=mask(np.sqrt(yy/np.maximum(n,1))),
                candidate_rms=mask(np.sqrt(xx/np.maximum(n,1))))


def resampling_counts(n, draws=4000):
    rng=np.random.default_rng(2026112021)
    return rng.multinomial(n,np.full(n,1/n),size=draws)


def aggregate(active,pred,ref,counts):
    """Resample item clusters jointly across all atoms, then average eligible atoms equally."""
    stats=terms(active,pred,ref)
    vals=metrics(stats.sum(axis=0))
    sums=(counts@stats.reshape(len(stats),-1)).reshape(len(counts),*stats.shape[1:])
    boot=metrics(sums)
    keep=np.isfinite(vals["cosine"])
    if not np.any(keep):
        raise ValueError("No measurable reference atom")
    full={m:np.nanmean(v[keep]) for m,v in vals.items()}
    draws={m:np.nanmean(b[:,keep],axis=1) for m,b in boot.items()}
    return vals,full,draws


def ci(point,boot):
    return dict(value=float(point),ci95=np.nanquantile(boot,[.025,.975]).tolist())


def summarize(folder,source):
    with np.load(folder/(source+"_effects.npz")) as f:
        a=dict(f)
    meta=json.loads((folder/(source+"_metadata.json")).read_text())
    out={}
    for split in ("daily","electricity","cal"):
        use=a["split"]==split
        m=a["active"][use]
        n=m.sum(axis=0)
        B=resampling_counts(len(m))
        report={"n":len(m),"firing_counts":n.tolist(),"source_atoms":meta["source_atoms"]}
        for refkey,arms in [
               ("source_native",["target_dense","source_projected","target_sham","target_coactive",
                                  "target_omp1","target_omp4","target_omp16"]+[f"target_wrong{i}" for i in range(4)]),
               ("source_projected",["target_dense","source_native"]),
               ("target_dense",["target_omp1","target_omp4","target_omp16","target_coactive","target_sham"])]:
            ref=a[refkey][use]
            reports={}
            for kind in ["full","shape"]:
                native=ref if kind=="full" else ref-ref.mean(-1,keepdims=True)
                combined={}
                by_atom={}
                point,draws={},{}
                for arm in arms:
                    pred=a[arm][use]
                    if kind=="shape":
                        pred=pred-pred.mean(-1,keepdims=True)
                    per_atom,p,b=aggregate(m,pred,native,B)
                    point[arm],draws[arm]=p,b
                    by_atom[arm]={k:[float(v) if np.isfinite(v) else None for v in vals] for k,vals in per_atom.items()}
                    combined[arm]={k:ci(p[k],b[k]) for k in p}
                if refkey=="source_native":
                    wrong=[f"target_wrong{i}" for i in range(4)]
                    point["wrong_mean"]={k:np.mean([point[w][k] for w in wrong]) for k in point[wrong[0]]}
                    draws["wrong_mean"]={k:np.mean([draws[w][k] for w in wrong],axis=0) for k in draws[wrong[0]]}
                    combined["wrong_mean"]={k:ci(v,draws["wrong_mean"][k]) for k,v in point["wrong_mean"].items()}
                    for null in ["wrong_mean","target_sham"]:
                        diff=draws["target_dense"]["cosine"]-draws[null]["cosine"]
                        p=point["target_dense"]["cosine"]-point[null]["cosine"]
                        lo=ci(p,diff)
                        lo["family_lower"]=float(np.quantile(diff,.00625))
                        combined["dense_minus_"+null]=lo
                valid_atoms=[i for i,q in zip(meta["source_atoms"],by_atom[arms[0]]["cosine"]) if q is not None]
                reports[kind]=dict(summary=combined,per_atom=by_atom,eligible_ids=valid_atoms)
            report[refkey]=reports
        # Descriptive influence of each evaluated source atom on the primary wrong-atom comparison.
        per=report["source_native"]["full"]["per_atom"]
        aligned=np.array(per["target_dense"]["cosine"],dtype=float)
        wrong=np.mean([np.array(per[f"target_wrong{i}"]["cosine"],dtype=float) for i in range(4)],axis=0)
        keep=np.isfinite(aligned)&np.isfinite(wrong)
        d=aligned[keep]-wrong[keep]
        report["source_specificity_leave_one_out"]=( (d.sum()-d)/(len(d)-1)).tolist() if len(d)>1 else []
        out[split]=report
    (folder/(source+"_scores.json")).write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=["bolt","timesfm"])
    p.add_argument("--folder",type=Path,default=Path(__file__).resolve().parent/"results")
    a=p.parse_args()
    summarize(a.folder,a.source)
