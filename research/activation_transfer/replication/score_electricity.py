"""Score the domain-informed feature experiment with paired whole-customer uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts


def summarize(path):
    with np.load(path) as f:
        a=dict(f)
    counts=a["active"].sum(axis=0)
    report=dict(n_selected_atoms=len(counts),n_customers=len(a["ids"]),atoms=a["atoms"].tolist(),
                calibration_counts=a["screen_counts"].tolist(),counts=counts.tolist(),
                matches=a["matches"].tolist(),correlation=a["matching_correlations"].tolist())
    for minimum in [1,8]:
        keep=counts>=minimum
        out=dict(n=int(keep.sum()))
        report[str(minimum)]=out
        if not keep.any():
            continue
        mask=a["active"][:,keep]
        draws=resampling_counts(len(mask))
        for kind in ["full","shape"]:
            ref=a["native"][:,keep].astype(float)
            if kind=="shape":
                ref-=ref.mean(axis=-1,keepdims=True)
            stats,points,boots={},{},{}
            for key,estimate in a.items():
                if key.split("_wrong")[0] not in ["m4_full","m4_small","electric","coactive"]:
                    continue
                pred=estimate[:,keep].astype(float)
                if kind=="shape":
                    pred-=pred.mean(axis=-1,keepdims=True)
                vals,points[key],boots[key]=aggregate(mask,pred,ref,draws)
                stats[key]={m:ci(v,boots[key][m]) for m,v in points[key].items()}
                stats[key]["per_atom_cosine"]=[float(v) if np.isfinite(v) else None for v in vals["cosine"]]
            for method in ["m4_full","m4_small","electric"]:
                name=method+"_wrong"
                arms=[method+"_wrong"+str(i) for i in range(4)]
                points[name]={k:np.mean([points[q][k] for q in arms]) for k in points[method]}
                boots[name]={k:np.mean([boots[q][k] for q in arms],axis=0) for k in points[method]}
                stats[name]={k:ci(v,boots[name][k]) for k,v in points[name].items()}
            for m,b in [("m4_full","m4_full_wrong"),("electric","m4_small")]:
                write={}
                for metric in ["cosine","skill"]:
                    diff=boots[m][metric]-boots[b][metric]
                    write[metric]=ci(points[m][metric]-points[b][metric],diff)
                    write[metric]["one_sided_family_lower"]=float(np.quantile(diff,.00625))
                stats[m+"_minus_"+b]=write
            out[kind]=stats
    path.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives",nargs="+",type=Path)
    for path in p.parse_args().archives:
        summarize(path)
