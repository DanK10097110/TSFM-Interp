"""Score prospective distributional interventions with equal source-feature weighting."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate,ci,resampling_counts
from research.activation_transfer.shape_transfer.quantiles.geometry import HERE,components

ARMS=("median_only","joint_local","joint_fixed","wrong_width")


def crossing(base,change,active):
    rows,features=np.nonzero(active)
    bounds=base[rows]+change[rows,features]
    return float(np.mean((bounds[:,:,0]>bounds[:,:,1])|(bounds[:,:,1]>bounds[:,:,2])))


def report(out):
    result={}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+".npz")) as f:
            data=dict(f)
        model=dict(channel_mean_weights=data["gains"].tolist(),channel_shape_weights=data["balance"].tolist(),
                   donor_fd=data["donor_fd"].tolist(),target_fd=data["target_fd"].tolist(),domains={})
        for lab in np.unique(data["split"]):
            use=data["split"]==lab
            active=data["active"][use]
            eligible=active.sum(axis=0)>=8
            mask=active[:,eligible]
            draws=resampling_counts(int(use.sum()))
            ref=components(data["source"][use][:,eligible].astype(float))
            methods={k:components(data[k][use][:,eligible].astype(float)) for k in ARMS}
            methods["donor_linear"]=data["source_linear"][use][:,eligible].astype(float)
            # Forecast-only comparator with privileged access to the actual donor median change.
            # Repeating it at all three reported quantiles leaves width and asymmetry unchanged.
            shared=np.zeros_like(ref)
            shared[:,:,:,0]=ref[:,:,:,0]
            methods["repeat_observed_donor_median"]=shared
            row=dict(n=int(use.sum()),features=data["atoms"][eligible].tolist(),firing=int(mask.sum()),
                     capped={k:int(data[k+"_clipped"][use][:,eligible].sum()) for k in ARMS},scores={})
            for name,ref_ in [("donor","source"),("target_joint","joint_local"),("target_median","median_only")]:
                base=data[("donor" if ref_=="source" else "target")+"_baseline"][use]
                delta=data[ref_][use][:,eligible]
                row[name+"_clean_crossing_rate"]=crossing(base,np.zeros_like(delta),mask)
                row[name+"_edited_crossing_rate"]=crossing(base,delta,mask)
            for comp,term in enumerate(("median","halfwidth","asymmetry")):
                for centered in (False,True):
                    name=term+("_shape" if centered else "_full")
                    reference=ref[:,:,:,comp]
                    if centered:
                        reference=reference-reference.mean(axis=2,keepdims=True)
                    scores,boots={},{}
                    for method,v in methods.items():
                        y=v[:,:,:,comp]
                        if centered:
                            y=y-y.mean(axis=2,keepdims=True)
                        per,score,boot=aggregate(mask,y,reference,draws)
                        scores[method]=dict(skill=ci(score["skill"],boot["skill"]),cosine=score["cosine"],
                            source_rms=score["source_rms"],measurable=int(np.isfinite(per["skill"]).sum()))
                        boots[method]=boot["skill"]
                    for other in ("median_only","joint_fixed","wrong_width"):
                        diff=scores["joint_local"]["skill"]["value"]-scores[other]["skill"]["value"]
                        scores["joint_minus_"+other]=ci(diff,boots["joint_local"]-boots[other])
                    row["scores"][name]=scores
            model["domains"][str(lab)]=row
        result[source]=model
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    report(p.parse_args().out)
