"""Score the prospective partial-dose interaction controller on each daily cohort separately."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import ci, resampling_counts
from research.activation_transfer.shape_transfer.interactions.run import OUT
from research.activation_transfer.shape_transfer.interactions.score import paired_scores


def run(out):
    result={}
    for name in ("original","extra","blind"):
        with np.load(out/(name+"_conditional.npz")) as f:
            d=dict(f)
        n=len(d["ids"])
        n_pairs=len(d["indices"])
        rows,pairs=d["rows"],d["pairs"]
        freq=np.bincount(pairs,minlength=n_pairs)>=8
        mask=freq[pairs]
        r=dict(n=n,pairs=int(freq.sum()),firing=int(mask.sum()),capped={a:int(d[a+"_capped"][mask].sum()) for a in ("zero","wrong","corrected")},
               eligible_pair_ids=d["atoms"][d["indices"][freq]].tolist(),scores={})
        for stage,leads in (("early",slice(0,8)),("late",slice(8,16))):
            for centered in (True,False):
                def transform(v):
                    v=v[:,leads].astype(float)
                    return v-v.mean(axis=1,keepdims=True) if centered else v
                src,rec=transform(d["source_joint"]),transform(d["target_joint"])
                source_sum=transform(d["source_a"].astype(float)+d["source_b"])
                target_sum=transform(d["target_a"].astype(float)+d["target_b"])
                actual=src-source_sum
                raw=rec-target_sum
                inner,corner=transform(d["inner"]),transform(d["corner"])
                y={a:transform(d[a]) for a in ("corrected","wrong","zero")}
                block={}
                for tag,truth,arms in (("interaction",actual,dict(independent=raw,inner_estimate=inner,corner_estimate=corner,
                        **{k:v-target_sum for k,v in y.items()},zero_forecast=np.zeros_like(raw))),
                        ("joint",src,dict(independent=rec, source_singleton_sum=source_sum, **y))):
                    vals,boots={},{}
                    counts=resampling_counts(n,draws=3000)
                    for arm,pred in arms.items():
                        v,b=paired_scores(pred,truth,rows,pairs,n,n_pairs,counts)
                        if v is None:
                            raise ValueError("No measurable frequent interactions")
                        vals[arm],boots[arm]=v,b
                    for other in ("independent","wrong","zero"):
                        vals["corrected_minus_"+other]=ci(vals["corrected"]["value"]-vals[other]["value"],boots["corrected"]-boots[other])
                    vals["reference_pooled_rms"]=float(np.sqrt(np.mean(truth[mask]**2)))
                    block[tag]=vals
                r["scores"][stage+("_shape" if centered else "_full")]=block
        result[name]=r
    (out/"conditional.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
