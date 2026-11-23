"""Summarize full physical last-block substitutions by VM cohort and write size."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.depth.feedback.data import OUT
from research.activation_transfer.shape_transfer.depth.feedback.last_block import ARMS


def run(out):
    with np.load(out/"last_block.npz") as f:
        d=dict(f)
    errors={k:np.mean((d[k]-d["linear"])**2,axis=-1)**.5 for k in ARMS}
    result=dict(activation=str(d["activation_type"]),zero_clean_max=float(d["zero_dose_max"]),groups={},large_cases={})
    for cohort in np.unique(d["split"]):
        result["groups"][str(cohort)]={}
        for group,size in (("large",d["norm_fraction"]>=.05),("small",d["norm_fraction"]<.05)):
            use=(d["split"]==cohort)&size
            outrow=dict(n=int(use.sum()),pooled_rms={})
            for k,r in errors.items():
                outrow["pooled_rms"][k]=float(np.mean(r[use]**2)**.5)
            result["groups"][str(cohort)][group]=outrow
    for i in np.flatnonzero(d["norm_fraction"]>=.05):
        result["large_cases"][str(d["ids"][i])]=dict(fraction=float(d["norm_fraction"][i]),rms={k:float(v[i]) for k,v in errors.items()})
    (out/"last_block.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
