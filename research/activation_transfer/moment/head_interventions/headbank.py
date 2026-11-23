"""Compare genuinely different readouts and independently trained readouts for one frozen encoder."""
from pathlib import Path
import json
import os
import numpy as np

from research.activation_transfer.moment.head_test.data import OUT as FITTED,SCRATCH
from research.activation_transfer.moment.head_test.fit import ridge

HERE=Path(__file__).resolve().parent


def bank(size):
    with np.load(FITTED/f"{size}_0_heads.npz") as f:
        prefix={"published":"interior_published","interior":"interior_hidden","terminal":"terminal_hidden"}
        heads={name:(f[key+"_weight"],f[key+"_bias"]) for name,key in prefix.items()}
    with np.load(SCRATCH/f"{size}_0/interior.npz") as f:
        h,y,split,ids=f["h"],f["y"],f["split"],f["ids"]
    n=len(h);fit=np.flatnonzero(split=="train")
    order=np.random.default_rng(22112229).permutation(fit)
    cal=split=="cal";d=h.shape[-1]
    meta={}
    for name,rows in zip(("half1","half2"),np.array_split(order,2)):
        x=h[rows].reshape(-1,d)
        target=y[rows].reshape(-1,8)
        w,b,m=ridge(x,target,h[cal].reshape(-1,d),y[cal].reshape(-1,8))
        heads[name]=(w,b)
        meta[name]=dict(**m,training_ids=ids[rows].tolist())
    (HERE/"results").mkdir(exist_ok=True,parents=True)
    summary=HERE/"results"/f"{size}_head_splits.json"
    temp=summary.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(meta,indent=2)+"\n")
    temp.replace(summary)
    return heads
