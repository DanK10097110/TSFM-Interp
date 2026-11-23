"""Check cross-study reuse by domain-qualified source ID and by exact normalized input bytes."""
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.replication.data import CACHE as OLD
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.depth.feedback.data import SCRATCH as MACHINES
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as EXTRA
from research.activation_transfer.shape_transfer.rolling.data import CACHE as ROLLING
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import CACHE as FIXED

HERE=Path(__file__).resolve().parent


def source_id(label):
    parts=str(label).split("/")
    return "/".join(parts[-2:])


def audit():
    groups={}
    def add(label,x,ids):
        keys=[source_id(v) for v in ids]
        digests=[hashlib.sha256(v.tobytes()).hexdigest() for v in x]
        groups[label]=dict(n=len(keys),items=set(keys),hashes=set(digests))
    with np.load(OLD/"data.npz") as f:
        for s in ("train","cal","daily","machines"):
            mask=f["split"]==s
            add("original_"+s,f["x"][mask],f["ids"][mask])
    for label,folder in (("additional",EXTRA),("blind",BLIND),("rolling",ROLLING)):
        with np.load(folder/"data.npz") as f:
            for d in np.unique(f["split"]):
                mask=f["split"]==d
                add(label+"_"+str(d),f["x"][mask],f["ids"][mask])
    with np.load(FIXED/"data.npz") as f:
        add("single_atom_new_sensors",f["x"],f["ids"])
    with np.load(MACHINES/"data.npz") as f:
        mask=f["split"]=="new"
        add("feedback_new_machines",f["x"][mask],f["ids"][mask])
    from research.activation_transfer.shape_transfer.dictionary_prior.refit.confirmation import TMP as GATE_CONFIRM
    with np.load(GATE_CONFIRM/"data.npz") as f:
        for domain in np.unique(f["split"]):
            mask=f["split"]==domain
            add("dictionary_gate_"+str(domain),f["x"][mask],f["ids"][mask])
    reuse=[]
    for k,l in combinations(groups,2):
        common=groups[k]["items"]&groups[l]["items"]
        same=groups[k]["hashes"]&groups[l]["hashes"]
        if common or same:
            reuse.append(dict(first=k,second=l,items=sorted(common),unique_shared_windows=len(same)))
    doc=dict(groups={k:dict(window_rows=v["n"],source_items=len(v["items"]),unique_normalized_windows=len(v["hashes"])) for k,v in groups.items()},
             exact_reuse_between_named_groups=reuse)
    (HERE/"cohort_audit.json").write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(doc)


if __name__=="__main__":
    audit()
