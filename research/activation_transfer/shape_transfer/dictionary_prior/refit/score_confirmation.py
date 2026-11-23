"""Confirm the paired gate contrast separately on new daily and weekly sources."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.dictionary_prior.refit.confirmation import ARMS,OUT
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals


def run(out, strictly_disjoint=False):
    results={}
    for model in ("bolt","timesfm"):
        for seed in (1,2):
            with np.load(out/f"confirm_{model}_{seed}.npz") as f:
                d=dict(f)
            allowed=np.ones(len(d["ids"]),bool)
            if strictly_disjoint:
                overlap=json.loads((out/"confirm_overlap.json").read_text())
                np.testing.assert_array_equal(d["ids"],overlap["ids"])
                allowed=np.array(overlap["frozen_order_mask"],dtype=bool)
            case={}
            for domain in np.unique(d["split"]):
                use=(d["split"]==domain)&allowed
                allmask=d["active"][use]
                keep=allmask.sum(axis=0)>=8
                nf=int(keep.sum())
                if nf==0:
                    case[str(domain)]=dict(features=0,n=int(use.sum()))
                    continue
                m=allmask[:,keep]
                row,atom=d["rows"],d["atom_index"]
                ev=use[row]&keep[atom]
                hist=resampling_counts(len(m),draws=4000)
                feature=np.random.default_rng(26112106).multinomial(nf,np.full(nf,1/nf),size=len(hist))
                z=dict(n=len(m),features=nf,atoms=d["atoms"][keep].tolist(),events=int(ev.sum()),
                       switched=int(np.count_nonzero(d["switch"][ev])),
                       capped={k:int(d[k+"_capped"][ev].sum()) for k in ARMS})
                for seg,sli in (("early",slice(0,8)),("late",slice(8,16))):
                    for part in ("full","shape"):
                        def get(key):
                            v=np.zeros((*d["active"].shape,8))
                            v[row,atom]=d[key][:,sli]
                            v=v[use][:,keep]
                            return v-v.mean(axis=-1,keepdims=True) if part=="shape" else v
                        native=get("source")
                        vals={k:intervals(m,get(k),native,hist,feature)[:3] for k in ARMS}
                        items={k:entry(v) for k,v in vals.items()}
                        for a,b in (("locked","hard"),("locked_equal","hard_equal"),("shoulder","hard"),("shoulder_equal","hard_equal")):
                            dif=tuple(x-y for x,y in zip(vals[a],vals[b]))
                            items[a+"_minus_"+b]=entry(dif)
                            if (seg,part,a,b)==("late","shape","locked","hard"):
                                items[a+"_minus_"+b]["family_one_sided_lower"]=float(np.nanquantile(dif[2],.05/8))
                        z[seg+"_"+part]=items
                case[str(domain)]=z
            results[f"{model}_donor{seed}"]=case
    name="confirmation_disjoint.json" if strictly_disjoint else "confirmation.json"
    (out/name).write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--strictly-disjoint",action="store_true")
    a=p.parse_args()
    run(a.out,a.strictly_disjoint)
