"""Use the same feature/historical-event masks and whole-history/feature intervals for each control."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import OUT


def score(out,kind):
    from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
    methods=(("active","other_history","pca","isotropic") if kind=="span" else
             ("true","hard","locked","shoulder","hard_equal","locked_equal","shoulder_equal"))
    pairs=(tuple(("active",k) for k in methods[1:]) if kind=="span" else
           (("hard_equal","hard"),("locked_equal","hard_equal"),("shoulder_equal","hard_equal"),("shoulder","hard")))
    result={}
    for model in ("bolt","timesfm"):
        for seed in (1,2):
            with np.load(out/f"{kind}_{model}_{seed}.npz") as f:
                data=dict(f)
            mask=data["mask"] if kind=="span" else data["active"]
            rows,cols=data["rows"],data["atom_index"]
            keep=mask.sum(axis=0)>=8
            ev=keep[cols]
            m=mask[:,keep]
            nfeat=m.shape[1]
            draws=resampling_counts(len(m),draws=2500)
            fw=np.random.default_rng(295710).multinomial(nfeat,np.full(nfeat,1/nfeat),size=len(draws))
            row=dict(features=nfeat,events=int(ev.sum()),capped={k:int(data[k+"_capped"][ev].sum()) for k in methods})
            if kind=="span":
                row["median_shared_ids_other_history"]=float(np.median(data["overlap"][rows[ev]]))
                row["median_squared_norm"]={k:float(np.median(data[k+"_fraction"][ev])) for k in methods}
            else:
                row["median_cosine"]={k:float(np.median(data[k+"_cosine"][ev])) for k in methods}
                row["median_length"]={k:float(np.median(data[k+"_length_ratio"][ev])) for k in methods}
                with np.load(out/f"encode_{model}_{seed}.npz") as old:
                    for k in ("hard","locked","true"):
                        np.testing.assert_allclose(data[k],old[k],atol=3e-5)
            for segment,sl in (("early",slice(0,8)),("late",slice(8,16))):
                def centered(key):
                    v=np.zeros((*mask.shape,8))
                    v[rows,cols]=data[key][:,sl].astype(float)
                    v=v[:,keep]
                    return v-v.mean(axis=-1,keepdims=True)
                truth=centered("source")
                vals={k:intervals(m,centered(k),truth,draws,fw)[:3] for k in methods}
                res={k:entry(v) for k,v in vals.items()}
                for a,b in pairs:
                    res[a+"_minus_"+b]=entry(tuple(x-y for x,y in zip(vals[a],vals[b])))
                row[segment]=res
            result[model+"_donor"+str(seed)]=row
    (out/(kind+"_summary.json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=("span","normgate"))
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    score(a.out,a.kind)
