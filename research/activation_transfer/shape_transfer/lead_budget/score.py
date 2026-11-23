"""Compare query placements on four identical evaluation coordinates, and on the further forecast."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.lead_budget.run import AUDIT,OUT
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals


def run(out):
    summary={}
    for model in ("bolt","timesfm"):
        with np.load(out/(model+".npz")) as f:
            data=dict(f)
        donor={}
        for domain in np.unique(data["split"]):
            use=data["split"]==domain
            a=data["active"][use]
            keep=a.sum(axis=0)>=8
            m=a[:,keep]
            nf=int(keep.sum())
            series=resampling_counts(len(m),draws=3000)
            atoms=np.random.default_rng(6060191).multinomial(nf,np.full(nf,1/nf),size=len(series))
            row=dict(n=len(m),active=int(m.sum()),atoms=data["atoms"][keep].tolist(),
                     selected_leads=(data["selected_columns"]+1).tolist(),scores={})
            for segment,leads in (("common_audit",AUDIT),("first8",np.arange(8)),
                                  ("all16",np.arange(16)),("beyond16",np.arange(16,32))):
                for centered in (True,False):
                    def segment_of(key):
                        v=np.take(data[key][use][:,keep].astype(float),leads,axis=-1)
                        return v-v.mean(axis=-1,keepdims=True) if centered else v
                    truth=segment_of("source")
                    scores={}
                    draws={}
                    def entry(p,b,c):
                        return dict(value=float(p),ci95=np.nanquantile(b,[.025,.975]).tolist(),ci95_features_and_histories=np.nanquantile(c,[.025,.975]).tolist())
                    for policy in ("first","spread","selected"):
                        for variant in ("","_regression","_interp"):
                            arm=policy+variant
                            p,b,c,ok=intervals(m,segment_of(arm),truth,series,atoms)
                            scores[arm]=dict(**entry(p,b,c),measurable=int(ok.sum()))
                            draws[arm]=(p,b,c)
                    for arm,ctrl in (("spread","first"),("selected","spread"),("spread","spread_regression"),("selected","selected_regression")):
                        scores[arm+"_minus_"+ctrl]=entry(*(x-y for x,y in zip(draws[arm],draws[ctrl])))
                    row["scores"][segment+("_shape" if centered else "_full")]=scores
            donor[str(domain)]=row
        summary[model]=donor
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
