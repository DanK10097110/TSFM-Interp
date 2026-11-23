"""Compare depth at the same evaluation items, reporting per-cue effects of changing sites."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(folder):
    report = json.loads((folder/"results.json").read_text())
    out={}
    with np.load(folder/"effects.npz") as forecasts:
        for i,domain in enumerate(["daily","weekly"]):
            n=len(report["series_ids"][domain])
            use=slice(i*n,(i+1)*n)
            draws=np.random.default_rng(202611204).integers(n,size=(4000,n))
            cells={}
            for j,depth in enumerate(report["depths"]):
                for cue in report["cues"]:
                    estimates=[]
                    for model in report["models"]:
                        effects=forecasts[f"d{j}_{model}_{cue}"][use].astype(float)
                        target,mapped=effects[:,0],effects[:,1]
                        error=((target-mapped)**2).sum(axis=-1)
                        energy=(target**2).sum(axis=-1)
                        true=1-error.sum()/energy.sum()
                        boot=1-error[draws].sum(axis=1)/energy[draws].sum(axis=1)
                        estimates.append(np.r_[true,boot])
                    cells[depth,cue]=np.stack(estimates).mean(axis=0)
                cells[depth,"both"]=np.stack([cells[depth,c] for c in report["cues"]]).mean(axis=0)
            rows=[]
            for cue in [*report["cues"],"both"]:
                for k,v in [(0,1),(0,2),(1,2)]:
                    dl,dr=report["depths"][k],report["depths"][v]
                    result=cells[dl,cue]-cells[dr,cue]
                    rows.append(dict(cue=cue,first=dl,second=dr,value=float(result[0]),
                                     ci95=np.quantile(result[1:],[.025,.975]).tolist()))
            out[domain]=rows
    (folder/"depth_pairs.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=Path(__file__).resolve().parent/"results")
    run(p.parse_args().folder)
