"""Site-specific intervention fidelity, sensitivity, dose, and paired final-minus-first effects."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate,ci,resampling_counts
from research.activation_transfer.shape_transfer.depth.geometry import HERE

ARMS=("fixed_5","fixed_20","local_5","local_20","wrong_20")


def summarize(out):
    report={}
    for target,source in [("bolt","timesfm"),("timesfm","bolt")]:
        with np.load(out/(source+"_reference.npz")) as native:
            full,ids=native["effects"],native["ids"]
        files=sorted(out.glob(target+"_[0-9]*.npz"),key=lambda f:int(f.stem.split("_")[1]))
        if len(files)!=5:
            raise ValueError(f"Missing depth for {target}")
        target_rows,comparison=[],{}
        first={}
        for file_num,file in enumerate(files):
            with np.load(file) as f:
                data=dict(f)
            np.testing.assert_array_equal(ids,data["ids"])
            for domain in np.unique(data["split"]):
                use=data["split"]==domain
                active=data["active"][use]
                keep=active.sum(axis=0)>=8
                m=active[:,keep]
                singular=data["singular_values"][use]
                row=dict(domain=str(domain),layer=str(data["layer"]),depth=int(file.stem.split("_")[1]),
                         n=int(use.sum()),n_features=int(keep.sum()),n_firing=int(m.sum()),radius=float(data["radius"]),
                         median_leading_sensitivity=float(np.median(singular[:,0])*data["radius"]),
                         median_eighth_sensitivity=float(np.median(singular[:,-1])*data["radius"]),
                         cap_count={k:int(data[k+"_clipped"][use][:,keep].sum()) for k in ARMS},
                         median_write_fraction={k:float(np.median(data[k+"_raw_norm_fraction"][use][:,keep][m]))
                                                for k in ("fixed","local","wrong")},scores={})
                draws=resampling_counts(int(use.sum()))
                for l,label in [(slice(0,8),"early"),(slice(8,16),"late")]:
                    for centered in (False,True):
                        tag=label+("_shape" if centered else "_full")
                        ref=full[use][:,keep,l].astype(float)
                        if centered:
                            ref-=ref.mean(axis=-1,keepdims=True)
                        entries={}
                        for arm in ARMS:
                            y=data[arm][use][:,keep,l].astype(float)
                            if centered:
                                y-=y.mean(axis=-1,keepdims=True)
                            _,point,boot=aggregate(m,y,ref,draws)
                            entries[arm]=ci(point["skill"],boot["skill"])
                            key="_".join((str(domain),tag,arm))
                            if file_num==0:
                                first[key]=(point["skill"],boot["skill"])
                            if file_num==len(files)-1:
                                p,b=first[key]
                                comparison[key]=ci(point["skill"]-p,boot["skill"]-b)
                        row["scores"][tag]=entries
                target_rows.append(row)
        report[target]=dict(rows=target_rows,final_minus_first=comparison)
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    summarize(p.parse_args().out)
