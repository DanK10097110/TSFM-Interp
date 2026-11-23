"""Evaluate actual one-sided steering against the requested, signed eight-step change."""
import argparse
import json
from pathlib import Path

import numpy as np


def item_scores(effect,wanted):
    """Per-series improvement over zero, gain, and orthogonal energetic error."""
    den=np.sum(wanted**2)
    gain=effect@wanted/den
    error=effect-wanted
    return dict(skill=1-np.sum(error**2,axis=1)/den,gain=gain,
                collateral_sq=np.sum((effect-gain[:,None]*wanted)**2,axis=1)/den)


def stats(x,idx):
    return dict(value=float(x.mean()),ci95=np.quantile(x[idx].mean(1),[.025,.975]).tolist())


def summarize(folder):
    names=["bolt","timesfm","timer"]
    zips={name:np.load(folder/(name+".npz")) for name in names}
    metadata={name:json.loads((folder/(name+".json")).read_text()) for name in names}
    ids=zips[names[0]]["ids"]
    for z in zips.values():
        np.testing.assert_array_equal(ids,z["ids"])
    out=[]
    pooled={}
    for domain in ("daily","weekly"):
        select=np.char.startswith(ids,domain+"/")
        B=np.random.default_rng(202611208).integers(select.sum(),size=(4000,select.sum()))
        all_deltas={arm:[] for arm in ["zero_shot","recalibrated","shuffled","target_native","source_native"]}
        for depth_idx,depth in enumerate([.5,.75]):
            for target in names:
                for source in names:
                    if target==source:
                        continue
                    key=f"{depth_idx}_{source}"
                    fc=zips[target][key][select].astype(float)
                    for j,cue in enumerate(metadata[target]["requests"]):
                        want=zips[target]["desired"][j]
                        alternatives={"zero_shot":fc[:,j,0],
                                      "recalibrated":fc[:,j,1],
                                      "target_native":zips[target][f"{depth_idx}_native"][select,j],
                                      "source_native":zips[source][f"{depth_idx}_native"][select,j]}
                        measures={k:item_scores(value,want) for k,value in alternatives.items()}
                        null=[item_scores(fc[:,j,s],want) for s in range(2,fc.shape[2])]
                        measures["shuffled"]={m:np.mean([n[m] for n in null],axis=0) for m in null[0]}
                        scored={arm:{m:stats(v,B) for m,v in measures[arm].items()} for arm in measures}
                        out.append(dict(domain=domain,source=source,target=target,depth=depth,request=cue,
                                        scale=metadata[target]["applied_dose"][key][j],metrics=scored,
                                        advantage=stats(measures["zero_shot"]["skill"]-measures["shuffled"]["skill"],B)))
                        for arm in all_deltas:
                            all_deltas[arm].append(measures[arm]["skill"])
        pooled[domain]={k:stats(np.mean(v,axis=0),B) for k,v in all_deltas.items()}
    (folder/"summary.json").write_text(json.dumps(dict(rows=out,pooled=pooled),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    summarize(p.parse_args().folder)
