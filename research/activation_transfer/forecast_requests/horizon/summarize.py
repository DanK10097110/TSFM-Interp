"""Score verified long-horizon forecasts against sustained and ending lifts."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
TARGETS=("bolt","timesfm","timer")
AMP=.05


def scores(x):
    """One row per history, unity for an exact match to the declared request."""
    if x.ndim!=2 or x.shape[1]!=32 or not np.isfinite(x).all():
        raise ValueError("expected 32 finite forecast changes per series")
    normalized=x.astype(float)/AMP
    out={}
    for label,block in [("early",normalized[:,:8]),("late",normalized[:,8:])]:
        out[label+"_gain"]=block.mean(1)
        out[label+"_energy"]=(block**2).mean(1)
        out[label+"_skill"]=1-((block-1)**2).mean(1)
    out["sustained32"]=1-((normalized-1)**2).mean(1)
    localized=np.r_[np.ones(8),np.zeros(24)]
    out["localized32"]=1-((normalized-localized)**2).sum(1)/8
    return out


def average(maps):
    return {k:np.mean([m[k] for m in maps],axis=0) for k in maps[0]}


def summarize(out):
    data={}
    for name in TARGETS:
        with np.load(out/(name+".npz")) as z:
            data[name]=dict(z)
    ids=data["bolt"]["ids"]
    for z in data.values():
        np.testing.assert_array_equal(ids,z["ids"])
    result={}
    for domain in ("daily","weekly"):
        idx=np.char.startswith(ids,domain+"/")
        if int(idx.sum())!=64:
            raise ValueError("The locked subset must contain 64 series per domain")
        draws=np.random.default_rng(2026112401).integers(64,size=(4000,64))
        def ci(v):
            return dict(value=float(v.mean()),ci95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())
        raw=[]
        groups={}
        for target,z in data.items():
            for i in range(2):
                effect=z[str(i)+"_effect"][idx]
                names=list(z[str(i)+"_arms"])
                for source in TARGETS:
                    if source==target:
                        continue
                    e={"native":[effect[:,names.index("native")]],"transfer":[effect[:,names.index(source)]],
                       "shuffle":[effect[:,names.index(f"{source}_shuffle{j}")] for j in range(4)]}
                    arm={a:average([scores(v) for v in series]) for a,series in e.items()}
                    arm["transfer_minus_shuffle"]={k:arm["transfer"][k]-arm["shuffle"][k] for k in arm["transfer"]}
                    raw.append((target,i,source,arm))
                    groups[f"{source}_to_{target}_depth_{i}"]={
                        a:{"metrics":{k:ci(v) for k,v in m.items()},
                           "mean_effect_by_lead":np.mean(e[a],axis=(0,1)).tolist()} for a,m in arm.items() if a in e}
                    groups[f"{source}_to_{target}_depth_{i}"]["transfer_minus_shuffle"]={k:ci(v) for k,v in arm["transfer_minus_shuffle"].items()}
        condition={}
        for name,records in [("all",raw),*[(t,[v for v in raw if v[0]==t]) for t in TARGETS],
                             *[(f"depth_{i}",[v for v in raw if v[1]==i]) for i in range(2)]]:
            condition[name]={arm:{k:ci(v) for k,v in average([row[3][arm] for row in records]).items()}
                             for arm in ("native","transfer","shuffle","transfer_minus_shuffle")}
        result[domain]={"n":64,"aggregate":condition,"cells":groups}
    (out/"summary.json").write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    summarize(p.parse_args().out)
