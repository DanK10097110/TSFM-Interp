"""Score paired hidden changes, their live patches and independently fitted native writes."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def energy(p,t):
    p,t=p.astype(float),t.astype(float)
    dot=(p*t).sum(-1)
    yy=(t*t).sum(-1)
    xx=(p*p).sum(-1)
    return np.stack([dot,yy,xx,dot/np.maximum(np.sqrt(yy*xx),1e-20)],axis=-1)


def scores(item,w):
    dot,yy,xx,cos=(w@item).T
    return {"recovery":(2*dot-xx)/np.maximum(yy,1e-20),"gain":dot/np.maximum(yy,1e-20),
            "cosine":cos/w.sum(1),"rms_ratio":np.sqrt(xx/np.maximum(yy,1e-20))}


def bounds(v):
    return {"value":float(v[0]),"ci95":np.quantile(v[1:],[.025,.975]).tolist()}


def group_detail(candidates,anchor="delta"):
    detail={arm:{m:bounds(v) for m,v in stats.items()} for arm,stats in candidates.items()}
    for baseline in ("clean","clean_delta_tuned","legacy","delta_shuffle","raw"):
        if baseline in candidates:
            detail[anchor+"_minus_"+baseline]={m:bounds(v-candidates[baseline][m]) for m,v in candidates[anchor].items()}
    if "joint" in candidates:
        for baseline in ("delta","clean"):
            detail["joint_minus_"+baseline]={m:bounds(v-candidates[baseline][m]) for m,v in candidates["joint"].items()}
    return detail


def average(arms):
    return {k:{m:np.mean([v[k][m] for v in arms],axis=0) for m in arms[0][k]} for k in arms[0]}


def summarize(out):
    final,keeper={},{}
    for source,target in [("bolt","timesfm"),("timesfm","bolt")]:
        stem=source+"_to_"+target
        hidden=dict(np.load(out/(stem+"_hidden.npz")))
        f=dict(np.load(out/(stem+"_forecasts.npz")))
        np.testing.assert_array_equal(f["ids"],hidden["ids"])
        final[stem]={}
        for domain in ("daily","weekly"):
            sel=f["split"]==domain
            n=int(sel.sum())
            w=np.vstack([np.ones(n),np.random.default_rng(2026112503).multinomial(n,np.full(n,1/n),size=2000)])
            positive={}
            generic=[]
            for j,family in enumerate(hidden["families"]):
                vals={k:scores(hidden[k][sel,j],w) for k in f["natural_keys"] if k!="native"}
                positive[str(family)]=group_detail(vals)
                if j<4:
                    generic.append(vals)
            positive["four_generic_equal_weight"]=group_detail(average(generic))
            nn={k:f["natural"][sel,j] for j,k in enumerate(f["natural_keys"])}
            natural={k:scores(energy(v,nn["native"]),w) for k,v in nn.items() if k!="native"}
            measures={"hidden":positive,"natural_shift4_patch":group_detail(natural),"writes":{}}
            for i,cue in enumerate(f["cues"]):
                item_ref={"request":np.broadcast_to(f["requests"][i],(n,8)),
                          "source_native":f["source_native"][sel,i],"target_native":f["target_native"][sel,i]}
                cue_detail={}
                for which,t in item_ref.items():
                    cue_detail[which]={}
                    for j,dose in enumerate(("raw","calibrated")):
                        arms={k:scores(energy(f["effects"][sel,a,j,i],t),w) for a,k in enumerate(f["arms"])}
                        cue_detail[which][dose]=group_detail(arms)
                        keeper.setdefault((domain,str(cue),which,dose),[]).append(arms)
                    if which=="request":
                        for site in ("source_native","target_native"):
                            cue_detail[which][site]={k:bounds(q) for k,q in scores(energy(item_ref[site],t),w).items()}
                measures["writes"][str(cue)]=cue_detail
            final[stem][domain]=measures
    final["pairs_equal_weight"]={d:{} for d in ("daily","weekly")}
    for (domain,cue,which,dose),arms in keeper.items():
        summary=final["pairs_equal_weight"][domain].setdefault(cue,{}).setdefault(which,{})
        summary[dose]=group_detail(average(arms))
    (out/"summary.json").write_text(json.dumps(final,separators=(",",":"),allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    summarize(p.parse_args().out)
