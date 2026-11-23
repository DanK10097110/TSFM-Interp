"""Learn transport from clean states, paired local changes or an equal-weight mixture."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from research.activation_transfer.shape_transfer.tangents.capture import CACHE, FEATURES, NFIT, NTRAIN, OUT

GRID=np.array([1e-5, .0001, .001, .01, .1, 1.])
ARMS=("clean","clean_delta_tuned","delta","joint","delta_shuffle","legacy")


def normalize_training(x,y):
    """One block contributes unit total squared source signal; target keeps the same coordinate units."""
    scale=max(np.linalg.norm(x),1e-15)
    return x/scale,y/scale


def clean_blocks(x,y):
    xm=x[:NTRAIN].mean(0)
    ym=y[:NTRAIN].mean(0)
    return normalize_training(x[:NTRAIN]-xm,y[:NTRAIN]-ym), (x[NTRAIN:NFIT]-xm,y[NTRAIN:NFIT]-ym)


def tangent_blocks(x,y,inputs):
    """Balance item input RMS first, then family source energy, with equal number of training rows per family."""
    scale=np.maximum(np.sqrt((inputs.astype(float)**2).mean(-1,keepdims=True)),.02)
    x,y=x[:,:5]/scale[:,:5],y[:,:5]/scale[:,:5]
    prep=[normalize_training(x[:NTRAIN,j],y[:NTRAIN,j]) for j in range(5)]
    a,b=[np.stack(v,axis=1)/np.sqrt(5) for v in zip(*prep)]
    # Normalize both sides by each validation family's target energy, so selection averages relative MSE.
    val=[(x[NTRAIN:NFIT,j],y[NTRAIN:NFIT,j]) for j in range(5)]
    val=[(a/max(np.linalg.norm(b),1e-15),b/max(np.linalg.norm(b),1e-15)) for a,b in val]
    return (a,b),tuple(np.concatenate(v) for v in zip(*val))


class SpectralRidge:
    def __init__(self,x,y):
        eigen,vectors=eigh(x.T@x,check_finite=False)
        keep=eigen>max(1e-15,eigen[-1]*1e-12)
        self.eigen=eigen[keep]
        self.vectors=vectors[:,keep]
        self.cross=self.vectors.T@(x.T@y)

    def map(self,p):
        coef=self.cross/(self.eigen+p*self.eigen[-1])[:,None]
        return self.vectors@coef

    def choose(self,val):
        x,y=val
        projected=x@self.vectors
        # Quadratic validation loss from sufficient statistics, without reconstructing enormous matrices per penalty.
        cov=(projected.T@projected)*(self.cross@self.cross.T)
        linear=np.sum((projected.T@y)*self.cross,axis=1)
        losses=[]
        for p in GRID:
            inv=1/(self.eigen+p*self.eigen[-1])
            losses.append(float(np.sum(y*y)-2*(linear@inv)+inv@cov@inv))
        return float(GRID[np.argmin(losses)]),losses

    def top_coverage(self,dirs):
        sv=dirs@self.vectors
        den=(dirs*dirs).sum(-1)
        return {str(k):((sv[:,-k:]**2).sum(-1)/den).tolist() for k in (16,32,128)}


def vector_stats(pred,target):
    p,t=pred.astype(float),target.astype(float)
    dot=(p*t).sum(-1)
    a,b=(p*p).sum(-1),(t*t).sum(-1)
    cosine=dot/np.maximum(np.sqrt(a*b),1e-20)
    return np.stack([dot,b,a,cosine],axis=-1)


def fit(target,cache,out):
    source="timesfm" if target=="bolt" else "bolt"
    data=dict(np.load(cache/"inputs.npz"))
    a,b=[dict(np.load(cache/(m+"_deltas.npz"))) for m in (source,target)]
    for z in (a,b):
        np.testing.assert_array_equal(z["ids"],data["ids"])
    raw=(data["plus"]-data["minus"])/2
    x,y=a["clean"].astype(float),b["clean"].astype(float)
    dx,dy=a["delta"].astype(float),b["delta"].astype(float)
    (cx,cy),cv=clean_blocks(x,y)
    (tx,ty),tv=tangent_blocks(dx,dy,raw)
    flat=lambda z:z.reshape(-1,z.shape[-1])
    clean=SpectralRidge(cx,cy)
    tangent=SpectralRidge(flat(tx),flat(ty))
    joint=SpectralRidge(np.concatenate([cx,flat(tx)])/np.sqrt(2),np.concatenate([cy,flat(ty)])/np.sqrt(2))
    perm=np.random.default_rng(2026112502).permutation(NTRAIN)
    sham=SpectralRidge(flat(tx),flat(ty[perm]))
    all_clean=SpectralRidge(*normalize_training(x[:NFIT]-x[:NFIT].mean(0),y[:NFIT]-y[:NFIT].mean(0)))
    (rx,ry),rv=tangent_blocks(raw,dy,raw)
    raw_reg=SpectralRidge(flat(rx),flat(ry))
    candidates={"clean":(clean,cv),"clean_delta_tuned":(clean,tv),"delta":(tangent,tv),
                "joint":(joint,tv),"raw":(raw_reg,rv)}
    penalties,curves={},{}
    for k,(reg,valid) in candidates.items():
        penalties[k],curves[k]=reg.choose(valid)
    penalties.update(delta_shuffle=penalties["delta"],legacy=.01)
    regs={key:reg for key,(reg,_) in candidates.items()}
    regs.update(delta_shuffle=sham,legacy=all_clean)
    item_stats={"ids":data["ids"][NFIT:],"split":data["split"][NFIT:],"families":FEATURES}
    patches={}
    meta={"source":source,"target":target,"penalties":penalties,"val_curves":curves,
          "source_fitted_direction_top_pc_energy":{k:reg.top_coverage(a["direction"]) for k,reg in
                                                     (("clean",clean),("delta",tangent),("joint",joint))},
          "native_direction_cosine":{}}
    for key,reg in regs.items():
        weights=reg.map(penalties[key])
        test=raw[NFIT:] if key=="raw" else dx[NFIT:]
        pred=test@weights
        item_stats[key]=vector_stats(pred,dy[NFIT:])
        patches["shift_"+key]=pred[:,5].astype("f4")
        if key!="raw":
            patches["write_"+key]=a["direction"]@weights
            meta["native_direction_cosine"][key]=vector_stats(patches["write_"+key],b["direction"])[:,-1].tolist()
    patches["write_delta_shuffle"]*=np.linalg.norm(patches["write_delta"],axis=1,keepdims=True)/np.linalg.norm(
                                      patches["write_delta_shuffle"],axis=1,keepdims=True)
    out.mkdir(exist_ok=True,parents=True)
    stem=source+"_to_"+target
    np.savez_compressed(out/(stem+"_hidden.npz"),**item_stats)
    np.savez_compressed(cache/(target+"_patches.npz"),**patches)
    (out/(stem+"_map.json")).write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
    print(stem,penalties,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=("bolt","timesfm"))
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    fit(a.target,a.cache,a.out)
