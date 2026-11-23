"""Fixed-atom dose skill and tied-score AP with shared two-way station/date resampling."""
import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from research.activation_transfer.moment.crosscoder.evaluate import (
    CACHE, EXTERNAL, ORACLE_ARMS, OUT, SOURCE_ARMS, dictionary, encode, raw_data, write_json,
)


def resample(shape, draws=3000, seed=22112418):
    """First row is the point estimate. Outer-product weights keep shared dates together."""
    rng=np.random.default_rng(seed)
    a=rng.multinomial(shape[0],np.full(shape[0],1/shape[0]),size=draws)
    if len(shape)==2:
        b=rng.multinomial(shape[1],np.full(shape[1],1/shape[1]),size=draws)
        a=(a[:,:,None]*b[:,None,:]).reshape(draws,-1)
    return np.vstack([np.ones(np.prod(shape)),a])


def ratio(a,b):
    return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)


def average_precision(scores,firing,w):
    order=np.argsort(-scores,kind="stable");score=scores[order]
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    pos=np.cumsum(w[:,order]*firing[order],axis=1)[:,ends]
    count=np.cumsum(w[:,order],axis=1)[:,ends]
    precision=np.divide(pos,count,out=np.zeros_like(pos),where=count>0)
    added=np.diff(pos,axis=1,prepend=np.zeros((len(w),1)))
    return ratio((added*precision).sum(1),pos[:,-1])


def interval(v):
    v=np.array(v);finite=v[1:][np.isfinite(v[1:])]
    return dict(estimate=None if not np.isfinite(v[0]) else float(v[0]),
                ci95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
                defined_bootstraps=int(len(finite)))


def score(y,dose,raw,w):
    truth=y>0;pred=dose>0
    skills=1-ratio(w@((y-dose)**2),w@(y**2))
    aps=np.stack([average_precision(raw[:,j],truth[:,j],w) for j in range(y.shape[1])],axis=1)
    eligible=np.isfinite(skills[0])&np.isfinite(aps[0])
    for j in np.flatnonzero(eligible):
        np.testing.assert_allclose(aps[0,j],average_precision_score(truth[:,j],raw[:,j]),atol=1e-9)
    macro={key:arr[:,eligible].mean(1) if eligible.any() else np.full(len(w),np.nan)
           for key,arr in (("skill",skills),("ap",aps))}
    summary=dict(macro={key:interval(vals) for key,vals in macro.items()},eligible_atoms=int(np.sum(eligible)),
                 per_atom_skill=[interval(v) for v in skills.T],per_atom_ap=[interval(v) for v in aps.T],
                 tp=np.sum(pred&truth,axis=0).tolist(),fp=np.sum(pred&~truth,axis=0).tolist(),
                 fn=np.sum(~pred&truth,axis=0).tolist(),fired_predicted=pred.sum(0).tolist(),
                 fired_actual=truth.sum(0).tolist())
    return macro,summary


def panel_scores(y,candidates,shape,seed,draws):
    w=resample(shape,draws=draws,seed=seed)
    assert len(y)==w.shape[1]
    arrays={};out={}
    for arm,(raw,dose) in candidates.items():
        assert np.isfinite(raw).all() and np.isfinite(dose).all() and (dose>=0).all()
        vals,report=score(y,dose,raw,w)
        arrays[arm]=vals;out[arm]=report
    for family in ("joint","masked","shuffled","oracle_joint"):
        names=[family+"1",family+"2"]
        if all(name in arrays for name in names):
            vals={k:(arrays[names[0]][k]+arrays[names[1]][k])/2 for k in ("skill","ap")}
            arrays["mean_"+family]=vals
            out["mean_"+family]=dict(macro={k:interval(v) for k,v in vals.items()})
    contrasts={}
    for name in arrays:
        if name in ("zero","constant"):
            continue
        refs=["raw","sae"]
        if name.startswith("mean_"):
            refs+=["mean_shuffled","mean_joint","mean_masked"]
        for ref in dict.fromkeys(refs):
            if ref in arrays and name!=ref:
                contrasts[name+" minus "+ref]={k:interval(v-arrays[ref][k]) for k,v in arrays[name].items()}
    return dict(n=len(y),true_counts=(y>0).sum(0).tolist(),prevalence=(y>0).mean(0).tolist(),
                true_dose_energy=(y*y).sum(0).tolist(),point_eligible=(y>0).any(0).tolist(),
                methods=out,paired_contrasts=contrasts)


def main(draws):
    data=raw_data();target=dictionary("large",22112201);atoms=data["atoms"]
    with np.load(EXTERNAL/"donor.npz") as f:
        y=f["dose"];np.testing.assert_array_equal(f["ids"],data["ids"])
        np.testing.assert_array_equal(f["dates"],data["dates"]);np.testing.assert_array_equal(f["atoms"],atoms)
    with np.load(EXTERNAL/"ordinary.npz") as f:
        live=encode(target,f["last"])[:,atoms]
        np.testing.assert_array_equal(live>0,y>0)
        np.testing.assert_allclose(live,y,atol=1e-5,rtol=1e-6)
    with np.load(CACHE/"large/eval.npz") as f:
        blind=encode(target,f["middle"][:,0])[:,atoms]
        np.testing.assert_array_equal(f["ids"],data["blind_ids"]);np.testing.assert_array_equal(f["split"],data["blind_domain"])
    cohorts={"noaa":(y,(len(data["ids"]),len(data["dates"])),22112418),
             "validation":(data["y"][768:],(256,),22112419)}
    for tag,sd in (("daily",22112420),("monash_traffic",22112421)):
        labels=blind[data["blind_domain"]==tag];cohorts[tag]=(labels,(len(labels),),sd)
    candidates={}
    mean=data["y"][:768].mean(0)
    for tag,(labels,_,_) in cohorts.items():
        z=np.zeros_like(labels);c=np.broadcast_to(mean,labels.shape)
        candidates[tag]={"zero":(z,z),"constant":(c,c)}
    for arm in (*SOURCE_ARMS,*ORACLE_ARMS):
        path=OUT/(arm+"_doses.npz")
        if not path.is_file():continue
        with np.load(path) as f:
            for key in ("atoms","dates","ids"):
                np.testing.assert_array_equal(f[key],data[key])
            np.testing.assert_array_equal(f["validation_ids"],data["m4_ids"][768:])
            candidates["noaa"][arm]=(f["raw"],f["dose"])
            candidates["validation"][arm]=(f["validation_raw"],f["validation_dose"])
            if "blind_ids" in f.files:
                np.testing.assert_array_equal(f["blind_ids"],data["blind_ids"])
                np.testing.assert_array_equal(f["blind_domain"],data["blind_domain"])
                for tag in ("daily","monash_traffic"):
                    take=f["blind_domain"]==tag
                    candidates[tag][arm]=(f["blind_raw"][take],f["blind_dose"][take])
    results=dict(atoms=atoms.tolist(),draws=draws,target_reencoding_max_abs=float(np.max(np.abs(live-y))),
                 m4_fit_positive_counts=(data["y"][:768]>0).sum(0).tolist(),fit_mean_doses=mean.tolist(),cohorts={})
    for tag,(labels,shape,sd) in cohorts.items():
        out=panel_scores(labels.astype(float),candidates[tag],shape,sd,draws)
        results["cohorts"][tag]=out
        print(tag,out["true_counts"],{name:tuple(round(v["estimate"],3) if v["estimate"] is not None else None
              for v in a["macro"].values()) for name,a in out["methods"].items()},flush=True)
    write_json(OUT/"evaluation.json",results)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws",type=int,default=3000)
    args=parser.parse_args();torch.set_num_threads(1);main(args.draws)
