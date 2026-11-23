"""Compare true receiver interventions, source-effect forecasts, and generic donor-unit derivatives."""
import argparse
import json

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.blind_frequency.score import scores
from research.activation_transfer.shape_transfer.state_predictor.fit import KEYS
from research.activation_transfer.shape_transfer.state_predictor.memory import HERE

ARMS=("mean","current",*KEYS,*(k+"_shuffled" for k in KEYS))
CONTRASTS=[(k,"mean") for k in ("current",*KEYS)]+[(k,k+"_shuffled") for k in KEYS]+[
    ("amplitude","dense"),("support","dense"),("raw","dense"),("amplitude","support")]


def settings_interval(point,series,features):
    return dict(value=float(point),series95=np.nanquantile(series,[.025,.975]).tolist(),
                crossed95=np.nanquantile(features,[.025,.975]).tolist())


def analyze(source,seed):
    with np.load(HERE/"results"/f"{source}_seed{seed}.npz") as f:
        a=dict(f)
    summary=dict(source=source,seed=seed,target=str(a["target"]),measurement_reuse=a["reuse"].tolist())
    if "max_archived_oracle_replay" in a:summary["max_oracle_replay"]=float(a["max_archived_oracle_replay"])
    domains={}
    for d in np.unique(a["split"]):
        idx=a["split"]==d
        cols=a["active"][idx].sum(0)>=8
        m=a["active"][idx][:,cols]
        p=m.shape[1]
        select=lambda x:x[idx][:,cols]
        row=dict(n=len(m),atoms=a["atoms"][cols].tolist(),firing_counts=m.sum(0).tolist(),
                 n_active=int(m.sum()),capped={k:int(select(a["cap_"+k])[m].sum()) for k in ARMS})
        domains[d]=row
        if p==0:continue
        sw=resampling_counts(len(m),4000)
        fw=np.random.default_rng(20261121084).multinomial(p,np.full(p,1/p),size=len(sw))
        z=select(a["codes"]).astype(float)
        for kind,prefix in (("physical","effect_"),("forecast","forecast_"),("unit_derivative","forecast_")):
            truth=select(a["forecast_current"] if kind=="unit_derivative" else a["native"]).astype(float)
            denom=np.where(m,z,1.)[...,None] if kind=="unit_derivative" else 1.
            truth=truth/denom
            for scale in ("full","centered"):
                ref=truth-truth.mean(-1,keepdims=True) if scale=="centered" else truth
                raw={};reported={}
                for arm in ARMS:
                    pred=select(a[prefix+arm]).astype(float)/denom
                    if scale=="centered":pred-=pred.mean(-1,keepdims=True)
                    raw[arm]=scores(pred,ref,m,sw,fw)
                    r=raw[arm]
                    detail=settings_interval(*r[:3]);detail["by_atom"]=[float(v) if np.isfinite(v) else None for v in r[3]]
                    reported[arm]=detail
                for aa,bb in CONTRASTS:
                    v,ss,ff=[raw[aa][q]-raw[bb][q] for q in range(3)]
                    r=settings_interval(v,ss,ff)
                    if seed==0 and kind=="physical" and scale=="centered" and (aa,bb) in (("dense","mean"),("amplitude","dense")):
                        r["primary_lower"]=float(np.nanquantile(ff,.003125))
                    reported[aa+"_minus_"+bb]=r
                row[kind+"_"+scale]=reported
    summary["domains"]=domains
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=["bolt","timesfm"]);p.add_argument("seed",type=int)
    a=p.parse_args()
    (HERE/"results"/f"{a.source}_seed{a.seed}_scores.json").write_text(json.dumps(analyze(a.source,a.seed),indent=2,allow_nan=False)+"\n")
