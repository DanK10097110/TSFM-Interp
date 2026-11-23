"""Test whether continuous gates' gains are explained by a simple change of vector norm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import SPLITS, ci, load_study, metric_vectors, rep_for_dose, train_validate_one
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS, dictionary_path
from research.activation_transfer.boundary.smooth import smooth_codes


def norm_match(hard, soft):
    denom = torch.linalg.vector_norm(hard,dim=-1,keepdim=True).clamp_min(1e-12)
    return hard * (torch.linalg.vector_norm(soft,dim=-1,keepdim=True)/denom)


@torch.inference_mode()
def codes(sae, ds):
    h0=torch.tensor(ds.clean[:,None,:])
    h1=torch.tensor(ds.states[:,:,1,:].copy())
    a0, a1=sae.pre_activations(h0),sae.pre_activations(h1)
    z0,z1=sae.sparsify(a0),sae.sparsify(a1)
    sm0,_=smooth_codes(a0,sae.k)
    sm1,_=smooth_codes(a1,sae.k)
    reps=rep_for_dose(sae,ds,1)
    for arm in sm0:
        reps[arm]=(sm1[arm]-sm0[arm]).numpy()
        reps["norm_"+arm]=(norm_match(z1,sm1[arm])-norm_match(z0,sm0[arm])).numpy()
    mask=(z0>0)|(z1>0)
    reps["union"]=((torch.relu(a1)-torch.relu(a0))*mask).numpy()
    return reps


def run(scratch,dataset):
    train,test=load_study(scratch,"m4"),load_study(scratch,dataset)
    idx=slice(*SPLITS["test"]) if dataset=="m4" else slice(None)
    tr,te={},{}
    for m in MODELS:
        s=load_sae_checkpoint(str(dictionary_path(m)))
        tr[m], new=codes(s,train[m]),codes(s,test[m])
        te[m]={k:v[idx] for k,v in new.items()}
    out={}
    for a,b in [("bolt","timesfm"),("timesfm","bolt")]:
        # Keep the solver and held-out validation rule identical across arms.
        base, pred={},{}
        extras=["shoulder","subtract","norm_shoulder","norm_subtract","union","code","fixed_code"]
        xt={k:tr[a][k] for k in extras}
        xv={k:te[a][k] for k in extras}
        for arm in extras:
            p,_,_=train_validate_one(xt[arm],tr[b]["dense"],xv[arm],32)
            pred[arm]=metric_vectors(p.reshape(te[b]["dense"].shape),te[b]["dense"])["cosine"]
            base[arm]=ci(pred[arm])
        for m,n in [("shoulder","norm_shoulder"),("subtract","norm_subtract"),
                    ("union","fixed_code"),("union","code")]:
            base[m+"_minus_"+n]=ci(pred[m]-pred[n])
        out[a+"_to_"+b]=base
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch",type=Path,default=DEFAULT_SCRATCH)
    parser.add_argument("--out",type=Path,required=True)
    ns=parser.parse_args()
    torch.set_num_threads(2)
    results={ds:run(ns.scratch,ds) for ds in ["m4","fast","weekly"]}
    ns.out.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
