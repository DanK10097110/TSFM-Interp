"""Test how much clean reconstruction loss is removable by updating the decoder bias."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import BOOT, SPLITS, read_data
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS, dictionary_path
from research.activation_transfer.boundary.smooth import smooth_codes


@torch.inference_mode()
def decodings(sae, x):
    pre=sae.pre_activations(torch.tensor(x))
    variants,_=smooth_codes(pre,sae.k)
    variants["hard"]=sae.sparsify(pre)
    return {key:sae.decode(z).numpy().astype(np.float64) for key,z in variants.items()}


def calibrate(x, pred, method):
    if method=="original":
        return 1., np.zeros(x.shape[-1])
    xx=x-x.mean(0)
    pp=pred-pred.mean(0)
    scale=float((xx*pp).sum()/max((pp**2).sum(),1e-12)) if method=="scaled" else 1.
    return scale, x.mean(0)-scale*pred.mean(0)


def run(scratch, seed=None):
    out={}
    for model in MODELS:
        file=dictionary_path(model) if seed is None else scratch/f"seed{seed}_{model}.pt"
        sae=load_sae_checkpoint(str(file))
        train=read_data(scratch,"m4",model).clean[:128].astype(np.float32)
        training=decodings(sae,train)
        out[model]={}
        for ds in ("m4","fast","weekly"):
            x=read_data(scratch,ds,model).clean
            if ds=="m4": x=x[slice(*SPLITS["test"])]
            decs=decodings(sae,x)
            x=np.asarray(x,np.float64)
            draws=np.random.default_rng(20261120).integers(len(x),size=(BOOT,len(x)))
            # Fixed assessment-set variance, same denominator for every paired comparison.
            var=((x-x.mean(0))**2).sum(axis=-1)
            group={}
            squared={}
            for enc, rec in decs.items():
                group[enc]={}
                for method in ("original","bias","scaled"):
                    scale,bias=calibrate(train.astype(np.float64),training[enc],method)
                    error=((scale*rec+bias-x)**2).sum(axis=-1)
                    squared[enc,method]=error
                    group[enc][method]={"fvu":float(error.sum()/var.sum()),
                                        "calibration_scale":scale,"offset_norm":float(np.linalg.norm(bias)),
                                        "ci95":np.quantile(error[draws].sum(-1)/var[draws].sum(-1),
                                                           [.025,.975]).tolist()}
            group["minus_hard"]={}
            for enc in ("shoulder","subtract"):
                for method in ("original","bias","scaled"):
                    delta=squared[enc,method]-squared["hard",method]
                    group["minus_hard"][f"{enc}_{method}"]={
                        "delta_fvu":float(delta.sum()/var.sum()),
                        "ci95":np.quantile(delta[draws].sum(-1)/var[draws].sum(-1),[.025,.975]).tolist()}
            out[model][ds]=group
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch",type=Path,default=DEFAULT_SCRATCH)
    p.add_argument("--seed",type=int,default=None)
    p.add_argument("--out",type=Path,required=True)
    ns=p.parse_args()
    torch.set_num_threads(2)
    ns.out.write_text(json.dumps(run(ns.scratch,ns.seed),indent=2,allow_nan=False)+"\n")
