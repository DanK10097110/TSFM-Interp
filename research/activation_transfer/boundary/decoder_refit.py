"""Fit an affine decoder residual correction on 128 clean daily calibration histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import BOOT, GRID, SPLITS, Ridge, ci, metric_vectors, read_data
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS, dictionary_path
from research.activation_transfer.boundary.smooth import smooth_codes


@torch.inference_mode()
def codes(sae,h):
    pre=sae.pre_activations(torch.tensor(h))
    soft,_=smooth_codes(pre,sae.k)
    return {'hard':sae.sparsify(pre).numpy(),**{k:v.numpy() for k,v in soft.items()}}


def corrected_decoder(z,h,wd,bias):
    n,v=SPLITS['train'][1],SPLITS['valid'][1]
    residual=h-z@wd-bias
    model=Ridge(z[:n],residual[:n],n-1)
    val=model.transform(z[n:v])
    losses=[np.square(model.predict(val,c)-residual[n:v]).mean() for c in GRID]
    penalty=GRID[np.argmin(losses)]
    v_r=model.y/(model.vals+penalty*model.scale)[:,None]
    change=(model.train.T@model.U)@v_r
    corrected=(wd+change).astype(np.float32)
    intercept=(bias+model.mean_y-model.mean_x@change).astype(np.float32)
    return corrected,intercept,penalty


def fvu(clean,approx,draws):
    h=clean.astype(np.float64)
    err=np.square(approx.astype(np.float64)-h).sum(-1)
    # Recenter each bootstrap to its own test conditional mean.
    e=np.square(h).sum(-1)
    denominator=e.sum()-np.square(h.sum(0)).sum()/len(h)
    value=err.sum()/denominator
    bootstrap=[]
    for chunk in np.array_split(draws,40):
        mean_sums=h[chunk].sum(1)
        den=e[chunk].sum(1)-np.square(mean_sums).sum(-1)/len(h)
        bootstrap.extend(err[chunk].sum(1)/den)
    bootstrap=np.array(bootstrap)
    return float(value),bootstrap


def interval(pt,bt):
    return {'value':float(pt),'ci95':np.quantile(bt,[.025,.975]).tolist()}


def run(model,scratch):
    sae=load_sae_checkpoint(str(dictionary_path(model)))
    tr=read_data(scratch,'m4',model)
    z=codes(sae,tr.clean)
    wd=sae.W_dec.detach().numpy(); b=sae.b_dec.detach().numpy()
    params={}; out={'model':model,'decoder':{},'test':{}}
    for arm,v in z.items():
        mat,bias,pen=corrected_decoder(v,tr.clean,wd,b)
        active=(v[:128]>0).any(0)
        rot=metric_vectors(mat[active],wd[active])['cosine']
        params[arm]=(mat,bias)
        out['decoder'][arm]={'penalty':pen,'calibration_atoms':int(active.sum()),
                             'decoder_direction_cosine_quantiles':np.quantile(rot,[0,.1,.5,.9,1]).tolist()}
    for dataset in ('m4','weekly','fast'):
        d=tr if dataset=='m4' else read_data(scratch,dataset,model)
        ix=slice(*SPLITS['test']) if dataset=='m4' else slice(None)
        h=d.clean[ix];alt=d.states[ix,:,1].copy();dx=alt-h[:,None]
        zz,za=codes(sae,h),codes(sae,alt)
        draws=np.random.default_rng(20261120).integers(len(h),size=(BOOT,len(h)))
        values={}; fv={}; cosine={}
        for arm in zz:
            for tag,(w,bb) in [('original',(wd,b)),('refit',params[arm])]:
                key=arm+'_'+tag
                rec=zz[arm]@w+bb
                pred=(za[arm]-zz[arm][:,None])@w
                pt,bt=fvu(h,rec,draws)
                v=metric_vectors(pred,dx)
                values[key]={'fvu':interval(pt,bt),**{k:ci(r) for k,r in v.items()}}
                fv[key]=(pt,bt);cosine[key]=v['cosine']
        for a,bname in [('shoulder_refit','hard_refit'),('subtract_refit','hard_refit'),
                        ('shoulder_refit','hard_original'),('hard_refit','hard_original')]:
            values[a+'_minus_'+bname]={'fvu':interval(fv[a][0]-fv[bname][0],fv[a][1]-fv[bname][1]),
                                      'cosine':ci(cosine[a]-cosine[bname])}
        out['test'][dataset]=values
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--scratch',type=Path,default=DEFAULT_SCRATCH)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    args.out.write_text(json.dumps(run(args.model,args.scratch),indent=2,allow_nan=False)+'\n')
