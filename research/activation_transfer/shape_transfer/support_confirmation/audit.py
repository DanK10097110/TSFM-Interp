"""Rebuild a subset of displacements independently and check actual singleton forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

from research.activation_transfer.atoms.materials import CACHE as ATOMS,MODELS,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT,adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.supports import ARM_NAMES
from tsfm_lens.sae.train import load_sae_checkpoint


def reconstruct(j,wave,weight,budget,axes=None):
    # Explicit 8x8 loss transformation and independent SVD-based damped inverse.
    n=len(wave);balance=np.full((n,n),(1-weight)/n)+np.eye(n)*weight
    design=balance@j;rhs=balance@wave
    if axes is not None:design=design@axes.T
    left,s,right=np.linalg.svd(design,full_matrices=False)
    damp=.001*s[0]**2
    value=right.T@((s/(s**2+damp))*(left.T@rhs))
    if axes is not None:value=axes.T@value
    ratio=min(1.,budget/max(np.linalg.norm(value),1e-15))
    return (value*ratio).astype('f4')


def ordinary(model,x):
    return model.predict(x[None,:],16,[.5])['quantiles'][0,:,0]


def audit(cohort,target,cache,out):
    z=dict(np.load(out/(cohort+'_'+target+'.npz')))
    src=str(z['source_model'])
    with np.load(cache/'data.npz') as f:x=f['x']
    js=np.load(cache/(src+'_early.npz'))['jac'].astype(float)
    jt=np.load(cache/(target+'_early.npz'))['jac'].astype(float)
    src_sae=load_sae_checkpoint(str(dictionary_path(src)));target_sae=load_sae_checkpoint(str(dictionary_path(target)))
    src_dec=src_sae.W_dec.detach().numpy();target_dec=target_sae.W_dec.detach().numpy()
    source=adapter_for_length(src,x.shape[1]);dst=adapter_for_length(target,x.shape[1]);ssite=MODELS[src][-1];tsite=MODELS[target][-1]
    srcsel=last_window_slice(source,z['source_pos']);dstsel=last_window_slice(dst,z['target_pos'])
    # Obtain historical clean PCA from the variable covariance eigensystem.
    h,_,fit_ids=read_materials(ATOMS,target)
    fit=np.isin(fit_ids,z['train_ids'])
    assert fit.sum()==1024
    pooled=h[fit].mean(1).astype(float);center=pooled-pooled.mean(0);d=pooled.shape[1]
    _,axes=eigh(center.T@center,subset_by_index=(d-32,d-1))
    pca=axes.T
    iso=np.linalg.qr(np.random.default_rng(2301).normal(size=(d,32)))[0].T
    result=[]
    for event in np.unique(np.linspace(0,len(z['rows'])-1,3,dtype=int)):
        r=int(z['rows'][event]);col=int(z['cols'][event]);a=int(z['atoms'][col])
        donor_vector=(-z['codes'][r,col]*src_dec[a].astype(float)).astype('f4')
        expected=js[r]@donor_vector
        dense=reconstruct(jt[r],expected,float(z['weight']),.2*z['radius'])
        np.testing.assert_allclose(dense,z['dense_vector'][event],atol=2e-5,rtol=3e-5)
        current=ordinary(dst,x[r])
        src0=ordinary(source,x[r])
        with add_native(source,ssite,srcsel,donor_vector):
            actual=ordinary(source,x[r])-src0
        np.testing.assert_allclose(actual,z['source'][event],atol=7e-5,rtol=.001)
        control=dict(id=str(z['ids'][r]),atom=a,donor_max=float(np.max(abs(actual-z['source'][event]))),arms={})
        for k in ('dense',*ARM_NAMES):
            if k=='dense':delta=dense
            else:
                if k=='pca':q=pca
                elif k=='isotropic':q=iso
                else:
                    kept=sorted(z[k+'_ids'][r])
                    q=np.linalg.svd(target_dec[kept].astype(float),full_matrices=False)[2]
                delta=reconstruct(jt[r],jt[r]@dense.astype(float),float(z['weight']),np.linalg.norm(dense.astype(float)),q)
            with add_native(dst,tsite,dstsel,delta):effect=ordinary(dst,x[r])-current
            discrepancy=float(np.max(abs(effect-z[k][event])))
            np.testing.assert_allclose(effect,z[k][event],atol=8e-5,rtol=.001)
            control['arms'][k]=discrepancy
        result.append(control)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--cache',type=Path,default=CACHE)
    a=p.parse_args();torch.set_num_threads(1)
    outcome={cohort:{model:audit(cohort,model,a.cache if cohort=='long' else a.cache/'short',a.out) for model in MODELS}
             for cohort in ('long','short')}
    (a.out/'replay.json').write_text(json.dumps(outcome,indent=2)+'\n',encoding='utf-8')
