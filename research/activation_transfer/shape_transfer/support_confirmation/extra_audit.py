"""Independently rebuild residual and wrong-derivative solves, then check actual patched waveforms."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

from research.activation_transfer.atoms.materials import CACHE as ATOMS,MODELS,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.continuation.probe import SCRATCH as CAL
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.support_confirmation.audit import ordinary,reconstruct
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT,adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.residual import PAIRS
from tsfm_lens.sae.train import load_sae_checkpoint


def audit(target,cache,out,stage):
    ref=dict(np.load(out/('short_'+target+'.npz')))
    if stage=='residual':f=dict(np.load(out/'bolt_residual.npz'))
    else:f=dict(np.load(out/(target+'_derivative.npz')))
    with np.load(cache/'data.npz') as d:x=d['x']
    with np.load(cache/(target+'_early.npz')) as d:j=d['jac'].astype(float)
    with np.load(CAL/(target+'_jac16.npz')) as d:avg=d['jac'][d['split']=='cal',:8].astype(float).mean(0)
    sae=load_sae_checkpoint(str(dictionary_path(target)));decoder=sae.W_dec.detach().numpy().astype(float)
    enc=adapter_for_length(target,x.shape[1]);sel=last_window_slice(enc,ref['target_pos']);site=MODELS[target][-1]
    if stage=='residual':
        train,_,ids=read_materials(ATOMS,target)
        h=train[np.isin(ids,ref['train_ids'])].mean(1).astype(float);h-=h.mean(0)
        _,axes=eigh(h.T@h,subset_by_index=(h.shape[1]-32,h.shape[1]-1));pca=axes.T
        random=np.linalg.qr(np.random.default_rng(2301).normal(size=(h.shape[1],32)))[0].T
    report=[]
    for event in np.unique(np.linspace(0,len(ref['rows'])-1,3,dtype=int)):
        r=int(ref['rows'][event]);goal=ref['receiver_dense_early_derivative'][event].astype(float)
        cap=float(ref['dense_length'][event]);weight=float(ref['weight'])
        def rows_at(arm,i):return np.linalg.qr(decoder[np.sort(ref[arm+'_ids'][i])].T)[0].T
        nt=rows_at('native',r);writes={}
        if stage=='residual':
            fs=rows_at('own_0',r);bank={'own_0':fs,'native':nt,'pca':pca,'isotropic':random}
            for key,(aa,bb) in PAIRS.items():
                raw=bank[aa];remove=bank[bb]
                gram=raw@remove.T
                remaining=raw-gram@remove
                # On this panel every residual has full row rank, so ordinary unpivoted QR is valid.
                assert np.linalg.matrix_rank(remaining,tol=1e-7)==32
                independent=np.linalg.qr(remaining.T)[0].T
                writes[key]=reconstruct(j[r],goal,weight,cap,independent)
            zero=reconstruct(j[r],goal,weight,cap,fs).astype(float)
            shared=(zero@nt.T)@nt
            writes['zero_shared']=shared.astype('f4');writes['zero_perpendicular']=(zero-shared).astype('f4')
        else:
            seed=2301+(event%4)
            other=int(f[f'assigned_{seed}'][r])
            fo=rows_at('native',other)
            for mode in ('balanced','dc'):
                for state,axes,which in [('NN',nt,j[r]),('NS',nt,j[other]),('SN',fo,j[r]),('SS',fo,j[other])]:
                    key=mode+'_NN' if state=='NN' else f'{mode}_{state}_{seed}'
                    if mode=='dc':query=goal.mean(keepdims=True);deriv=which.mean(0,keepdims=True);w=1.
                    else:query=goal;deriv=which;w=weight
                    writes[key]=reconstruct(deriv,query,w,cap,axes)
            writes['cal_N']=reconstruct(avg,goal,weight,cap,nt)
            writes[f'cal_S_{seed}']=reconstruct(avg,goal,weight,cap,fo)
        baseline=ordinary(enc,x[r]);errors={}
        for key,v in writes.items():
            with add_native(enc,site,sel,v):actual=ordinary(enc,x[r])-baseline
            expected=f[key][event]
            err=float(np.max(abs(actual-expected)))
            np.testing.assert_allclose(actual,expected,atol=.0001,rtol=.001)
            errors[key]=err
        report.append(dict(id=str(ref['ids'][r]),feature=int(ref['atoms'][ref['cols'][event]]),errors=errors))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['residual','derivative'])
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--cache',type=Path,default=CACHE/'short')
    a=p.parse_args();torch.set_num_threads(1);models=['bolt'] if a.stage=='residual' else list(MODELS)
    record={m:audit(m,a.cache,a.out,a.stage) for m in models}
    (a.out/(a.stage+'_replay.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
