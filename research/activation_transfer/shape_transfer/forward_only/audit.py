"""Audit forward-only calibrations against sealed CAL derivatives and independent live additions."""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.forward_only.probe import HERE
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH
from tsfm_lens.extraction.hooks import _primary, _rebuild
from tsfm_lens.sae.train import load_sae_checkpoint


@contextmanager
def add_native(adapter,layer,selector,vector):
    def hook(module,args,out):
        original=_primary(out)
        hidden=original.clone()
        hidden[:,selector(hidden.shape[1])] += torch.as_tensor(vector,dtype=hidden.dtype)
        return _rebuild(out,hidden)
    handle=dict(adapter.module.named_modules())[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def point(adapter,x):
    return adapter.predict(x[None],8,[.5])['quantiles'][0,:,0]


def audit(name,cache,grad,out):
    with np.load(out/(name+'_probes.npz')) as a,np.load(grad/(name+'_jac.npz')) as g:
        probe=dict(a)
        cal=g['split']=='cal'
        np.testing.assert_array_equal(g['ids'][cal],probe['cal_ids'])
        cal_jac=g['jac'][cal].astype(float)
        audit_cal={}
        for span in ('pc','random'):
            ref=cal_jac@a[span+'_basis'].T
            audit_cal[span+'_relative_error']=float(np.linalg.norm(a[span+'_jac']-ref)/np.linalg.norm(ref))
    donor=name
    recipient=next(m for m in MODELS if m!=donor)
    with np.load(out/(name+'_effects.npz')) as a:
        effect=dict(a)
    with np.load(cache/'data.npz') as f:
        selected=np.isin(f['split'],['daily','machines'])
        x=f['x'][selected]
        np.testing.assert_array_equal(f['ids'][selected],probe['ids'])
    cases=[]
    for domain in ('daily','machines'):
        rows=np.flatnonzero((probe['split']==domain)&effect['active'].any(axis=1))[:3]
        cases += [(i,int(np.flatnonzero(effect['active'][i])[0])) for i in rows]
    src=adapter_for(donor)
    slayer=MODELS[donor][-1]
    ssel=last_window_slice(src,probe['positions'])
    decoder=load_sae_checkpoint(str(dictionary_path(donor))).W_dec.detach().numpy()[probe['atoms']].astype(float)
    references=[]
    for i,j in cases:
        delta=decoder[j]/np.linalg.norm(decoder[j])*probe['test_steps'][i,j]
        signed=[]
        for sign in (1.,-1.):
            with add_native(src,slayer,ssel,sign*delta):
                signed.append(point(src,x[i]))
        half=(signed[0]-signed[1])/2
        gap=float(abs(half-probe['half_difference'][i,j]).max())
        np.testing.assert_allclose(half,probe['half_difference'][i,j],atol=4e-5,rtol=1e-3)
        references.append(dict(id=str(probe['ids'][i]),atom=int(probe['atoms'][j]),half_difference_abs_max=gap))
    del src
    rec=adapter_for(recipient)
    _,rpos,_=read_materials(cache,recipient)
    rsel=last_window_slice(rec,rpos)
    rlayer=MODELS[recipient][-1]
    for n,(i,j) in enumerate(cases):
        base=point(rec,x[i]); item=references[n]
        for span in ('pc','random'):
            for strategy in ('current','mean'):
                response=(probe['test_decoder_der'][i,j] if strategy=='current' else probe['cal_decoder_der'][:,j].mean(axis=0))
                raw=-probe['codes'][i,j]*(effect[span+'_library']@response)
                scale=min(1.,.2*effect['radius']/np.linalg.norm(raw))
                arm=span+'_'+strategy
                assert (scale<1.)==effect[arm+'_clipped'][i,j]
                with add_native(rec,rlayer,rsel,scale*raw):
                    actual=point(rec,x[i])-base
                np.testing.assert_allclose(actual,effect[arm][i,j],atol=8e-5,rtol=.003)
                item[arm+'_abs_max']=float(abs(actual-effect[arm][i,j]).max())
    return dict(calibration=audit_cal,cases=references)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--grad',type=Path,default=SCRATCH)
    args=p.parse_args();torch.set_num_threads(1)
    result={s:audit(s,args.cache,args.grad,args.out) for s in ('bolt','timesfm')}
    (args.out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
