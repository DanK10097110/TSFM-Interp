"""Capture physically aligned 32-observation views and full native H16 Jacobians."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.extraction.hooks import token_patch
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as TRAIN
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as TEST
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.physical_ramp.study import true_captures

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('TSFM_CROSS_FAMILY_RESULTS',HERE/'results'))
CACHE=Path(os.environ.get('TSFM_BOLT_CROSSCODER','/tmp/tsfm_bolt_timesfm_crosscoder'))
VIEWS=('timesfm','bolt')
SEEDS=(22112401,22112402)
METHODS=('joint','masked','shuffled')
TRAIN_COUNT=768
PATCHES=15

def fingerprint(arr):
    return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def histories():
    with np.load(TRAIN/'data.npz') as f:
        keep=f['split']=='train';xt=f['x'][keep];it=f['ids'][keep]
    with np.load(TEST/'data.npz') as f:
        xe=f['x'];ie=f['ids'];dom=f['split']
    assert xt.shape==(1024,480) and xe.shape==(256,480)
    assert not set(it) & set(ie)
    assert len({fingerprint(z) for z in np.vstack((xt,xe))})==1280
    return xt,it,xe,ie,dom

def groups(spans):
    """Partition each physical 32-observation interval into its complete native data tokens."""
    spans=np.asarray(spans)
    rows=[]
    for start in range(0,480,32):
        ids=np.flatnonzero((spans[:,0]>=start)&(spans[:,1]<=start+32)&(spans[:,1]>spans[:,0]))
        assert len(ids)>0
        assert spans[ids[0],0]==start and spans[ids[-1],1]==start+32
        np.testing.assert_array_equal(spans[ids[:-1],1],spans[ids[1:],0])
        rows.append(ids)
    return np.stack(rows)

def archived_endpoint(cache,model,ids):
    state,pos,item=read_materials(cache,model)
    if len(item)!=len(ids):
        with np.load(cache/'data.npz') as f:
            keep=f['split']=='train'
        state=state[keep];item=item[keep]
    np.testing.assert_array_equal(item,ids)
    return state,pos

def capture(model):
    xt,it,xe,ie,dom=histories();a=adapter_for(model);site=MODELS[model][-1]
    pos=groups(a.token_time_spans());assert pos.shape==(15,1 if model=='timesfm' else 2)
    # Request real H16 on each clean input, exactly as in the intervention.
    all_states,forecasts=true_captures(a,site,np.vstack((xt,xe)),pos.ravel(),24)
    full=all_states.reshape(len(all_states),15,pos.shape[1],-1)
    delta=[]
    for v,cache,ids in [(full[:1024],TRAIN,it),(full[1024:],TEST,ie)]:
        orig,oldpos=archived_endpoint(cache,model,ids)
        np.testing.assert_array_equal(pos[-1],oldpos)
        err=float(np.max(abs(v[:,-1]-orig)));delta.append(err)
        np.testing.assert_allclose(v[:,-1],orig,atol=8e-5,rtol=2e-5)
    with np.load(TEST/(model+'_early.npz')) as check:
        clean_err=float(abs(forecasts[1024:]-check['forecast']).max())
        np.testing.assert_allclose(forecasts[1024:],check['forecast'],atol=.00012,rtol=4e-5)
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    tmp=CACHE/(model+'_pooled.partial.npy');np.save(tmp,full.mean(2));tmp.replace(CACHE/(model+'_pooled.npy'))
    np.savez_compressed(OUT/(model+'_alignment.npz'),train_ids=it,val_ids=it[768:],test_ids=ie,domains=dom,token_ids=pos,
            spans=a.token_time_spans(),group_intervals=np.c_[np.arange(0,480,32),np.arange(32,481,32)],
            ordinary=forecasts[1024:],train_x_sha=fingerprint(xt),test_x_sha=fingerprint(xe),pooled_sha=fingerprint(full.mean(2)),
            max_old_hidden=delta,max_old_forecast=clean_err,site=site,checkpoint=MODELS[model][2],revision=MODELS[model][3])
    print(model,'15 physical intervals',delta,clean_err,flush=True)

def jacobians(model,batch=8):
    _,_,x,ids,domains=histories();h,pos=archived_endpoint(TEST,model,ids)
    a=adapter_for(model);site=MODELS[model][-1];sel=last_window_slice(a,pos)
    rows=[];base=[]
    for lo in range(0,len(x),batch):
        z=torch.from_numpy(h[lo:lo+batch])
        d=torch.zeros_like(z[:,0],requires_grad=True)
        with torch.enable_grad(),token_patch(a.module,site,sel,z+d[:,None,:]):
            y=a.predict_differentiable(x[lo:lo+batch],16)
            j=[torch.autograd.grad(y[:,n].sum(),d,retain_graph=n<15)[0] for n in range(16)]
        rows.append(torch.stack(j,dim=1).detach().numpy());base.append(y.detach().numpy())
    jac=np.concatenate(rows);base=np.concatenate(base)
    with np.load(TEST/(model+'_early.npz')) as prior:
        np.testing.assert_array_equal(ids,prior['ids']);first=float(np.max(abs(jac[:,:8]-prior['jac'])))
        np.testing.assert_allclose(jac[:,:8],prior['jac'],atol=2e-4,rtol=2e-4)
        ordinary=prior['forecast'];np.testing.assert_allclose(base,ordinary,atol=.00012,rtol=4e-5)
    ix=np.r_[np.arange(0,8),np.arange(128,136)]
    move=np.random.default_rng(22112492).normal(size=(len(ix),h.shape[-1])).astype('f4')
    move*=.002*np.median(np.linalg.norm(h.mean(1),axis=1))/np.linalg.norm(move,axis=1)[:,None]
    hi,lo=[forecast16(a,site,x[ix],h[ix],sign*move,sel,8) for sign in [1,-1]]
    measured=(hi-lo)/2;linear=np.einsum('nhd,nd->nh',jac[ix],move)
    errors=[float(np.linalg.norm(v-u)/np.linalg.norm(u)) for v,u in zip(np.split(linear,2,axis=1),np.split(measured,2,axis=1))]
    assert np.max(errors)<.1,errors
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    tmp=CACHE/(model+'_jac.partial.npy');np.save(tmp,jac);tmp.replace(CACHE/(model+'_jac.npy'))
    np.savez_compressed(OUT/(model+'_jac_check.npz'),ids=ids,domains=domains,positions=pos,source_hash=fingerprint(x),
                        jac_sha=fingerprint(jac),leading_max=first,clean_max=float(abs(base-ordinary).max()),
                        rows=ix,fd_relative=errors,finite=measured,linear=linear)
    print(model,'jacobian',jac.shape,errors,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capture','jac']);p.add_argument('model',choices=VIEWS);a=p.parse_args()
    torch.set_num_threads(2)
    capture(a.model) if a.stage=='capture' else jacobians(a.model)
