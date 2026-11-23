"""Build bounded interventions from eight derivatives, physically apply, and record all 16 leads."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.continuation.probe import HERE as CONTINUATION
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.span_inverse import constrained
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape,solve_adaptive
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT,adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.supports import ARM_NAMES,candidates,count_overlap
from tsfm_lens.sae.train import load_sae_checkpoint


def limit_norm(proposed,budget):
    norm=np.linalg.norm(proposed,axis=-1)
    scale=np.minimum(1.,np.asarray(budget)/np.maximum(norm,1e-15))
    return (proposed*scale[:,None]).astype('f4'),norm,scale


def archive_inputs(model,cache,ids):
    h,positions,saved_ids=read_materials(cache,model)
    np.testing.assert_array_equal(saved_ids,ids)
    with np.load(cache/(model+'_early.npz')) as f:
        np.testing.assert_array_equal(ids,f['ids'])
        jac,base,radius,error=f['jac'].astype(float),f['forecast'],float(f['radius']),float(f['fd_error'])
    assert jac.shape[1]==8
    return h,positions,jac,base,radius,error


def run(source,cohort,cache,out,batch):
    target=next(k for k in MODELS if k!=source)
    with np.load(cache/'data.npz') as f:x,ids=f['x'],f['ids']
    hs,ps,js,base_s,_,fs=archive_inputs(source,cache,ids)
    ht,pt,jt,base_t,radius,ft=archive_inputs(target,cache,ids)
    with np.load(REPLICATION/'results'/(source+'_seed0.npz')) as f:atoms=f['atoms']
    with np.load(CONTINUATION/'results'/(source+'.npz')) as f:weight=float(f['weight'])
    src_sae=load_sae_checkpoint(str(dictionary_path(source)))
    ds=src_sae.W_dec.detach().numpy()[atoms].astype(float)
    with torch.inference_mode():z=src_sae.encode(torch.from_numpy(hs.mean(1))).numpy()[:,atoms]
    rows,cols=np.nonzero(z>0)
    source_delta=(-z[rows,cols,None]*ds[cols]).astype('f4')
    goal=np.einsum('nhd,nd->nh',js[rows],source_delta.astype(float))
    proposed=solve_adaptive(boost_shape(jt[rows],weight),boost_shape(goal[:,:,None],weight))[:,:,0]
    dense,requested,dense_fraction=limit_norm(proposed,.2*radius)
    length=np.linalg.norm(dense.astype(float),axis=1)
    assert np.all(length>0)
    src=adapter_for_length(source,x.shape[1]);srcsel=last_window_slice(src,ps)
    ordinary_src=forecast16(src,MODELS[source][-1],x[rows],hs[rows],np.zeros_like(source_delta),srcsel,batch)
    np.testing.assert_allclose(ordinary_src,base_s[rows],atol=3e-5,rtol=5e-5)
    natural=forecast16(src,MODELS[source][-1],x[rows],hs[rows],source_delta,srcsel,batch)-ordinary_src
    del src
    selections,assigned,axes,counts,fit_ids=candidates(target,ht.mean(1))
    assert set(ids).isdisjoint(fit_ids)
    dst=adapter_for_length(target,x.shape[1]);select=last_window_slice(dst,pt)
    def forecast(v):return forecast16(dst,MODELS[target][-1],x[rows],ht[rows],v,select,batch)
    baseline=forecast(np.zeros_like(dense));np.testing.assert_allclose(baseline,base_t[rows],atol=3e-5,rtol=5e-5)
    rec=dict(source_model=source,target_model=target,ids=ids,atoms=atoms,rows=rows,cols=cols,
             codes=z,source=natural,dense=forecast(dense)-baseline,dense_vector=dense,
             source_early_derivative=goal,receiver_dense_early_derivative=np.einsum('nhd,nd->nh',jt[rows],dense),
             ordinary_source=ordinary_src,ordinary_target=baseline,dense_requested=requested,dense_scale=dense_fraction,
             dense_length=length,weight=weight,radius=radius,site=MODELS[target][-1],source_pos=ps,target_pos=pt,
             fd_source=fs,fd_receiver=ft,train_ids=fit_ids,frequency_counts=counts,arms=ARM_NAMES,context=x.shape[1])
    for label,assignment in assigned.items():rec[label+'_assignment']=assignment
    for label,labels in selections.items():
        rec[label+'_ids']=labels;rec[label+'_overlap']=count_overlap(labels,selections['native'])
    for arm,q in axes.items():
        responses=[];norms=[];scales=[];hashes=[]
        for i in range(0,len(rows),batch):
            seg=slice(i,i+batch);r=rows[seg]
            basis=q[r] if q.ndim==3 else np.broadcast_to(q,(len(r),*q.shape))
            raw=constrained(jt[r],basis,dense[seg].astype(float),weight)
            change,norm,scale=limit_norm(raw,length[seg])
            assert np.all(np.linalg.norm(change.astype(float),axis=1)<=length[seg]+1e-5)
            wave=forecast16(dst,MODELS[target][-1],x[r],ht[r],change,select,batch)-baseline[seg]
            responses.append(wave);norms.append(norm);scales.append(scale)
            hashes.extend(sha256(v.tobytes()).hexdigest() for v in change)
        rec[arm]=np.concatenate(responses);rec[arm+'_length']=np.concatenate(norms)*np.concatenate(scales)
        rec[arm+'_scale']=np.concatenate(scales);rec[arm+'_raw_hash']=hashes
        print(cohort,source,'->',target,arm,len(rows),flush=True)
    np.savez_compressed(out/(cohort+'_'+target+'.npz'),**rec)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',choices=MODELS)
    p.add_argument('--cohort',choices=['short','long'],default='short');p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24);a=p.parse_args();torch.set_num_threads(1)
    run(a.source,a.cohort,a.cache if a.cohort=='long' else a.cache/'short',a.out,a.batch)
