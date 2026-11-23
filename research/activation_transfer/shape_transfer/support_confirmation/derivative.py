"""Cross current versus reassigned Jacobian with current versus reassigned candidate span."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.continuation.probe import SCRATCH as CAL
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape,solve_adaptive
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT,adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.run import limit_norm
from research.activation_transfer.shape_transfer.support_confirmation.score import reduce_draws
from research.activation_transfer.shape_transfer.support_confirmation.supports import SEEDS,basis
from tsfm_lens.sae.train import load_sae_checkpoint


def fit_goal(jac,axes,held_goal,budget,weight,mode):
    design=jac@axes.swapaxes(-2,-1)
    response=held_goal[...,None]
    if mode=='dc':
        design=design.mean(axis=1,keepdims=True)
        response=response.mean(axis=1,keepdims=True)
    else:
        design=boost_shape(design,weight);response=boost_shape(response,weight)
    coefs=solve_adaptive(design,response)[:,:,0]
    raw=np.einsum('nr,nrd->nd',coefs,axes)
    return limit_norm(raw,budget)[0]


def run(target,cache,out,batch):
    ref=dict(np.load(out/('short_'+target+'.npz')))
    h,pos,ids=read_materials(cache,target);rows=ref['rows'];native=ref['native_ids']
    np.testing.assert_array_equal(ids,ref['ids'])
    with np.load(cache/'data.npz') as z:x=z['x']
    with np.load(cache/(target+'_early.npz')) as z:j=z['jac'].astype(float)
    with np.load(CAL/(target+'_jac16.npz')) as z:
        cal=z['split']=='cal';avg=z['jac'][cal,:8].astype(float).mean(0);fit_ids=z['ids'][cal]
    assert cal.sum()==96 and set(fit_ids).isdisjoint(ids)
    goal=ref['receiver_dense_early_derivative'].astype(float)
    np.testing.assert_allclose(goal,np.einsum('nhd,nd->nh',j[rows],ref['dense_vector']),atol=2e-6)
    length=ref['dense_length'];weight=float(ref['weight'])
    sae=load_sae_checkpoint(str(dictionary_path(target)));q=basis(sae.W_dec.detach().numpy()[native])
    enc=adapter_for_length(target,x.shape[1]);sel=last_window_slice(enc,pos);site=MODELS[target][-1]
    record=dict(ids=ids,rows=rows,cols=ref['cols'],atoms=ref['atoms'],goal=goal,
                average_cal_ids=fit_ids,native_ids=native,weight=weight,budget=length)
    def estimate(key,which_j,which_q,mode):
        estimates=[];lengths=[];linear=[];used=[]
        for i in range(0,len(rows),batch):
            sl=slice(i,i+batch);r=rows[sl]
            actual_j=j[r]
            if which_j is None:given_j=np.broadcast_to(avg,(len(r),*avg.shape))
            else:given_j=j[which_j[r]]
            which=which_q[r];delta=fit_goal(given_j,q[which],goal[sl],length[sl],weight,mode)
            effect=forecast16(enc,site,x[r],h[r],delta,sel,batch)-ref['ordinary_target'][sl]
            estimates.append(effect);lengths.append(np.linalg.norm(delta.astype(float),axis=1))
            linear.append(np.einsum('nhd,nd->nh',actual_j,delta))
            used.append(np.einsum('nhd,nd->nh',given_j,delta))
        record[key]=np.concatenate(estimates);record[key+'_norm']=np.concatenate(lengths)
        record[key+'_pred_current']=np.concatenate(linear).astype('f4');record[key+'_pred_given']=np.concatenate(used).astype('f4')
        print(target,key,flush=True)
    ident=np.arange(len(ids))
    for mode in ('balanced','dc'):
        estimate(mode+'_NN',ident,ident,mode)
        if mode=='balanced':
            np.testing.assert_allclose(record[mode+'_NN'],ref['native'],atol=5e-5,rtol=.0005)
            estimate('cal_N',None,ident,mode)
        for seed in SEEDS:
            other=ref[f'swap_{seed}_assignment'];record[f'assigned_{seed}']=other
            # First letter indexes candidate support, second the derivative source.
            for which,ji,qi in [('SN',ident,other),('NS',other,ident),('SS',other,other)]:
                key=f'{mode}_{which}_{seed}'
                if mode=='balanced' and which=='SN':
                    record[key]=ref[f'swap_{seed}'];record[key+'_norm']=ref[f'swap_{seed}_length']
                else:estimate(key,ji,qi,mode)
            if mode=='balanced':estimate(f'cal_S_{seed}',None,other,mode)
    np.savez_compressed(out/(target+'_derivative.npz'),**record)


def adjust(a,b):return tuple(x-y for x,y in zip(a,b))


def score(out):
    result={}
    for target in ('bolt','timesfm'):
        ref=dict(np.load(out/('short_'+target+'.npz')))
        data=dict(np.load(out/(target+'_derivative.npz')))
        eligible=(ref['codes']>0).sum(0)>=8
        mask=ref['codes'][:,eligible]>0;n=len(mask);nf=mask.shape[1]
        hdraw=resampling_counts(n,draws=3000)
        fdraw=np.random.default_rng(221106).multinomial(nf,np.full(nf,1/nf),3000)
        blocks={};result[target]=blocks;live=eligible[ref['cols']]
        blocks['median_realized_norm_fractions']={k[:-5]:float(np.median(data[k][live]/data['budget'][live])) for k in data if k.endswith('_norm')}
        for segment,sl in [('early',slice(0,8)),('late',slice(8,16))]:
            for kind in ('full','shape','level'):
                def cube(y):
                    a=y[:,sl].astype(float)
                    if kind=='shape':a-=a.mean(1,keepdims=True)
                    if kind=='level':a=a.mean(1,keepdims=True)
                    v=np.zeros((n,len(eligible),a.shape[1]));v[ref['rows'],ref['cols']]=a
                    return v[:,eligible]
                for reference in ('source','dense'):
                    truth=cube(ref[reference]);summary={};blocks['_'.join((segment,kind,reference))]=summary
                    def measure(k):return intervals(mask,cube(data[k]),truth,hdraw,fdraw)[:3]
                    for mode in ('balanced','dc'):
                        vals={'NN':measure(mode+'_NN')}
                        for state in ('SN','NS','SS'):
                            vals[state]=reduce_draws([measure(f'{mode}_{state}_{s}') for s in SEEDS])
                        delta_j=reduce_draws([adjust(vals['NN'],vals['NS']),adjust(vals['SN'],vals['SS'])])
                        delta_s=reduce_draws([adjust(vals['NN'],vals['SN']),adjust(vals['NS'],vals['SS'])])
                        summary[mode]={k:entry(v) for k,v in vals.items()}
                        for k,v in [('derivative_cost',delta_j),('support_cost',delta_s),
                                    ('derivative_minus_support_cost',adjust(delta_j,delta_s)),
                                    ('native_span_derivative_cost',adjust(vals['NN'],vals['NS'])),
                                    ('current_j_support_cost',adjust(vals['NN'],vals['SN'])),
                                    ('both_wrong_minus_wrong_j_only',adjust(vals['SS'],vals['NS']))]:
                            summary[mode][k]=entry(v)
                        if mode=='balanced':
                            cal_n=measure('cal_N')
                            cal_s=reduce_draws([measure(f'cal_S_{s}') for s in SEEDS])
                            summary[mode].update(cal_N=entry(cal_n),cal_S=entry(cal_s),
                                    current_minus_cal=entry(adjust(vals['NN'],cal_n)))
    (out/'derivative.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('score',*MODELS))
    p.add_argument('--cache',type=Path,default=CACHE/'short');p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24)
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='score':score(a.out)
    else:run(a.stage,a.cache,a.out,a.batch)
