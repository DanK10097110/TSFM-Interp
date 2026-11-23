"""Run inferred native doses as real H16 deletions in the receiving forecaster."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.moment.crosscoder.architecture_check.run import OUT
from research.activation_transfer.moment.crosscoder.architecture_check.score import divide, macro_skill, result
from research.activation_transfer.shape_transfer.cross_seed_groups.study import population
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from tsfm_lens.sae.train import load_sae_checkpoint

ARMS=('state_exact','direct','explicit_gate')


def run(target):
    source='timesfm' if target=='bolt' else 'bolt'
    stem=f'{source}_to_{target}'
    with np.load(OUT/(stem+'.npz')) as f:
        data=dict(f)
    sel=data['domain']=='noaa'
    x,h,pos,ids,station,base,atoms,z,r,c,measured,_=population(target,'noaa')
    np.testing.assert_array_equal(data['nominated'],atoms)
    np.testing.assert_array_equal(ids,data['ids'][sel])
    np.testing.assert_allclose(z,data['truth'][sel,:len(atoms)],atol=3e-5,rtol=1e-5)
    dict_model=load_sae_checkpoint(str(dictionary_path(target)))
    dirs=dict_model.W_dec.detach().numpy()[atoms]
    model=adapter_for(target)
    site=MODELS[target][-1]
    window=last_window_slice(model,pos)
    batch=24
    ix=np.linspace(0,len(r)-1,12,dtype=int)
    rows,cols=r[ix],c[ix]
    zero=forecast16(model,site,x[rows],h[rows],np.zeros_like(dirs[cols]),window,batch)
    ordinary=forecast16(model,site,x[rows],h[rows],None,window,batch)
    np.testing.assert_allclose(ordinary,zero,atol=.00015,rtol=2e-5)
    true=forecast16(model,site,x[rows],h[rows],-z[rows,cols,None]*dirs[cols],window,batch)-zero
    np.testing.assert_allclose(true,measured[ix],atol=.00028,rtol=.003)
    native=np.zeros((len(x),len(atoms),16),'f4');native[r,c]=measured
    archive=dict(ids=ids,atoms=atoms,station=station,sd=x.astype(float).std(1),truth=native,
                 archived_native_max_abs_c=float(np.max(np.abs(true-measured[ix]))),gates=z>0)
    for arm in ARMS:
        inferred=data[arm][sel,:len(atoms)]
        rr,cc=np.nonzero(inferred>0)
        delta=-inferred[rr,cc,None]*dirs[cc]
        estimate=np.zeros_like(native)
        estimate[rr,cc]=forecast16(model,site,x[rr],h[rr],delta,window,batch)-base[rr]
        rows=np.unique(np.r_[np.linspace(0,len(rr)-1,5,dtype=int), np.flatnonzero(z[rr,cc]==0)[:1]])
        a=rr[rows]; dd=delta[rows]
        # Independent matched zero-write difference at a fresh batch grouping.
        native_check=forecast16(model,site,x[a],h[a],dd,window,batch)-forecast16(model,site,x[a],h[a],np.zeros_like(dd),window,batch)
        np.testing.assert_allclose(native_check,estimate[a,cc[rows]],atol=.0003,rtol=.003)
        archive[arm]=estimate
        archive[arm+'_replay_max_abs_c']=float(np.max(np.abs(native_check-estimate[a,cc[rows]])))
        print(stem,arm,len(rr),flush=True)
    np.savez_compressed(OUT/(stem+'_weather_waves.npz'),**archive)


def score(target):
    source='timesfm' if target=='bolt' else 'bolt'
    stem=f'{source}_to_{target}'
    with np.load(OUT/(stem+'_weather_waves.npz')) as f:
        z=dict(f)
    with np.load(OUT/(stem+'.npz')) as f:
        four=f['four_nominated']
    _,rows=np.unique(z['station'],return_inverse=True)
    n=len(np.unique(rows))
    weight=np.vstack((np.ones(n),np.random.default_rng(22112431).multinomial(n,np.ones(n)/n,3000)))
    def aggregate(v):
        vals=np.zeros((n,v.shape[1]));np.add.at(vals,rows,v)
        return weight@vals
    def component(v):
        a=v.astype(float)
        return (a-a.mean(-1,keepdims=True))/z['sd'][:,None,None]
    report={}
    for name,indices in [('nominated',np.arange(len(z['atoms']))),('four_nominated',np.array([np.where(z['atoms']==a)[0].item() for a in four]))]:
        ref=component(z['truth'])[:,indices]
        power=aggregate(np.mean(ref**2,axis=2))
        out={};scores={}
        for arm in ARMS:
            est=component(z[arm])[:,indices]
            error=aggregate(np.mean((est-ref)**2,axis=2))
            macro=macro_skill(error,power);micro=1-divide(error.sum(1),power.sum(1))
            scores[arm]=macro
            r=z['truth'][:,indices].astype(float);e=z[arm][:,indices].astype(float)
            r-=r.mean(2,keepdims=True);e-=e.mean(2,keepdims=True)
            out[arm]=dict(macro=result(macro),pooled=result(micro),centered_error_rms_C=float(np.sqrt(np.mean((r-e)**2))),
                reference_rms_C=float(np.sqrt(np.mean(r*r))),closed_gate_rms_C=float(np.sqrt(np.mean(e[~z['gates'][:,indices]]**2))))
        report[name]=dict(available_atoms=int((power[0]>0).sum()),nominated_atoms=len(indices),arms=out,
             state_minus_direct=result(scores['state_exact']-scores['direct']),gate_minus_direct=result(scores['explicit_gate']-scores['direct']))
    (OUT/(stem+'_weather_physical.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('target',choices=('bolt','timesfm'))
    a=p.parse_args();torch.set_num_threads(2);run(a.target);score(a.target)
