"""Physically compare useful components of zero-ID-overlap and orthogonalized recipient spans."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.span_inverse import constrained
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT,adapter_for_length
from research.activation_transfer.shape_transfer.support_confirmation.run import limit_norm
from research.activation_transfer.shape_transfer.support_confirmation.supports import basis,fixed_directions
from tsfm_lens.sae.train import load_sae_checkpoint


PAIRS={'zero_off_native':('own_0','native'),'native_off_zero':('native','own_0'),
       'random_off_native':('isotropic','native'),'zero_off_random':('own_0','isotropic'),
       'pca_off_native':('pca','native')}
COMPONENTS=('zero_shared','zero_perpendicular')
BASES=('native','own_0','isotropic','pca')
TOL=1e-7


def orthogonal_part(source,remove):
    gram=source@remove.swapaxes(-2,-1)
    residual=source-gram@remove
    principal=np.linalg.svd(gram,compute_uv=False)
    _,strength,axes=np.linalg.svd(residual,full_matrices=False)
    rank=(strength>TOL).sum(axis=-1)
    axes=np.where((strength>TOL)[...,None],axes,0.)
    # Nonzero axes span the residual span after orthonormalizing; a zero row has no coefficient.
    leakage=axes@remove.swapaxes(-2,-1)
    assert np.max(abs(leakage))<1e-7
    return axes,strength,rank,principal


def inner(x,y):
    return np.einsum('nd,nd->n',x,y)


def cosine(x,y):
    return inner(x,y)/np.maximum(np.sqrt(inner(x,x)*inner(y,y)),1e-20)


def run(cache,out,batch):
    ref=dict(np.load(out/'short_bolt.npz'));ids=ref['ids'];target='bolt';rows=ref['rows']
    with np.load(cache/'data.npz') as d:x=d['x'];np.testing.assert_array_equal(ids,d['ids'])
    h,pos,hids=read_materials(cache,target);np.testing.assert_array_equal(hids,ids)
    with np.load(cache/'bolt_early.npz') as f:j=f['jac'].astype(float)
    sae=load_sae_checkpoint(str(dictionary_path(target)));dec=sae.W_dec.detach().numpy()
    _,_,static,_=fixed_directions(target,sae)
    axes={'native':basis(dec[ref['native_ids']]),'own_0':basis(dec[ref['own_0_ids']])}
    axes.update({k:np.broadcast_to(v,(len(ids),*v.shape)) for k,v in static.items()})
    dense=ref['dense_vector'].astype(float);length=ref['dense_length'];weight=float(ref['weight'])
    rec=dict(ids=ids,rows=rows,cols=ref['cols'],atoms=ref['atoms'],tol=TOL)
    for arm,(raw,remove) in PAIRS.items():
        axes[arm],strength,ranks,principal=orthogonal_part(axes[raw],axes[remove])
        rec[arm+'_singular']=strength;rec[arm+'_rank']=ranks;rec[arm+'_principal_cosines']=principal
    enc=adapter_for_length(target,x.shape[1]);site=MODELS[target][-1]
    select=last_window_slice(enc,pos);writes={}
    for arm,q in axes.items():
        pieces=[]
        for i in range(0,len(rows),batch):
            sub=slice(i,i+batch);r=rows[sub];selected=q[r];nonzero=np.any(selected!=0,axis=(1,2))
            value=np.zeros_like(dense[sub])
            if nonzero.any():
                value[nonzero]=constrained(j[r[nonzero]],selected[nonzero],dense[sub][nonzero],weight)
            capped,_,_=limit_norm(value,length[sub]);pieces.append(capped.astype(float))
        writes[arm]=np.concatenate(pieces)
    fv=writes['own_0'];native=axes['native'][rows]
    shared=np.einsum('nkd,nd->nk',native,fv)
    writes['zero_shared']=np.einsum('nk,nkd->nd',shared,native)
    writes['zero_perpendicular']=fv-writes['zero_shared']
    np.testing.assert_allclose(inner(writes['zero_shared'],writes['zero_perpendicular']),0,atol=3e-10)
    assert np.max(abs(cosine(writes['native'],writes['zero_off_native'])))<2e-6
    for arm,v in writes.items():
        rec[arm+'_write']=v.astype('f4')
        jv=np.einsum('nhd,nd->nh',j[rows],v)
        rec[arm+'_linear']=jv.astype('f4');rec[arm+'_cosine_native']=cosine(v,writes['native'])
        rec[arm+'_cosine_zero']=cosine(v,writes['own_0'])
        rec[arm+'_length_fraction']=np.linalg.norm(v,axis=1)/length
        if arm in BASES:
            # Independently refit and replay all four original physical arms before interpreting projected alternatives.
            e=forecast16(enc,site,x[rows],h[rows],v.astype('f4'),select,batch)-ref['ordinary_target']
            np.testing.assert_allclose(e,ref[arm],atol=4e-5,rtol=.0005)
            rec[arm+'_replay_max']=np.max(abs(e-ref[arm]))
            rec[arm]=e
        else:
            rec[arm]=forecast16(enc,site,x[rows],h[rows],v.astype('f4'),select,batch)-ref['ordinary_target']
        print(arm,flush=True)
    np.savez_compressed(out/'bolt_residual.npz',**rec)


def score(out):
    ref=dict(np.load(out/'short_bolt.npz'));r=dict(np.load(out/'bolt_residual.npz'))
    np.testing.assert_array_equal(ref['rows'],r['rows']);np.testing.assert_array_equal(ref['cols'],r['cols'])
    mask=ref['codes']>0;keep=mask.sum(0)>=8;live=keep[r['cols']];n=len(mask)
    arms=(*BASES,*PAIRS,*COMPONENTS)
    draws=resampling_counts(n,draws=3000)
    nf=int(keep.sum());fdraws=np.random.default_rng(221106).multinomial(nf,np.full(nf,1/nf),3000)
    def summary(v):
        return dict(mean=float(np.mean(v)),quantiles=np.quantile(v,[0,.1,.5,.9,1]).tolist())
    report=dict(subspaces={},waveforms={},endpoints={},geometry={})
    for k,(a,b) in PAIRS.items():
        principal=np.clip(r[k+'_principal_cosines'],0.,1.);angles=np.rad2deg(np.arccos(principal))
        strength=r[k+'_singular'];rank=r[k+'_rank']
        condition=strength[:,0]/np.take_along_axis(strength,np.maximum(rank-1,0)[:,None],axis=1)[:,0]
        report['subspaces'][k]=dict(origin=a,removed=b,ranks=summary(rank),
            projector_overlap=summary(np.mean(principal**2,axis=1)),largest_cosine=summary(principal[:,0]),
            first_angle=summary(angles[:,0]),middle_angle=summary(np.median(angles,axis=1)),
            all_angles=summary(angles.ravel()),singular_min=summary(strength[:,-1]),positive_condition=summary(condition),
            mean_angles_under={str(deg):float(np.mean(np.sum(angles<deg,axis=1))) for deg in [15,30,45]})
    for k in arms:
        report['geometry'][k]={name:summary(r[k+'_'+name][live]) for name in ['cosine_native','cosine_zero','length_fraction']}
        y=r[k][live,:8].astype(float);native=r['native'][live,:8].astype(float)
        truth=ref['dense'][live,:8].astype(float)
        for v in (y,native,truth):v-=v.mean(1,keepdims=True)
        report['waveforms'][k]=dict(event_centered_cosine_native=summary(cosine(y,native)),
                event_centered_cosine_dense=summary(cosine(y,truth)),
                centered_rms_ratio_vs_native=float(np.linalg.norm(y)/np.linalg.norm(native)),
                centered_rms_ratio_vs_dense=float(np.linalg.norm(y)/np.linalg.norm(truth)),
                pooled_signed_gain_vs_dense=float(np.sum(y*truth)/np.sum(truth**2)),
                rms_difference_vs_native=float(np.sqrt(np.mean((y-native)**2))))
    orig=r['own_0_write'].astype(float);shared=r['zero_shared_write'].astype(float)
    report['shared_zero_write_energy']=summary((inner(shared,shared)/inner(orig,orig))[live])
    nonlinear=r['zero_perpendicular']+r['zero_shared']-r['own_0']
    report['component_nonadditivity']=dict(early_rms=float(np.sqrt(np.mean(nonlinear[live,:8]**2))),
                                           zero_early_rms=float(np.sqrt(np.mean(r['own_0'][live,:8]**2))))
    for seg,sl in [('early',slice(0,8)),('late',slice(8,16))]:
        def wave(y):
            new=np.zeros((n,len(keep),8));v=y[:,sl].astype(float);v-=v.mean(1,keepdims=True)
            new[r['rows'],r['cols']]=v
            return new[:,keep]
        for ref_name in ('source','dense'):
            truth=wave(ref[ref_name]);fits={}
            for k in arms:fits[k]=intervals(mask[:,keep],wave(r[k]),truth,draws,fdraws)[:3]
            metrics={k:entry(value) for k,value in fits.items()}
            contrasts=[(k,pair[0]) for k,pair in PAIRS.items()]
            contrasts.extend((k,'own_0') for k in COMPONENTS)
            contrasts.extend([('zero_off_native','native'),('zero_off_native','random_off_native')])
            for a,b in contrasts:metrics[a+'_minus_'+b]=entry(tuple(x-y for x,y in zip(fits[a],fits[b])))
            report['endpoints'][seg+'_'+ref_name]=metrics
    (out/'residual.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['run','score'])
    p.add_argument('--cache',type=Path,default=CACHE/'short');p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--batch',type=int,default=24);a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='run':run(a.cache,a.out,a.batch)
    else:score(a.out)
