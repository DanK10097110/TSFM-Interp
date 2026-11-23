"""Test minimum-norm level cancellation in recipient hidden space by live forecast intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite
from research.activation_transfer.forecast_requests.run import (
    ALIGN, CAL, CUES, DATA_ROOT, FIT, N_ALIGN, TEST, adapter_for, data, desired, learn_directions, live_effect,
)
from research.activation_transfer.steering.study import patch_effect

HERE=Path(__file__).resolve().parent
ORIGINAL=Path('/tmp/tsfm_forecast_requests_20261120')
JACOBIANS=Path('/tmp/tsfm_shape_response_20261120')
EPS=1e-20


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def cos(a,b):
    return np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),EPS)


def unit(x):
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(norms<=1e-12):raise ValueError('Unusable hidden direction')
    return x/norms


def remove_dc(mapped,mean_output,g,basis):
    """Cancel the full mapped direction's mean shift with the minimum-norm PC-space adjustment."""
    length=np.linalg.norm(mapped,axis=-1)
    g2=g@g
    if g2<=1e-18:raise ValueError('FIT mean level gradient is too small')
    coefficient=-mean_output/g2
    correction=coefficient[:,None]*(g@basis)[None,:]
    before_rescale=mapped+correction
    corrected=unit(before_rescale)*length[:,None]
    residual=(mean_output+coefficient*g2)*length/np.linalg.norm(before_rescale,axis=-1)
    return corrected,correction,residual


def random_lifts(g,basis,norm,count=8):
    rng=np.random.default_rng(2026112031)
    coeff=unit(rng.normal(size=(count,len(g))))
    signs=np.where(coeff@g>=0.,1.,-1.)
    return (coeff*signs[:,None])@basis*norm,signs


def spectrum(jac):
    n,h,k=jac.shape
    e=np.ones(h)/np.sqrt(h)
    level=np.einsum('h,nhk->nk',e,jac)
    shape=jac-jac.mean(axis=1,keepdims=True)
    total=np.square(jac).sum()
    gram=np.einsum('nhk,nlk->hl',jac,jac)/n
    evals,axes=np.linalg.eigh(gram)
    avg=jac.mean(0);u,s,_=np.linalg.svd(avg,full_matrices=False)
    shape_s=np.linalg.svd(shape.mean(0),compute_uv=False)
    return {'fit_histories':n,'leads':h,'basis_rank':k,'dc_fraction_all':float(np.square(level).sum()/total),
            'dc_fraction_mean':float(np.square(e@avg).sum()/np.square(avg).sum()),
            'shared_dc_gradient_fraction':float(np.square(level.mean(0)).sum()/np.square(level).sum(1).mean()),
            'shared_shape_response_fraction':float(np.square(shape.mean(0)).sum()/(np.square(shape).sum()/n)),
            'rms_sensitivity_dc_as_output_norm':float(np.sqrt(np.square(level).sum()/n)),
            'rms_sensitivity_shape_as_output_norm':float(np.sqrt(np.square(shape).sum()/n)),
            'rms_sensitivity_per_orthogonal_shape_axis':float(np.sqrt(np.square(shape).sum()/(n*(h-1)))),
            'mean_jac_singular_values':s.tolist(),'mean_centered_jac_singular_values':shape_s.tolist(),
            'pooled_output_rms_singular_values':np.sqrt(np.maximum(evals[::-1],0.)).tolist(),
            'mean_top_left_vector_abs_cosine_dc':float(abs(e@u[:,0])),
            'pooled_top_output_axis_abs_cosine_dc':float(abs(e@axes[:,-1]))}


def describe(y,native):
    z=y-y.mean(axis=-1,keepdims=True);target=native-native.mean(axis=-1,keepdims=True)
    return {'fit_mean_shift':float(y.mean()),'mean_item_dc_rms':float(np.sqrt(np.mean(y.mean(-1)**2))),
            'centered_rms':float(np.sqrt(np.mean(z*z))),
            'centered_cosine_to_native':float(cos(z,target).mean()),
            'mean_response_centered_cosine_to_native':float(cos(z.mean(0),target.mean(0)))}


def study(target,original,jcache,out):
    source='timesfm' if target=='bolt' else 'bolt'
    ids,x=data(DATA_ROOT);fit_ids=ids[FIT]
    states={};native={};metadata={}
    for m in [target,source]:
        path=original/(m+'_fit.npz')
        with np.load(path) as f:
            np.testing.assert_array_equal(f['ids'],ids)
            states[m]=f['s0'];native[m]=f['d0'];metadata[m]=json.loads(str(f['metadata']))
    site=InterventionSite(**metadata[target]['site'][0]);norms=np.linalg.norm(states[target],axis=1)
    jacfile=jcache/(target+'_jac.npz')
    with np.load(jacfile) as f:
        np.testing.assert_array_equal(f['fit_ids'],fit_ids);np.testing.assert_array_equal(f['ids'],ids)
        np.testing.assert_allclose(f['original_native'],native[target],rtol=1e-8,atol=1e-8)
        assert str(f['fit_sha256'])==digest(original/(target+'_fit.npz'))
        assert json.loads(str(f['site_json']))==metadata[target]['site'][0]
        basis=f['basis'].astype(float);jac=f['jac'].astype(float);step=float(f['step']);radius=float(f['radius'])
    shape=native[target].shape[1];assert basis.shape==(24,shape) and jac.shape==(64,8,24)
    np.testing.assert_allclose(basis@basis.T,np.eye(24),atol=1e-7)
    assert abs(step-.01*radius)<1e-7
    model=SteeringModel.from_adapter(adapter_for(target),quantiles=[.5]);batch=32
    # Re-probe two PC axes on three disjoint FIT rows to check the cached units/axis order, separately from TEST.
    fit_rows=np.arange(FIT.start,FIT.stop)
    qi=np.array([1,22,60]);pc=np.array([2,17]);rows=fit_rows[qi]
    replay=patch_effect(model,site,x[rows],norms[rows],basis[pc]*step,batch).transpose(0,2,1)/step
    reprobe=float(np.max(np.abs(replay-jac[qi][:,:,pc])))
    np.testing.assert_allclose(replay,jac[qi][:,:,pc],atol=6e-4,rtol=.004)
    rng=np.random.default_rng(202611207);perms=[rng.permutation(N_ALIGN) for _ in range(4)]
    mapped,_=learn_directions(states[source][ALIGN],states[target][ALIGN],native[source],perms)
    lengths=np.linalg.norm(native[target],axis=1);mapped=unit(mapped)*lengths[:,None]
    # Measure even the mapped component outside the recipient PC subspace directly on FIT.
    response=patch_effect(model,site,x[FIT],norms[FIT],unit(mapped)*step,batch)/step
    response*=lengths[None,:,None]
    ncoef=native[target]@basis.T
    native_lin=np.einsum('nhk,ck->nch',jac,ncoef)
    proj=mapped@basis.T
    projected_lin=np.einsum('nhk,ck->nch',jac,proj)
    g=jac.mean(axis=(0,1))
    measured=response.mean(axis=(0,2))
    corrected,adjustment,resid=remove_dc(mapped[1:],measured[1:],g,basis)
    corrective=np.einsum('nhk,ck->nch',jac,adjustment@basis.T)
    factor=lengths[1:]/np.linalg.norm(mapped[1:]+adjustment,axis=1)
    corrected_lin=(response[:,1:]+corrective)*factor[None,:,None]
    np.testing.assert_allclose(corrected_lin.mean(axis=(0,2)),resid,atol=1e-8)
    lifts,signs=random_lifts(g,basis,lengths[0]); directions=[];labels=[];cue_index=[]
    for ci,cue in enumerate(CUES):
        for arm,d in [('native',native[target][ci]),('mapped',mapped[ci])]:
            labels.append(cue+'/'+arm);directions.append(d);cue_index.append(ci)
        if ci:
            labels.append(cue+'/corrected');directions.append(corrected[ci-1]);cue_index.append(ci)
    for i,d in enumerate(lifts):
        labels.append('level/random'+str(i));directions.append(d);cue_index.append(0)
    directions=np.stack(directions);cue_index=np.array(cue_index);requests=desired()[cue_index]
    np.testing.assert_allclose(np.linalg.norm(directions,axis=1),lengths[cue_index],rtol=1e-8)
    ecal=live_effect(model,site,x[CAL],norms[CAL],directions,batch)
    scalar=np.clip(np.einsum('nah,ah->a',ecal,requests)/np.maximum(np.sum(ecal**2,axis=(0,2)),1e-15),0.,1.)
    # Recheck the frozen native write against the original forecast archive on evaluation histories.
    e=live_effect(model,site,x[TEST],norms[TEST],directions,batch)
    e_scaled=live_effect(model,site,x[TEST],norms[TEST],directions*scalar[:,None],batch)
    with np.load(Path(__file__).resolve().parents[2]/'forecast_requests/results'/(target+'.npz')) as archive:
        np.testing.assert_array_equal(archive['ids'],ids[TEST]);expected=archive['0_native']
    idx=[labels.index(c+'/native') for c in CUES];native_error=float(np.max(np.abs(expected-e[:,idx])))
    np.testing.assert_allclose(e[:,idx],expected,atol=3e-4,rtol=.003)
    diagnostics={}
    for i,c in enumerate(CUES):
        diagnostics[c]={'original_mapped':describe(response[:,i],native_lin[:,i]),
                        'pc_projected_mapped':describe(projected_lin[:,i],native_lin[:,i]),
                        'native':describe(native_lin[:,i],native_lin[:,i]),
                        'mapped_norm_fraction_in_recipient_pc':float(np.linalg.norm(proj[i])/lengths[i])}
        if i:
            diagnostics[c].update(corrected=describe(corrected_lin[:,i-1],native_lin[:,i]),
                norm_of_min_correction_over_budget=float(np.linalg.norm(adjustment[i-1])/lengths[i]),
                norm_restoring_gain=float(factor[i-1]),corrected_mapped_hidden_cosine=float(cos(mapped[i],corrected[i-1])))
    test_ids=ids[TEST];out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(target+'_effects.npz'),ids=test_ids,fit_ids=ids[FIT],cal_ids=ids[CAL],align_ids=ids[ALIGN],
        labels=np.array(labels),cue=cue_index,desired=requests,write_norm=np.linalg.norm(directions,axis=1),
        cal_scalar=scalar,cal_effect=ecal,equal_budget=e,calibrated=e_scaled,directions=directions,
        mapped_fit_direct=response,corrected_fit_linear=corrected_lin,native_fit_linear=native_lin)
    summary={'source':source,'target':target,'site':metadata[target]['site'][0],'target_checkpoint':metadata[target]['checkpoint'],
             'fit_sha256':digest(original/(target+'_fit.npz')),'jac_sha256':digest(jacfile),'source_fit_sha256':digest(original/(source+'_fit.npz')),
             'fit_reprobe_abs_max':reprobe,'native_effect_archive_max_abs_error':native_error,'radius':radius,'step':step,
             'geometry':spectrum(jac),'directions':diagnostics,'random_orientation_signs':signs.tolist(),
             'labels':labels,'cal_scalar':scalar.tolist()}
    (out/(target+'_geometry.json')).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('target',choices=['bolt','timesfm']);p.add_argument('--original',type=Path,default=ORIGINAL)
    p.add_argument('--jacobians',type=Path,default=JACOBIANS);p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads);study(a.target,a.original,a.jacobians,a.out)
