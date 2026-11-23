"""Independently replay sparse probe subsets, library fitting and live single-case interventions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as PRIOR_CACHE
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH_CACHE
from research.activation_transfer.shape_transfer.forward_only.audit import add_native,point
from research.activation_transfer.shape_transfer.forward_only.probe import HERE as FORWARD
from research.activation_transfer.shape_transfer.calibration_budget.study import HERE, SEEDS, materials
from tsfm_lens.sae.train import load_sae_checkpoint


def independent_fit(jac,basis,weight):
    constant=np.ones((8,8))/8
    metric=constant+weight*(np.eye(8)-constant)
    a=np.concatenate([metric@sample for sample in jac],axis=0)
    goals=np.tile(metric,(len(jac),1))
    penalty=.001*np.linalg.svd(a,compute_uv=False)[0]**2
    k=jac.shape[-1]
    coeff=np.linalg.lstsq(np.vstack([a,np.sqrt(penalty)*np.eye(k)]),np.vstack([goals,np.zeros((k,8))]),rcond=None)[0]
    return basis.T@coeff


def audit(source,prior,fresh,out):
    target=next(s for s in MODELS if s!=source)
    lib=dict(np.load(out/(target+'_libraries.npz')))
    target_cal=dict(np.load(FORWARD/'results'/(target+'_probes.npz')))
    basis=lib['basis'];jac=target_cal['pc_jac'].astype(float);weight=lib['weight']
    max_fit=0.;pilot_agreement=[]
    for j,seed in enumerate(SEEDS):
        r=np.random.default_rng(seed).permutation(96)
        np.testing.assert_array_equal(r,lib['permutations'][j])
        for n,k in ((4,24),(12,8),(16,24),(48,8),(4,8),(16,8)):
            ref=independent_fit(jac[r[:n],:,:k],basis[:k],weight)
            max_fit=max(max_fit,float(np.max(abs(ref-lib[f'{n}x{k}_{j}']))))
        pa=lib[f'pilot_axes_{j}'];measured=lib[f'pilot_measured_{j}'].transpose(0,2,1).astype(float)
        ref=independent_fit(measured,pa,weight)
        max_fit=max(max_fit,float(np.max(abs(ref-lib[f'pilot_{j}']))))
        # An independent finite-probe basis at the same CAL histories validates the real pilot measurements.
        projected=jac[r[4:40]]@(basis@pa.T)
        pilot_agreement.append(float(np.linalg.norm(projected-measured)/np.linalg.norm(measured)))
    assert max_fit<1e-7 and max(pilot_agreement)<.05
    single=dict(np.load(out/(source+'_single.npz')))
    single_fit_error=0.
    for n,k,repeats in [(1,24,8),(2,24,3),(3,8,3)]:
        arm=f'{n}x{k}'
        for j in range(repeats):
            sampled=np.random.default_rng(SEEDS[0]+j).permutation(96)[:n]
            np.testing.assert_array_equal(sampled,single[arm+'_rows'][j])
            check=independent_fit(jac[sampled,:,:k],basis[:k],weight)
            single_fit_error=max(single_fit_error,float(abs(check-single[arm+'_library'][j]).max()))
    assert single_fit_error<1e-7
    report=dict(fit_abs_max=max_fit,single_history_fit_abs_max=single_fit_error,
                pilot_central_difference_relative_error=pilot_agreement,items=[])
    model=adapter_for(target);layer=MODELS[target][-1]
    for cohort,cache in [('original',prior),('fresh',fresh)]:
        probe_file=(FORWARD/'results'/(source+'_probes.npz') if cohort=='original' else out/(source+'_fresh_probes.npz'))
        with np.load(probe_file) as p,np.load(out/(source+'_'+cohort+'.npz')) as a:
            x,_,pos,ids,_=materials(target,cache)
            lookup={v:i for i,v in enumerate(ids)};select=last_window_slice(model,pos)
            for domain in np.unique(p['split']):
                rows=np.flatnonzero((p['split']==domain)&(p['codes']>0).any(axis=1))[:2]
                for row in rows:
                    feat=np.flatnonzero(p['codes'][row]>0)[0];item=p['ids'][row];obs=x[lookup[item]]
                    clean=point(model,obs);der=p['test_decoder_der'][row,feat];code=p['codes'][row,feat]
                    result=dict(cohort=cohort,domain=str(domain),id=str(item),atom=int(p['atoms'][feat]),arms={})
                    for arm,index in [('baseline',0),('4x24',0),('12x8',1),('16x24',2),('48x8',0),('pilot',1),('wrong',2)]:
                        name=('baseline' if arm=='baseline' else f"{'16x24' if arm=='wrong' else arm}_{index}")
                        values=der[[3,6,1,7,0,2,5,4]] if arm=='wrong' else der
                        write=-code*(lib[name]@values)
                        norm=np.linalg.norm(write)
                        scale=min(1.,float(.2*lib['radius'])/max(norm,1e-20))
                        with add_native(model,layer,select,scale*write):
                            returned=point(model,obs)-clean
                        expect=(a[arm][row,feat] if arm=='baseline' else a[arm][index,row,feat])
                        np.testing.assert_allclose(returned,expect,atol=8e-5,rtol=.002)
                        result['arms'][arm]=float(abs(returned-expect).max())
                    if cohort=='fresh':
                        for arm,index in [('1x24',7),('2x24',1),('3x8',2)]:
                            write=-code*(single[arm+'_library'][index]@der)
                            scale=min(1.,float(.2*lib['radius'])/max(np.linalg.norm(write),1e-20))
                            with add_native(model,layer,select,scale*write):returned=point(model,obs)-clean
                            expect=single[arm][index,row,feat]
                            np.testing.assert_allclose(returned,expect,atol=8e-5,rtol=.002)
                            result['arms'][arm]=float(abs(returned-expect).max())
                    report['items'].append(result)
    del model
    model=adapter_for(source);layer=MODELS[source][-1]
    x,_,pos,ids,_=materials(source,fresh);select=last_window_slice(model,pos)
    with np.load(out/(source+'_fresh_probes.npz')) as p:
        np.testing.assert_array_equal(ids,p['ids']);decoder=load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[p['atoms']]
        errors=[]
        for domain in np.unique(p['split']):
            for i in np.flatnonzero((p['split']==domain)&(p['codes']>0).any(axis=1))[:3]:
                j=np.flatnonzero(p['codes'][i]>0)[0]
                vec=decoder[j].astype(float)*p['steps'][i,j]/np.linalg.norm(decoder[j].astype(float))
                pair=[]
                for sign in (1.,-1.):
                    with add_native(model,layer,select,sign*vec):pair.append(point(model,x[i]))
                half=(pair[0]-pair[1])/2
                np.testing.assert_allclose(half,p['half_difference'][i,j],atol=5e-5,rtol=.001)
                errors.append(float(abs(half-p['half_difference'][i,j]).max()))
        report['fresh_source_tiny_half_errors']=errors
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--prior',type=Path,default=PRIOR_CACHE);p.add_argument('--fresh',type=Path,default=FRESH_CACHE)
    a=p.parse_args();torch.set_num_threads(1)
    results={m:audit(m,a.prior,a.fresh,a.out) for m in MODELS}
    (a.out/'replay.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
