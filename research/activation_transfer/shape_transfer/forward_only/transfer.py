"""Fit eight label-free recipient writes, mix with donor forward probes, apply in real forecasts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.forward_only.probe import HERE
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape


def shape_weight(cal_random):
    level = cal_random.mean(axis=1,keepdims=True)
    return np.sqrt(np.mean(level**2)/np.mean((cal_random-level)**2))


def fit_library(jac, basis, weight):
    """Eight unrestricted ridge columns, embedded into the chosen target span."""
    n, horizon, rank = jac.shape
    identity = np.broadcast_to(np.eye(horizon),(n,horizon,horizon))
    design = boost_shape(jac,weight).reshape(n*horizon,rank)
    desired = boost_shape(identity,weight).reshape(n*horizon,horizon)
    gram = design.T@design
    spectral = np.linalg.svd(design,compute_uv=False)
    ridge = .001*spectral[0]**2
    coeff = np.linalg.solve(gram + ridge*np.eye(rank), design.T@desired)
    library = basis.T @ coeff
    response = jac @ coeff
    mean_response = response.mean(axis=0)
    return library, dict(singular_values_weighted_design=spectral.tolist(),ridge=float(ridge),
        unweighted_mean_output_operator=mean_response.tolist(),
        singular_values_mean_output_operator=np.linalg.svd(mean_response,compute_uv=False).tolist(),
        unweighted_train_unit_impulse_mse=float(np.mean((response-identity)**2)),
        trained_impulse_norms=np.linalg.norm(library,axis=0).tolist())


def run(source,cache,out,batch):
    target = next(n for n in MODELS if n!=source)
    with np.load(out/(source+'_probes.npz')) as f:
        names, labels, atoms, codes = f['ids'],f['split'],f['atoms'],f['codes']
        observed = f['test_decoder_der'].transpose(0,2,1)
        mean = f['cal_decoder_der'].mean(0).T
        weighting = shape_weight(f['random_jac'].astype(float))
        cal_ids = f['cal_ids']
    with np.load(out/(target+'_probes.npz')) as f:
        np.testing.assert_array_equal(names,f['ids'])
        np.testing.assert_array_equal(cal_ids,f['cal_ids'])
        radius = float(f['radius'])
        spans = {key:(f[key+'_jac'].astype(float),f[key+'_basis'].astype(float)) for key in ('pc','random')}
    with np.load(cache/'data.npz') as f:
        keep = np.isin(f['split'],['daily','machines'])
        ids, x = f['ids'],f['x'][keep]
        np.testing.assert_array_equal(names,ids[keep])
    tokens,pos,all_ids = read_materials(cache,target)
    np.testing.assert_array_equal(ids,all_ids)
    tokens = tokens[keep]
    adapter = adapter_for(target)
    layer = MODELS[target][-1]
    selector = last_window_slice(adapter,pos)
    base = change_forecasts(adapter,layer,x,tokens,None,selector,batch)
    state_args = (adapter,layer,selector,x,tokens,base,codes)
    results = dict(ids=names,split=labels,atoms=atoms,codes=codes,active=codes>0,base=base,
                   shape_weight=weighting,radius=radius,source_linear=(-codes[:,:,None]*observed.transpose(0,2,1)).astype('f4'))
    details = {'source':source,'target':target,'weight':float(weighting),'library':{}}
    for span,(jac,basis) in spans.items():
        library,stats = fit_library(jac,basis,weighting)
        results[span+'_library'] = library
        details['library'][span] = stats
        for name, request in [('current',observed),('mean',mean)]:
            start = time.perf_counter()
            arm = span+'_'+name
            effects,clips = apply(*state_args,library@request,radius,batch)
            results[arm],results[arm+'_clipped'] = effects,clips
            stats[name+'_seconds'] = time.perf_counter()-start
            print(source,'to',target,arm,clips.sum(),flush=True)
    np.savez_compressed(out/(source+'_effects.npz'),**results)
    (out/(source+'_fit.json')).write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=MODELS)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--batch',type=int,default=16)
    a=p.parse_args()
    torch.set_num_threads(1)
    run(a.source,a.cache,a.out,a.batch)
