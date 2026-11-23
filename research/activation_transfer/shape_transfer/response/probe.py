"""Create a shareable exact-site local forecast Jacobian for the original 64 fit histories."""
import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch
from research.activation_transfer.forecast_requests.run import (
    ALIGN, CAL, FIT, TEST, RANK, DATA_ROOT, adapter_for, data, patch_effect, sites_and_states,
)
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite

HERE=Path(__file__).resolve().parent
ORIGINAL=Path('/tmp/tsfm_forecast_requests_20261120')
CACHE=Path('/tmp/tsfm_shape_response_20261120')


def collect(name,cache,original,batch):
    cache.mkdir(parents=True,exist_ok=True)
    ids,x=data(DATA_ROOT); fit_file=original/(name+'_fit.npz')
    with fit_file.open('rb') as reader:
        fit_sha=file_digest(reader,'sha256').hexdigest()
    with np.load(fit_file) as f:
        np.testing.assert_array_equal(ids,f['ids']); states=f['s0']; native=f['d0']; meta=json.loads(str(f['metadata']))
    if len(set(ids))!=len(ids) or len(x[FIT])!=64:
        raise ValueError('Unexpected or overlapping probe cohorts')
    h=states[ALIGN].astype(float)
    _,_,v=np.linalg.svd(h-h.mean(0),full_matrices=False); basis=v[:RANK]
    np.testing.assert_allclose(basis@basis.T,np.eye(RANK),atol=1e-12)
    radius=float(np.median(np.linalg.norm(h,axis=1))); step=.01*radius
    np.testing.assert_allclose(step,meta['fit'][0]['step'])
    model=SteeringModel.from_adapter(adapter_for(name),quantiles=[.5]); site=InterventionSite(**meta['site'][0])
    sites,recorded=sites_and_states(model,[site.module],x[FIT][:3],3)
    if asdict(sites[site.module])!=asdict(site): raise ValueError('Wrong intervention site')
    np.testing.assert_allclose(recorded[site.module],states[FIT][:3],atol=4e-5,rtol=1e-4)
    norms=np.linalg.norm(states[FIT],axis=1)
    effects=patch_effect(model,site,x[FIT],norms,basis*step,batch)
    repeat=patch_effect(model,site,x[FIT][:1],norms[:1],basis[:2]*step,1)
    np.testing.assert_allclose(effects[:1,:2],repeat,atol=4e-5,rtol=2e-3)
    jac=(effects/step).transpose(0,2,1).astype(float)
    if jac.shape!=(64,8,24) or not np.isfinite(jac).all() or np.min(np.linalg.norm(jac,axis=(0,1)))<=0:
        raise ValueError('Inactive or invalid local Jacobian')
    destination=cache/(name+'_jac.npz'); partial=cache/(name+'_pending.npz')
    np.savez_compressed(partial,basis=basis,jac=jac,fit_ids=ids[FIT],ids=ids,radius=radius,step=step,
                        original_native=native,site_json=json.dumps(asdict(site)),checkpoint_json=json.dumps(meta['checkpoint']),
                        fit_sha256=fit_sha,repeat_max_abs=float(np.max(abs(effects[:1,:2]-repeat))))
    partial.replace(destination)
    print(destination,jac.shape,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('model',choices=['timesfm','bolt']);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--original',type=Path,default=ORIGINAL);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(1);collect(a.model,a.cache,a.original,a.batch_size)
