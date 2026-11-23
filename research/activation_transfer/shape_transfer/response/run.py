"""Fit a common forecast-derivative bridge and execute candidate writes in the recipient."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from research.activation_transfer.forecast_requests.run import ALIGN, FIT, TEST, DATA_ROOT, adapter_for, data, desired, learn_directions, live_effect
from research.activation_transfer.shape_transfer.response.model import goals, limit_norm, match_norm, policies, ridge_response
from research.activation_transfer.shape_transfer.response.probe import CACHE, HERE, ORIGINAL
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite

WAVES=['level','tilt','sine','early_hump_residual','period_four_residual']
ARMS=['response','clean_capped','clean_matched','shuffled_capped','shuffled_matched','mean_capped','mean_matched','native']
PERM=np.array([3,6,1,4,7,2,5,0])


def run(target,cache,original,batch):
    source='timesfm' if target=='bolt' else 'bolt'; ids,x=data(DATA_ROOT)
    probes={};states={};native={};errors={};probe_paths={};fingerprints={}
    for name in [source,target]:
        probe_paths[name]=cache/(name+'_jac.npz')
        with np.load(probe_paths[name]) as f: probes[name]=dict(f)
        p=probes[name];np.testing.assert_array_equal(ids,p['ids']);np.testing.assert_array_equal(ids[FIT],p['fit_ids'])
        fit_path=original/(name+'_fit.npz')
        with fit_path.open('rb') as reader: sha=file_digest(reader,'sha256').hexdigest()
        if sha!=str(p['fit_sha256']): raise ValueError('Original fit cache changed since probe capture')
        with np.load(fit_path) as f:
            np.testing.assert_array_equal(ids,f['ids']); states[name]=f['s0']
        with probe_paths[name].open('rb') as reader: fingerprints[name]=file_digest(reader,'sha256').hexdigest()
        native[name],errors[name]=policies(p['jac'],p['basis'],float(p['radius']),p['original_native'])
    s,t=probes[source],probes[target]; js,jt=s['jac'],t['jac'];q=native[source]@s['basis'].T
    np.testing.assert_allclose(goals()[:3],desired(),atol=1e-15)
    observed_mean=js.mean(axis=0,keepdims=True); training_template=(observed_mean[0]@q.T).T
    donor_free=np.broadcast_to(observed_mean,js.shape)
    matrices={};response={}
    for name,design in [('response',js),('shuffled',js[:,PERM]),('mean',donor_free)]:
        matrices[name],penalty=ridge_response(jt,design)
        response[name]=q@matrices[name].T@t['basis']
    clean,_=learn_directions(states[source][ALIGN],states[target][ALIGN],native[source],[np.arange(len(states[source][ALIGN]))])
    maxnorm=.2*float(t['radius']);aligned=limit_norm(response['response'],maxnorm)
    cap=lambda v:limit_norm(v,maxnorm); match=lambda v:match_norm(v,aligned)
    physical=dict(response=aligned,clean_capped=cap(clean),clean_matched=match(clean),
                  shuffled_capped=cap(response['shuffled']),shuffled_matched=match(response['shuffled']),
                  mean_capped=cap(response['mean']),mean_matched=match(response['mean']),native=native[target])
    directions=np.stack([physical[n] for n in ARMS],axis=1)
    np.testing.assert_allclose(np.linalg.norm(directions[:,[0,2,4,6]],axis=-1),np.repeat(np.linalg.norm(aligned,axis=1)[:,None],4,axis=1),rtol=1e-8)
    if np.linalg.norm(directions,axis=-1).max()>maxnorm+1e-8: raise ValueError('Exceeded common target trust radius')
    site=InterventionSite(**json.loads(str(t['site_json'])));model=SteeringModel.from_adapter(adapter_for(target),quantiles=[.5]); norms=np.linalg.norm(states[target],axis=1)
    flat=directions.reshape(-1,directions.shape[-1]);dest=HERE/'results';dest.mkdir(exist_ok=True)
    dx=live_effect(model,site,x[TEST],norms[TEST],flat,batch).reshape(len(ids[TEST]),len(WAVES),len(ARMS),8)
    single=live_effect(model,site,x[TEST][:1],norms[TEST][:1],flat,1).reshape(1,len(WAVES),len(ARMS),8)
    np.testing.assert_allclose(single,dx[:1],atol=3e-4,rtol=2e-3)
    zero=live_effect(model,site,x[TEST][:5],norms[TEST][:5],np.zeros_like(flat[:1]),batch)
    if abs(zero).max()>1e-5 or not np.isfinite(dx).all(): raise ValueError('Invalid zero-dose/replay')
    previous=HERE.parents[1]/'forecast_requests/results'/(target+'.npz')
    with np.load(previous) as f:
        np.testing.assert_array_equal(f['ids'],ids[TEST]); archive=f['0_native']
    np.testing.assert_allclose(dx[:,:3,-1],archive,atol=3e-4,rtol=2e-3)
    np.savez_compressed(dest/(target+'_effects.npz'),ids=ids[TEST],arms=ARMS,waves=WAVES,goals=goals(),effects=dx,
            donor_free_source=training_template,applied_write_directions=directions,source_policies=native[source],
            **{'map_'+k:v for k,v in matrices.items()})
    meta=dict(source=source,target=target,site=json.loads(str(t['site_json'])),source_pca_fit_disagreement=errors[source].tolist(),
              target_pca_fit_disagreement=errors[target].tolist(),J_sha256=fingerprints,penalty=penalty,lead_perm=PERM.tolist(),
              recipient_trust_radius=maxnorm,source_native_norm=np.linalg.norm(native[source],axis=1).tolist(),
              before_clip={k:np.linalg.norm(v,axis=1).tolist() for k,v in {'clean':clean,**response}.items()},
              applied_norm={k:np.linalg.norm(v,axis=1).tolist() for k,v in physical.items()},
              max_replay_error=float(abs(single-dx[:1]).max()),zero_drift=float(abs(zero).max()),
              max_archived_native_error=float(abs(dx[:,:3,-1]-archive).max()))
    (dest/(target+'_metadata.json')).write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    # Each process publishes the source-independent probe for its own recipient.
    shutil.copy2(probe_paths[target],dest/(target+'_jac.npz'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('target',choices=['bolt','timesfm']);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--original',type=Path,default=ORIGINAL);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(1);run(a.target,a.cache,a.original,a.batch_size)
