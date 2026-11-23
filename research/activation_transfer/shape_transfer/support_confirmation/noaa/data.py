"""Expose only weather observations and station metadata to transfer fits; check raw-unit forecasts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.support_confirmation.data import adapter_for_length,probe
from research.activation_transfer.shape_transfer.support_confirmation.supports import active_ids,count_overlap
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
CACHE=Path('/tmp/tsfm_support_noaa_20261122')
PANEL=HERE.parents[2]/'moment/forecast_quality/post_release/results/panel.npz'
ORIGINS=('2026-01-10','2026-04-10','2026-07-10','2026-10-10')
SIGNATURE='53c57574063cb50ddf300b6490c383c4f7d1a6c7866e16ae515b48b5300334fb'


def prepare(cache,out):
    with np.load(PANEL) as f:
        context=f['context'];stations=f['ids'];regions=f['region']
    assert context.shape==(46,4,480) and len(set(stations))==46
    assert sha256(context.tobytes()).hexdigest()==SIGNATURE
    x=context.reshape(-1,480).astype('f4')
    assert np.isfinite(x).all() and np.min(x.std(1))>0
    site=np.repeat(np.arange(46),4);origin=np.tile(np.arange(4),46)
    ids=np.array([f'noaa/{stations[i]}/{ORIGINS[j]}' for i,j in zip(site,origin)])
    cache.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache/'data.npz',x=x,ids=ids,split=np.full(len(x),'noaa'),station=site,origin=origin,
                        source_stations=stations,regions=regions,sd=x.astype(float).std(1))
    doc=dict(context_original_sha256=SIGNATURE,context_float32_sha256=sha256(x.tobytes()).hexdigest(),
             source_stations=stations.tolist(),origins=ORIGINS,ids=ids.tolist(),regions=regions.tolist(),
             model_input_units='raw degrees Celsius',effect_units='own historical population SD')
    (out/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')


def verify_normalization(model,cache,out):
    with np.load(cache/'data.npz') as f:x=f['x'];ids=f['ids'];sd=f['sd']
    h,pos,ids_h=read_materials(cache,model)
    np.testing.assert_array_equal(ids,ids_h)
    enc=adapter_for_length(model,480);site=MODELS[model][-1];sae=load_sae_checkpoint(str(dictionary_path(model)))
    i=np.array([0,5,10,15,80,85,180,183]);mean=x[i].astype(float).mean(1)
    scaled=((x[i].astype(float)-mean[:,None])/sd[i,None]).astype('f4')
    alt=capture_raw_tokens(enc,scaled,[site])[site][:,pos].numpy()
    original_ids=active_ids(sae,h[i].mean(1));alt_ids=active_ids(sae,alt.mean(1))
    with np.load(cache/(model+'_early.npz')) as f:base=f['forecast'][i]
    other=forecast16(enc,site,scaled,alt,None,last_window_slice(enc,pos),8)
    gap=(base-mean[:,None])/sd[i,None]-other
    doc=dict(ids=ids[i].tolist(),native_support_overlap=count_overlap(original_ids,alt_ids).tolist(),
             ordinary_forecast_sup_error_in_sd=np.max(abs(gap),axis=1).tolist(),
             pooled_hidden_max_abs=float(np.max(np.abs(alt.mean(1)-h[i].mean(1)))))
    (out/(model+'_normalization.json')).write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['data',*MODELS]);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='data':prepare(a.cache,a.out)
    else:
        probe(a.stage,a.cache,8)
        verify_normalization(a.stage,a.cache,a.out)
