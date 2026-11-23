"""Build adjacent normalized input histories and collect the original atom sites."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from tsfm_lens.extraction.extract import capture_raw_tokens

from research.activation_transfer.atoms.materials import CACHE, HERE as ATOMS, HUB, MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.capture import DATA_REV, digest

HERE=Path(__file__).resolve().parent
SCRATCH=Path('/tmp/tsfm_matched_windows_20261120')


def adjacent(raw):
    raw=np.asarray(raw,dtype=np.float64)
    end=len(raw)-8
    start=end-480
    if start<0:
        raise ValueError('original atom window is too short')
    if start>0 and np.isfinite(raw[start-1]):
        step=-1
    elif end<len(raw) and np.isfinite(raw[end]):
        step=1
    else:
        raise ValueError('neither adjacent window is finite')
    base=raw[start:end]
    mu,sd=base.mean(),base.std()
    if not np.isfinite(base).all() or sd<1e-8:
        raise ValueError('invalid original atom history')
    clean=((base-mu)/sd).astype(np.float32)
    shifted=((raw[start+step:end+step]-mu)/sd).astype(np.float32)
    return clean,shifted,end,step,mu,sd


def inputs(cache,scratch,out):
    path=HUB/'datasets--Salesforce--GiftEval/snapshots'/DATA_REV/'m4_daily/data-00000-of-00001.arrow'
    meta=json.loads((ATOMS/'results/manifest.json').read_text())
    if digest(path)!=meta['files']['daily']['sha256']:
        raise ValueError('M4 snapshot fingerprint changed')
    with np.load(cache/'data.npz') as f:
        sel=f['split']=='daily';train=f['split']=='train';ids=f['ids'][sel];train_ids=f['ids'][train];x=f['x'][sel]
        if hashlib.sha256(f['x'].tobytes()).hexdigest()!=meta['context_sha256']:
            raise ValueError('archived normalized histories changed')
    assert len(x)==160 and len(train_ids)==1024
    np.testing.assert_array_equal(ids,[k for k,v in meta['split'].items() if v=='daily'])
    with pa.memory_map(str(path)) as stream:
        table=pa.ipc.open_stream(stream).read_all()
    lookup={'daily/'+str(v):i for i,v in enumerate(table['item_id'].to_pylist())}
    rows=[adjacent(table['target'][lookup[k]].values.to_numpy(zero_copy_only=False)) for k in ids]
    clean,shift,end,steps,mu,sd=map(np.asarray,zip(*rows))
    np.testing.assert_array_equal(clean,x)
    scratch.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(scratch/'inputs.npz',ids=ids,clean=clean,adjacent=shift,end=end,step=steps)
    report={'source':str(path.name),'data_revision':DATA_REV,'source_sha256':digest(path),
            'original_material_sha256':digest(cache/'data.npz'),'original_manifest_sha256':digest(ATOMS/'results/manifest.json'),
            'ids':ids.tolist(),'map_train_ids':train_ids.tolist(),'end_exclusive':end.tolist(),'step':steps.tolist(),
            'clean_mean':mu.tolist(),'clean_sd':sd.tolist(),'n_backward':int((steps==-1).sum()),
            'n_forward':int((steps==1).sum()),'n_unusable':0,
            'daily_clean_bytes_sha256':hashlib.sha256(clean.tobytes()).hexdigest(),
            'adjacent_bytes_sha256':hashlib.sha256(shift.tobytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


def capture(model,cache,scratch):
    with np.load(scratch/'inputs.npz') as f:
        ids,x,x1=f['ids'],f['clean'],f['adjacent']
    with np.load(cache/'data.npz') as f:
        daily=np.flatnonzero(f['split']=='daily')
    with np.load(cache/(model+'.npz')) as f:
        pos=f['positions']; original=f['tokens'][daily]
        np.testing.assert_array_equal(ids,f['ids'][daily]); assert str(f['sae_sha256'])==digest(dictionary_path(model))
    enc=adapter_for(model);layer=MODELS[model][-1]
    spans=enc.token_time_spans()
    np.testing.assert_array_equal(np.flatnonzero(spans[:,0]>=448),pos)
    expect=np.array([[448,480]]) if model=='timesfm' else np.array([[448,464],[464,480]])
    np.testing.assert_array_equal(spans[pos],expect)
    def tokens(series):
        return np.concatenate([capture_raw_tokens(enc,series[i:i+24],[layer])[layer][:,pos,:].numpy()
                               for i in range(0,len(series),24)])
    current=tokens(x); moved=tokens(x1)
    # Original archive used full 24-row batches with different neighboring items at shard edges.
    err=float(np.max(np.abs(current-original)))
    np.testing.assert_allclose(current,original,atol=8e-5,rtol=2e-5)
    # Measuring both endpoints in matching batches avoids representing a replay difference as motion.
    delta=moved.mean(axis=1)-current.mean(axis=1)
    np.savez_compressed(scratch/f'{model}_shift.npz',ids=ids,delta=delta,positions=pos,
                        spans=spans[pos],native_replay_absmax=err)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['inputs',*MODELS]);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--scratch',type=Path,default=SCRATCH);p.add_argument('--results',type=Path,default=HERE/'results')
    args=p.parse_args();torch.set_num_threads(3)
    if args.stage=='inputs':inputs(args.cache,args.scratch,args.results)
    else:capture(args.stage,args.cache,args.scratch)
