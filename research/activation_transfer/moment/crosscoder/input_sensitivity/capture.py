"""Run native MOMENT at both signs of seven shape pulses and one level pulse on the observed input."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.token_paths.geometry import CACHE as GEOMETRY, histories, layers, load
from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates.run import capture

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
SCRATCH=Path(os.environ.get('TSFM_MOMENT_CROSSCODER','/tmp/tsfm_moment_crosscoder'))/'input_sensitivity'
DOMAINS=('daily','monash_traffic')
SEED=22112441
DOSES=(.15,.075)


def basis():
    patterns=np.zeros((8,8),dtype=float)
    for j in range(1,8):
        patterns[j-1,:j]=np.sqrt(8/(j*(j+1)))
        patterns[j-1,j]=-j*patterns[j-1,0]
    patterns[-1]=1.
    np.testing.assert_allclose(patterns@patterns.T/8,np.eye(8),atol=1e-14)
    return patterns


def panel():
    x,ids,domain=histories(False)
    rng=np.random.default_rng(SEED)
    keep=[]
    for label in DOMAINS:
        where=np.flatnonzero(domain==label)
        assert len(where)==128
        keep.extend(np.sort(rng.choice(where,64,replace=False)))
    keep=np.asarray(keep)
    assert len(set(ids[keep]))==128
    return x[keep],ids[keep],domain[keep],keep


def freeze():
    x,ids,domain,keep=panel()
    OUT.mkdir(parents=True,exist_ok=True)
    data=dict(ids=ids.tolist(),domain=domain.tolist(),indices=keep.tolist(),rng=SEED,
              input_sha256=sha256(x.tobytes()).hexdigest(),patterns=basis().tolist(),pulse_sizes=list(DOSES))
    (OUT/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')


def pulses(clean,amp):
    sd=clean.astype(float).std(1)
    result=np.broadcast_to(clean[:,None,None],(len(clean),8,2,480)).copy()
    signed=np.array([1.,-1.])[None,None,:,None]
    increments=sd[:,None,None,None]*amp*basis()[None,:,None,:]*signed
    result[:,:,:,-8:]+=increments
    np.testing.assert_array_equal(result[:,:,:,:-8],np.broadcast_to(clean[:,None,None,:-8],result[:,:,:,:-8].shape))
    np.testing.assert_allclose((result[:,:,0,-8:]-result[:,:,1,-8:])*.5,increments[:,:,0],atol=2e-6)
    return result,sd*amp


def run(size):
    x,ids,domain,keep=panel()
    manifest=json.loads((OUT/'manifest.json').read_text())
    np.testing.assert_array_equal(manifest['ids'],ids)
    assert manifest['input_sha256']==sha256(x.tobytes()).hexdigest()
    model=load(size)
    history,future=model.patch_layout(x.shape[1],16)
    assert history.stop-1==61 and future.start==62 and future.stop==64
    clean=capture(model,x)
    with np.load(GEOMETRY/size/'eval.npz') as f:
        np.testing.assert_array_equal(f['ids'][keep],ids)
        np.testing.assert_allclose(clean,f['middle'][keep,0],atol=4e-5,rtol=3e-5)
    SCRATCH.mkdir(parents=True,exist_ok=True)
    for amp in DOSES:
        scaled,amplitude=pulses(x,amp)
        # Flattening at a fixed input size is valid for independent eval-mode forwards.
        states=capture(model,scaled.reshape(-1,480)).reshape(len(x),8,2,-1)
        sensitivity=(states[:,:,0].astype(float)-states[:,:,1].astype(float))/(2*amplitude[:,None,None])
        # Spot-replay a split input batch of sign-ordered pairs to catch layout mistakes.
        check=np.array([0,7,36,64,101,127]);j=np.array([0,7,6,1,4,3]);s=np.array([0,1,0,1,1,0])
        verify=capture(model,scaled[check,j,s])
        np.testing.assert_allclose(verify,states[check,j,s],atol=6e-5,rtol=5e-5)
        path=SCRATCH/f'{size}_{amp}.npz'
        tmp=path.with_suffix('.partial.npz')
        np.savez_compressed(tmp,ids=ids,domain=domain,sd=amplitude/amp,sensitivity=sensitivity.astype('f4'),clean=clean,
                            plusminus_replay_max_abs=float(np.max(abs(verify-states[check,j,s]))),site=layers(model)['middle'])
        tmp.replace(path)
        print(size,amp,'completed',len(x)*16,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','small','large'))
    args=p.parse_args();torch.set_num_threads(2)
    if args.stage=='freeze':freeze()
    else:run(args.stage)
