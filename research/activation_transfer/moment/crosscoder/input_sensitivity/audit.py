"""Rebuild signed input pulses and capture middle states while the entire real H16 graph runs."""
import argparse
import json

import numpy as np
import torch
from scipy.linalg import helmert

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.input_sensitivity.capture import DOSES, OUT, SCRATCH, panel
from research.activation_transfer.moment.token_paths.geometry import layers, load
from tsfm_lens.extraction.hooks import ActivationCatcher


def run(size):
    raw,ids,_,_=panel()
    rows=np.array([0,63,64,127]);x=raw[rows]
    native=load(size)
    observed,future=native.patch_layout(480,16)
    assert observed.stop==future.start
    pos=observed.stop-1
    site=layers(native)['middle']
    shapes=np.vstack((np.sqrt(8)*helmert(8),np.ones((1,8))))
    reports={}
    for fraction in DOSES:
        amps=np.std(x,axis=1,dtype=float)*fraction
        with np.load(SCRATCH/f'{size}_{fraction}.npz') as f:
            np.testing.assert_array_equal(f['ids'][rows],ids[rows]);baseline=f['sensitivity'][rows]
        # Reversed sign concatenation and no early stop; construct raw pulses independently.
        hidden=[]
        for sign in (-1,1):
            pieces=[]
            for v in shapes:
                history=x.copy();history[:,-8:]+=sign*amps[:,None]*v
                with ActivationCatcher(native.module,[site]) as hook:
                    forecast=native.predict(history,16,[.5])['point']
                    assert forecast.shape==(4,16)
                    pieces.append(hook.collect()[site][:,pos].numpy())
            hidden.append(np.stack(pieces,axis=1))
        checked=(hidden[1].astype(float)-hidden[0])/(2*amps[:,None,None])
        np.testing.assert_allclose(checked,baseline,atol=9e-4,rtol=9e-4)
        err=checked-baseline
        audits={}
        for method in cd.METHODS:
            for seed in cd.SEEDS:
                with np.load(cd.CACHE/f'{method}_{seed}.npz') as f:
                    axis=f['decoder_'+size].astype(float)
                # Receiving axes are chosen by source-only clean codes throughout, including this audit.
                model,means=cd.load_artifact(method,seed)
                with np.load(SCRATCH/f'small_{fraction}.npz') as f:
                    clean=f['clean'][rows]
                with torch.inference_mode():code=cd.encode_views(model,means,clean).numpy()
                ix=np.nonzero(code>0)[1].reshape(len(rows),32)
                u=axis[ix]/np.linalg.norm(axis[ix],axis=-1,keepdims=True)
                gap=u@err[:,:7].transpose(0,2,1);truth=u@checked[:,:7].transpose(0,2,1)
                ratio=np.linalg.norm(gap,axis=-1)/np.maximum(np.linalg.norm(truth,axis=-1),1e-12)
                audits[f'{method}_{seed}']=dict(max_abs_projection_error=float(np.abs(gap).max()),
                    median_relative_signature_error=float(np.median(ratio)),max_relative_signature_error=float(np.max(ratio)))
        reports[str(fraction)]=dict(identities=ids[rows].tolist(),native_slot=pos,max_abs_hidden_derivative_error=float(np.abs(err).max()),
                                    projected=audits)
    (OUT/f'replay_{size}.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('size',choices=cd.SIZES)
    a=p.parse_args();torch.set_num_threads(2);run(a.size)
