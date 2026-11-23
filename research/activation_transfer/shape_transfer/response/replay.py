"""Independently replay archived physical writes with one direct PyTorch pre-hook."""
import argparse
import json

import numpy as np
import torch
from research.activation_transfer.forecast_requests.run import TEST, DATA_ROOT, adapter_for, data
from research.activation_transfer.shape_transfer.response.probe import HERE


def replay(target):
    root=HERE/'results';ids,x=data(DATA_ROOT)
    with np.load(root/(target+'_effects.npz')) as f:
        np.testing.assert_array_equal(ids[TEST],f['ids']);directions=f['applied_write_directions']; expected=f['effects']
        choose=np.r_[np.arange(4),128+np.arange(4)]
        names=f['ids'][choose].tolist()
    x=x[TEST][choose]; expected=expected[choose]; shape=expected.shape
    site=json.loads((root/(target+'_metadata.json')).read_text())['site']
    if site['path']!=['args',0] or site['mode']!='last': raise ValueError('Unsupported recorded input tree')
    adapter=adapter_for(target); block=adapter.module.get_submodule(site['module']); baselines=adapter.predict(x,8,[.5])['quantiles'][:,:,0]
    arms=directions.reshape(-1,directions.shape[-1]); indices=np.arange(len(x)*len(arms)); altered=[]
    for packed in np.array_split(indices,20):
        rows,arm=np.divmod(packed,len(arms)); calls=[]
        def inject(module,args):
            if list(args[0].shape[1:])!=site['shape']: raise ValueError('Unexpected live model path')
            moved=args[0].clone();moved[:,-1]+=torch.as_tensor(arms[arm],dtype=moved.dtype,device=moved.device)
            calls.append(1);return (moved,*args[1:])
        h=block.register_forward_pre_hook(inject)
        try:
            altered.append(adapter.predict(x[rows],8,[.5])['quantiles'][:,:,0]-baselines[rows])
        finally:
            h.remove()
        if len(calls)!=1: raise ValueError('Hook did not fire once on the actual recipient path')
    direct=np.concatenate(altered).reshape(shape);np.testing.assert_allclose(direct,expected,rtol=2e-3,atol=3e-4)
    result=dict(target=target,ids=names,n_forecast_vectors=int(np.prod(shape[:-1])),
                max_abs_error=float(np.max(abs(direct-expected))),rms_error=float(np.sqrt(np.mean((direct-expected)**2))))
    (root/(target+'_replay.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('target',choices=['bolt','timesfm']);a=p.parse_args();torch.set_num_threads(1);replay(a.target)
