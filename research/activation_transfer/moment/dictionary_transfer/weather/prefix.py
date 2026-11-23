"""Reuse exact ordinary pre-intervention states while the original encoder suffix and head run live."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import numpy as np
import torch


class EndPrefix(Exception):
    """Stop a native call immediately after it produces the desired full ordinary middle state."""


def obtain(a,x,middle,slot,cache,signature,batch):
    path=cache/'ordinary_prefix.npz'
    with (cache/'ordinary_prefix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            with np.load(path) as f:
                assert str(f['signature'])==signature
                return f['states']
        results=[]
        def take(_,args,out):
            results.append(out[0].detach().numpy().copy())
            raise EndPrefix()
        ref=a.module.get_submodule(middle)
        hook=ref.register_forward_hook(take)
        try:
            for i in range(0,len(x),batch):
                try:a.predict(x[i:i+batch],16,[.5])
                except EndPrefix:continue
                raise AssertionError('Middle capture did not intercept the native call')
        finally:hook.remove()
        states=np.concatenate(results)
        assert len(states)==len(x) and states.shape[1]>slot
        tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,states=states,signature=signature);tmp.replace(path)
        return states


@contextmanager
def cached_middle(a,states,site):
    """First block computes the real masked position bias; exact saved states enter the suffix."""
    blocks=a.module.encoder.block
    idx=int(site.rsplit('.',1)[1]);saved=[]
    def pass_first(hidden,attention_mask,position_bias,*args,**kwargs):
        assert position_bias is not None and not kwargs.get('output_attentions',False)
        return hidden,position_bias
    def restore_middle(hidden,attention_mask,position_bias,*args,**kwargs):
        native=torch.from_numpy(states).to(hidden)
        assert native.shape==hidden.shape
        return native,position_bias
    try:
        for i in range(1,idx+1):
            block=blocks[i];saved.append((block,block.forward))
            block.forward=restore_middle if i==idx else pass_first
        yield
    finally:
        for block,func in saved:block.forward=func
