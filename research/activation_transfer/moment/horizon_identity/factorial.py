"""Isolate absolute token position and attention to an extra future patch in the public H16 graph."""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,layers,load
from tsfm_lens.extraction.hooks import ActivationCatcher

HERE=Path(__file__).resolve().parent
CELLS=("native8","native16","position8","one_mask","both8")


@contextmanager
def horizon_factors(model,shift_position,hide_second):
    with ExitStack() as stack:
        if shift_position:
            mod=model.module.patch_embedding.position_embedding
            def shift(layer,inputs,output):
                n=output.shape[-2]
                moved=layer.pe[:,1:n+1].to(output)
                if moved.shape!=output.shape:raise ValueError("Unexpected positional layout")
                return moved
            hook=mod.register_forward_hook(shift)
            stack.callback(hook.remove)
        if hide_second:
            def drop(_,args,kwargs):
                mask=kwargs["attention_mask"].clone()
                if mask.shape[1]!=64 or not torch.all(mask[:,-1]==1):raise ValueError("Unexpected mask")
                mask[:,-1]=0
                return args,{**kwargs,"attention_mask":mask}
            hook=model.module.encoder.register_forward_pre_hook(drop,with_kwargs=True)
            stack.callback(hook.remove)
        yield


def collect(adapter,x,horizon,cell,batch):
    moved=cell in ("position8","both8")
    hidden=cell in ("one_mask","both8")
    layer=layers(adapter)["middle"]
    observed,future=adapter.patch_layout(x.shape[1],horizon)
    pos=[observed.stop-1,future.start]
    out=[];states=[]
    with horizon_factors(adapter,moved,hidden):
        for lo in range(0,len(x),batch):
            with ActivationCatcher(adapter.module,[layer]) as cap:
                y=adapter.predict(x[lo:lo+batch],horizon,[.5])["point"]
                t=cap.collect()[layer]
            out.append(y)
            states.append(t[:,pos].detach().numpy())
    return np.concatenate(out),np.concatenate(states)


def run(size,out,batch):
    adapter=load(size)
    x,ids,split=histories(False)
    cache=CACHE/size
    cache.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    data=dict(ids=ids,domain=split)
    hidden=dict(ids=ids,domain=split)
    for cell in CELLS:
        horizon=8 if cell=="native8" else 16
        data[cell],hidden[cell]=collect(adapter,x,horizon,cell,batch)
        print(size,cell,flush=True)
    paired=np.concatenate([np.flatnonzero(split==s)[:8] for s in np.unique(split)])
    for short,long in ((7,8),(9,16),(17,24)):
        ys,hs=collect(adapter,x[paired],short,"native",batch)
        yl,hl=collect(adapter,x[paired],long,"native",batch)
        np.testing.assert_allclose(ys,yl[:,:short],atol=3e-6,rtol=5e-6)
        np.testing.assert_allclose(hs,hl,atol=3e-5,rtol=2e-5)
        data[f"prefix_{short}_{long}"]=float(np.max(np.abs(ys-yl[:,:short])))
    np.testing.assert_allclose(data["native8"],data["both8"][:,:8],atol=2e-5,rtol=4e-5)
    np.testing.assert_allclose(hidden["native8"],hidden["both8"],atol=.001,rtol=.0005)
    np.savez_compressed(cache/"factors_hidden.npz",**hidden)
    np.savez_compressed(out/(size+"_factors.npz"),**data)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("size",choices=("small","large"))
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(a.threads);run(a.size,a.out,a.batch)
