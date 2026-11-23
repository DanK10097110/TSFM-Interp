"""Record MOMENT patch states and forecast derivatives on a fixed 16-step public forecast."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.replication.data import CACHE as TRAIN
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as EVAL
from tsfm_lens.config import DataConfig,ModelConfig
from tsfm_lens.extraction.hooks import ActivationCatcher, token_patch
from tsfm_lens.models import build_adapter

HERE=Path(__file__).resolve().parent
CACHE=Path(os.environ.get("TSFM_MOMENT_GEOMETRY","/tmp/tsfm_moment_nov22_geometry"))
ROOT=Path(os.environ.get("TSFM_MOMENT_CHECKPOINTS","/tmp/tsfm_moment_native"))
SLOTS={"history":slice(-3,-2),"first":slice(-2,-1),"second":slice(-1,None),"both":slice(-2,None)}
NATIVE_POSITIONS=np.array([61,62,63])


def load(size):
    spec=ModelConfig(name="MOMENT-"+size,adapter="moment",checkpoint=str(ROOT/size),batch_size=16)
    a=build_adapter(spec,DataConfig(context_len=480,horizon=16),torch.device("cpu"),torch.float32)
    a.ensure_loaded()
    a.module.requires_grad_(False)
    return a


def layers(adapter):
    return dict(middle=f"encoder.block.{len(adapter.module.encoder.block)//2-1}",readout="encoder.final_layer_norm")


def histories(training):
    path=TRAIN if training else EVAL
    with np.load(path/"data.npz") as f:
        mask=f["split"]=="train" if training else np.ones(len(f["ids"]),dtype=bool)
        return f["x"][mask],f["ids"][mask],f["split"][mask]


def capture_training(size,batch):
    adapter=load(size)
    sites=layers(adapter)
    x,ids,_=histories(True)
    _,eval_ids,_=histories(False)
    assert len(set(ids)&set(eval_ids))==0
    cache=CACHE/size
    cache.mkdir(parents=True,exist_ok=True)
    d=adapter.module.config.d_model
    # Dense scratch matrix; the public checkpoint and item IDs suffice to reproduce it.
    observed=np.lib.format.open_memmap(cache/"train_observed.npy",mode="w+",dtype="f4",shape=(len(x),60,d))
    tail={site:np.empty((len(x),3,d),dtype="f4") for site in sites}
    for lo in range(0,len(x),batch):
        ix=slice(lo,lo+batch)
        with ActivationCatcher(adapter.module,list(sites.values())) as cap:
            adapter.predict(x[ix],16,[.5])
            found=cap.collect()
        for site,name in sites.items():
            full=found[name].float().cpu().numpy()
            if full.shape[1]!=64: raise ValueError(full.shape)
            tail[site][ix]=full[:,NATIVE_POSITIONS,:]
            if site=="middle": observed[ix]=full[:,2:62,:]
    observed.flush()
    np.savez_compressed(cache/"train.npz",ids=ids,positions=NATIVE_POSITIONS,**tail)


def capture_geometry(size,batch):
    adapter=load(size)
    sites=layers(adapter)
    x,ids,split=histories(False)
    cache=CACHE/size
    cache.mkdir(parents=True,exist_ok=True)
    d=adapter.module.config.d_model
    record=dict(ids=ids,split=split,positions=NATIVE_POSITIONS,base=np.empty((len(x),16),"f4"))
    for site in sites:
        record[site]=np.empty((len(x),3,d),"f4")
        record[site+"_jac"]=np.empty((len(x),16,3,d),"f4")
    for lo in range(0,len(x),batch):
        ix=slice(lo,lo+batch)
        traced={}
        def tap(name):
            def hook(_,inputs,output):
                v=output if isinstance(output,torch.Tensor) else output[0]
                if not v.requires_grad: v.requires_grad_(True)
                traced[name]=v
            return hook
        handles=[adapter.module.get_submodule(layer).register_forward_hook(tap(name)) for name,layer in sites.items()]
        try:
            with torch.enable_grad():
                y=adapter.predict_differentiable(x[ix],16)
                record["base"][ix]=y.detach().numpy()
                for site,t in traced.items():
                    record[site][ix]=t.detach()[:,NATIVE_POSITIONS].numpy()
                for lead in range(16):
                    grads=torch.autograd.grad(y[:,lead].sum(),tuple(traced[n] for n in sites),retain_graph=lead<15)
                    for name,grad in zip(sites,grads):
                        record[name+"_jac"][ix,lead]=grad[:,NATIVE_POSITIONS].detach().numpy()
        finally:
            for hook in handles:hook.remove()
        if lo%(batch*8)==0:print(size,lo,len(x),flush=True)
    if not np.isfinite(record["middle_jac"]).all(): raise ValueError("Non-finite derivative")
    g=record["readout_jac"]
    np.testing.assert_array_equal(g[:,:,0,:],0.)
    np.testing.assert_array_equal(g[:,8:,1,:],0.)
    np.testing.assert_array_equal(g[:,:8,2,:],0.)
    public=adapter.predict(x[:8],16,[.5])["quantiles"][:,:,0]
    np.testing.assert_allclose(public,record["base"][:8],atol=1e-5,rtol=1e-5)
    np.savez_compressed(cache/"eval.npz",**record)


def physical(adapter,layer,x,states,slot,delta,batch):
    outputs=[]
    nslots=states[:,SLOTS[slot]].shape[1]
    start={"history":61,"first":62,"second":63,"both":62}[slot]
    def select(length):
        if length!=64:raise ValueError("Native MOMENT sequence changed length")
        return slice(start,start+nslots)
    for lo in range(0,len(x),batch):
        ix=slice(lo,lo+batch)
        edited=states[ix,SLOTS[slot]]+delta[ix,None,:].astype("f4")
        with token_patch(adapter.module,layer,select,torch.as_tensor(edited)):
            outputs.append(adapter.predict(x[ix],16,[.5])["quantiles"][:,:,0])
    return np.concatenate(outputs)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("train","eval"))
    p.add_argument("size",choices=("small","large"))
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    (capture_training if a.stage=="train" else capture_geometry)(a.size,a.batch)
