"""Capture mask-token features with the actual published reconstruction and short-forecast graphs."""
import argparse
from hashlib import sha256
import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.data import OUT as QUALITY
from research.activation_transfer.moment.head_test.data import SCRATCH
from research.activation_transfer.moment.token_paths.geometry import ROOT
from tsfm_lens.config import DataConfig,ModelConfig
from tsfm_lens.extraction.hooks import ActivationCatcher
from tsfm_lens.models import build_adapter


def adapter(size,seed):
    torch.manual_seed(seed or 16531)
    cfg=ModelConfig(name=f"MOMENT-{size}",adapter="moment",checkpoint=str(ROOT/size),random_init=bool(seed))
    a=build_adapter(cfg,DataConfig(context_len=480,horizon=16),torch.device("cpu"),torch.float32)
    a.ensure_loaded()
    a.module.requires_grad_(False)
    return a


def scaling(norm):
    """The affine transformation applied by the model to its decoder output."""
    sd=norm.stdev.detach().cpu().numpy().reshape(-1).astype(float)
    mu=norm.mean.detach().cpu().numpy().reshape(-1).astype(float)
    if norm.affine:
        sd=sd/(float(norm.affine_weight.detach().item())+norm.eps**2)
        mu=mu-float(norm.affine_bias.detach().item())*sd
    return mu,sd


def batch_features(a,x,holes=None,horizon=16):
    n=len(x)
    with torch.inference_mode(),ActivationCatcher(a.module,["encoder.final_layer_norm"]) as cap:
        if holes is None:
            y=a.predict(x,horizon,[.5])["point"].reshape(n,-1,8)
            h=cap.collect()["encoder.final_layer_norm"][:,-y.shape[1]:].cpu().numpy()
        else:
            vals,present=a.prepare(x)
            loc=holes+(vals.shape[-1]-x.shape[-1])//8
            if x.shape[-1]%8 or np.min(loc)<0:raise ValueError("Expected complete input patches")
            row=np.arange(n)[:,None]
            allowed=torch.ones_like(present)
            patch_mask=allowed.view(n,-1,8)
            patch_mask[row,loc]=0
            rec=a.module.reconstruction(x_enc=vals,input_mask=present,mask=allowed).reconstruction[:,0]
            y=rec.reshape(n,-1,8).cpu().numpy()[row,loc]
            h=cap.collect()["encoder.final_layer_norm"].cpu().numpy()[row,loc]
        mu,sd=scaling(a.module.normalizer)
    return dict(h=h,pub=y,mu=mu,sd=sd)


def collect(a,x,holes,batch,horizon=16):
    chunks=[]
    for lo in range(0,len(x),batch):
        ix=slice(lo,lo+batch)
        chunks.append(batch_features(a,x[ix],None if holes is None else holes[ix],horizon))
    return {k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]}


def run(size,seed,stage,batch):
    model=adapter(size,seed)
    path=SCRATCH/f"{size}_{seed}"
    path.mkdir(exist_ok=True,parents=True)
    head=model.module.head.linear
    weight=head.weight.detach().numpy()
    bias=head.bias.detach().numpy()
    if stage=="train":
        with np.load(SCRATCH/"train.npz") as f:
            x,holes,ids,split=f["context"],f["holes"],f["ids"],f["split"]
        ix=np.arange(8); corrupted=x[ix].copy().reshape(len(ix),60,8)
        corrupted[ix[:,None],holes[ix]]=np.random.default_rng(916).normal(1000,20,size=(len(ix),8,8))
        clean=batch_features(model,x[ix],holes[ix])
        wrong=batch_features(model,corrupted.reshape(len(ix),480),holes[ix])
        for key in ("h","pub","mu","sd"):
            np.testing.assert_array_equal(wrong[key],clean[key])
        for task,visible,mask in (("interior",x,holes),("terminal",x[:,:-16],None)):
            rec=collect(model,visible,mask,batch)
            if task=="interior":
                target=x.reshape(len(x),60,8)[np.arange(len(x))[:,None],holes]
            else:
                target=x[:,-16:].reshape(len(x),2,8)
            rec["y"]=(target-rec["mu"][:,None,None])/rec["sd"][:,None,None]
            rec.update(ids=ids,split=split,weight=weight,bias=bias)
            # Check saved hidden states and published linear head in actual model units.
            z=rec["h"]@weight.T+bias
            raw=z*rec["sd"][:,None,None]+rec["mu"][:,None,None]
            np.testing.assert_allclose((raw-rec["pub"])/rec["sd"][:,None,None],0.,atol=5e-5)
            np.savez_compressed(path/(task+".npz"),**rec)
            print(size,seed,task,len(x),flush=True)
    else:
        panel=QUALITY/"panel.npz" if stage=="test" else QUALITY.parent/"post_release/results/panel.npz"
        with np.load(panel) as f:
            x=f["context"].reshape(-1,480)
            ids=f["id" if stage=="test" else "ids"]
        for H in (8,16):
            rec=collect(model,x,None,batch,H)
            rec.update(ids=ids,weight=weight,bias=bias,seed=seed)
            np.savez_compressed(path/f"{stage}{H}.npz",**rec)
            print(size,seed,H,len(x),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("size",choices=("small","base","large"))
    p.add_argument("seed",type=int,choices=(0,619,947))
    p.add_argument("stage",choices=("train","test","external"))
    p.add_argument("--threads",type=int,default=2);p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(a.threads);run(a.size,a.seed,a.stage,a.batch)
