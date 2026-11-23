"""Learn activation-only maps and freeze individual atom matches using ordinary training histories."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.moment.token_paths.geometry import CACHE
from research.activation_transfer.moment.horizon_identity.train import HERE as DICTS,SEEDS
from research.activation_transfer.moment.head_test.fit import ridge
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
SIZES=("small","large")


def dictionary(size,seed):
    return load_sae_checkpoint(str(CACHE/size/f"sae_{seed}.pt"))


def hidden(size):
    return np.load(CACHE/size/"train_observed.npy",mmap_mode="r")


def encode(sae,h):
    with torch.inference_mode():
        return np.concatenate([sae.encode(torch.from_numpy(h[i:i+128].copy())).numpy() for i in range(0,len(h),128)])


def learn_map(source,target):
    a,b=hidden(source),hidden(target)
    with np.load(CACHE/source/"train.npz") as f:ids=f["ids"]
    with np.load(CACHE/target/"train.npz") as f:np.testing.assert_array_equal(ids,f["ids"])
    order=np.random.default_rng(22112251).permutation(len(ids))
    tr,va,te=order[:128],order[128:192],order[192:320]
    def values(arr,ix):return arr[ix].reshape(-1,arr.shape[-1]).astype(float)
    w,bias,meta=ridge(values(a,tr),values(b,tr),values(a,va),values(b,va))
    ref=values(b,te)
    pred=values(a,te)@w+bias
    meta.update(validation_ids=ids[va].tolist(),training_ids=ids[tr].tolist(),blind_map_ids=ids[te].tolist(),
                heldout_hidden_r2=float(1-np.sum((pred-ref)**2)/np.sum((ref-values(b,tr).mean(0))**2)))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/f"{source}_to_{target}_map.npz",w=w)
    (OUT/f"{source}_to_{target}_map.json").write_text(json.dumps(meta,indent=2)+"\n")


def matches(source,target,seed):
    other=SEEDS[1] if seed==SEEDS[0] else SEEDS[0]
    a,b=dictionary(source,seed),dictionary(target,other)
    ca=encode(a,hidden(source)[:,-1]);cb=encode(b,hidden(target)[:,-1]).astype(float)
    with np.load(DICTS/"results"/f"{source}_{seed}_dictionary.npz") as f:atoms,dec=f["ids"],f["decoder"]
    sx=ca[:,atoms].astype(float)
    sx-=sx.mean(0)
    sy=cb-cb.mean(0)
    denom=np.linalg.norm(sx,axis=0)[:,None]*np.linalg.norm(sy,axis=0)
    corr=np.divide(sx.T@sy,denom,out=np.zeros_like(denom),where=denom>0)
    frequent=(cb>0).sum(0)>=8
    corr[:,~frequent]=-np.inf
    corr_ids=corr.argmax(1)
    if source==target:mapped=dec.astype(float)
    else:
        with np.load(OUT/f"{source}_to_{target}_map.npz") as f:mapped=dec@f["w"]
    directions=b.W_dec.detach().numpy().astype(float)
    np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,atol=1e-4)
    cos=mapped@directions.T/np.linalg.norm(mapped,axis=1)[:,None]
    cos[:,~frequent]=-np.inf
    geom_ids=cos.argmax(1)
    random=np.random.default_rng(22112255+seed).choice(np.flatnonzero(frequent),size=len(atoms),replace=False)
    ix=np.arange(len(atoms))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/f"{source}_{seed}_to_{target}_matches.npz",atoms=atoms,mapped=mapped,source_decoder=dec,
                        corr_id=corr_ids,geom_id=geom_ids,random_id=random,
                        corr_best=corr[ix,corr_ids],geom_best=cos[ix,geom_ids],corr_at_geom=corr[ix,geom_ids],
                        cos_at_corr=cos[ix,corr_ids],eligible_target=frequent,recipient_seed=other)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("map","match"));p.add_argument("source",choices=SIZES);p.add_argument("target",choices=SIZES)
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0]);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="map":learn_map(a.source,a.target)
    else:matches(a.source,a.target,a.seed)
