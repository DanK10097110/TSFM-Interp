"""Fit two reproducible TopK dictionaries on 60 observed patches per disjoint training item."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.vectors import choose_atoms
from research.activation_transfer.moment.token_paths.geometry import CACHE
from tsfm_lens.sae.train import SAETrainConfig, train_sae, save_sae

HERE=Path(__file__).resolve().parent
SEEDS=(22112201,22112202)


def tensor_hash(values):
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def run(size,seed):
    cache=CACHE/size
    # train.npz is written after capture is complete.
    with np.load(cache/"train.npz") as f:
        ids=f["ids"]
    obs=np.load(cache/"train_observed.npy",mmap_mode="r")
    n,t,d=obs.shape
    if (n,t)!=(1024,60) or len(set(ids))!=1024:
        raise ValueError("Unexpected training panel")
    train=np.asarray(obs.reshape(-1,d))
    cfg=SAETrainConfig(dict_size_mult=4,k=32,lr=.001,epochs=6,batch_size=512,seed=seed)
    sae,history=train_sae(train,cfg,torch.device("cpu"),context=f"MOMENT {size}, observed middle H16, seed {seed}")
    path=cache/f"sae_{seed}.pt"
    save_sae(sae,path)
    def encode(x):
        with torch.inference_mode():
            return np.concatenate([sae.encode(torch.as_tensor(x[k:k+128].copy())).numpy() for k in range(0,len(x),128)])
    last_codes=encode(obs[:,-1])
    chosen=choose_atoms(last_codes,16,2026112017)
    fit_ids=ids
    with np.load(cache/"eval.npz") as f:
        assert not set(fit_ids)&set(f["ids"])
        ev=f["middle"][:,0]
        domains=f["split"]
    # Use a deterministic uniform subsample of training patches to fit the PCA baseline on all positions.
    rng=np.random.default_rng(1221011)
    ix=rng.choice(len(train),min(len(train),16384),replace=False)
    sample=train[ix].astype(float)
    mu=sample.mean(0)
    gram=(sample-mu).T@(sample-mu)
    pcs=np.linalg.eigh(gram)[1][:,-32:]
    pred_pca=(ev-mu)@pcs@pcs.T+mu
    codes=encode(ev)
    with torch.no_grad(): pred=sae.decode(torch.as_tensor(codes)).numpy()
    sums={}
    for dom in np.unique(domains):
        mask=domains==dom
        y=ev[mask].astype(float)
        energy=np.sum((y-mu)**2)
        sums[str(dom)]=dict(histories=int(mask.sum()),sae_r2=float(1-np.sum((pred[mask]-y)**2)/energy),
                           pca_r2=float(1-np.sum((pred_pca[mask]-y)**2)/energy),
                           eligible_ids=chosen[(codes[mask][:,chosen]>0).sum(0)>=8].tolist(),
                           n_positive=np.sum(codes[mask]>0,axis=1).mean().item())
    (HERE/"results").mkdir(exist_ok=True,parents=True)
    np.savez_compressed(HERE/"results"/f"{size}_{seed}_dictionary.npz",ids=chosen,
                        decoder=sae.W_dec.detach().numpy()[chosen],train_firings=(last_codes[:,chosen]>0).sum(0))
    results=dict(checkpoint=size,seed=seed,hidden=d,dict_size=4*d,k=32,train_rows=len(train),train_series=n,train_sha256=tensor_hash(train),
                 selected=chosen.tolist(),train_meta=sae.train_meta,loss=history,eval=sums)
    (HERE/"results"/f"{size}_{seed}_dictionary.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(size,seed,sums,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("size",choices=("small","large"))
    p.add_argument("seed",type=int,choices=SEEDS)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);run(a.size,a.seed)
