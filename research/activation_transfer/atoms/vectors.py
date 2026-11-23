"""Select source atoms without outcomes, then translate or approximate their native removals."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
import torch

from research.activation_transfer.atoms.materials import CACHE, MODELS, dictionary_path
from tsfm_lens.sae.train import load_sae_checkpoint


def choose_atoms(codes, count, seed):
    rate=np.mean(codes>0,axis=0)
    rng=np.random.default_rng(seed)
    pool=np.flatnonzero((rate>=.05)&(rate<=.60))
    selected=rng.permutation(pool)[:count].tolist()
    if len(selected)<count:
        pool=np.flatnonzero((rate>=.02)&(rate<=.95))
        other=np.setdiff1d(pool,selected)
        selected.extend(rng.permutation(other)[:count-len(selected)].tolist())
    if len(selected)<2:
        raise ValueError("Insufficient eligible source atoms for derangement")
    return np.array(selected)


def bridge(source,target,dirs,seed):
    """Source-side and recipient-side writes using the same clean ridge identification weights."""
    x=source.astype(float)-source.mean(0,dtype=float)
    y=target.astype(float)-target.mean(0,dtype=float)
    gram=x@x.T
    top=eigh(gram,subset_by_index=[len(x)-1,len(x)-1],eigvals_only=True)[0]
    alpha=.01*top
    weights=cho_solve(cho_factor(gram+np.eye(len(x))*alpha),x@dirs.T).T
    full=weights@y
    sham=weights@y[np.random.default_rng(seed).permutation(len(x))]
    sham*=np.linalg.norm(full,axis=1,keepdims=True)/np.maximum(np.linalg.norm(sham,axis=1,keepdims=True),1e-15)
    return full,sham,weights@x


def approximate(direction,dictionary,ids):
    """Fit nested signed write directions by residual correlation and least-squares refitting."""
    unit=dictionary[ids].astype(float)
    unit/=np.maximum(np.linalg.norm(unit,axis=-1,keepdims=True),1e-15)
    residual=direction.copy()
    chosen=[]
    records={}
    for k in range(1,min(len(ids),16)+1):
        scores=np.abs(unit@residual)
        scores[chosen]=-np.inf
        chosen.append(int(np.argmax(scores)))
        rows=unit[chosen]
        weights=np.linalg.lstsq(rows.T,direction,rcond=1e-8)[0]
        pred=weights@rows
        residual=direction-pred
        if k in (1,4,16):
            records[k]=(pred,ids[chosen].copy())
    if 16 not in records:
        raise ValueError("Insufficient active destination dictionary atoms")
    return records


def coactive(source_codes,target_codes,targets):
    """Highest positive Pearson correlation between train codes, without causal labels."""
    x=source_codes.astype(float)-source_codes.mean(0,dtype=float)
    y=target_codes[:,targets].astype(float)-target_codes[:,targets].mean(0,dtype=float)
    norm=np.linalg.norm(x,axis=0)[:,None]*np.linalg.norm(y,axis=0)
    corr=(x.T@y)/np.maximum(norm,1e-15)
    index=np.argmax(corr,axis=1)
    return targets[index],corr[np.arange(len(index)),index]


def make(cache,name):
    source=name
    target=next(m for m in MODELS if m!=source)
    with np.load(cache/"data.npz") as data:
        train=data["split"]=="train"
        ids=data["ids"]
    states={}
    codes={}
    dec={}
    sha={}
    for model in [source,target]:
        with np.load(cache/(model+".npz")) as z:
            np.testing.assert_array_equal(ids,z["ids"])
            states[model]=z["tokens"].mean(axis=1)
            sha[model]=str(z["sae_sha256"])
        sae=load_sae_checkpoint(str(dictionary_path(model)))
        with torch.inference_mode():
            codes[model]=sae.encode(torch.from_numpy(states[model])).numpy()
            dec[model]=sae.W_dec.numpy().copy()
    selected=choose_atoms(codes[source][train],16,2026112017)
    eligible=np.flatnonzero(np.mean(codes[target][train]>0,axis=0)>=.01)
    dirs=dec[source][selected]
    concept,traincorr=coactive(codes[source][train][:,selected],codes[target][train],eligible)
    full,sham,projection=bridge(states[source][train],states[target][train],dirs,2026112018)
    approximations=[approximate(d,dec[target],eligible) for d in full]
    rng=np.random.default_rng(2026112019)
    deranged=[]
    for _ in range(4):
        order=rng.permutation(len(selected))
        perm=np.empty_like(order)
        perm[order]=np.roll(order,1)
        deranged.append(perm)
    arrays=dict(ids=ids,source=source,target=target,source_atoms=selected,target_match=concept,
                target_match_corr=traincorr,target_match_codes=codes[target][:,concept],
                source_codes=codes[source][:,selected],source_decoder=dirs,source_projection=projection,
                target_dense=full,target_sham=sham,target_match_decoder=dec[target][concept],
                wrong_pairs=np.asarray(deranged),target_eligible=len(eligible),
                source_sae_sha=sha[source],target_sae_sha=sha[target])
    for k in (1,4,16):
        arrays[f"target_omp{k}"]=np.stack([record[k][0] for record in approximations])
        arrays[f"target_omp{k}_ids"]=np.stack([record[k][1] for record in approximations])
    np.savez_compressed(cache/(source+"_vectors.npz"),**arrays)
    print(source,"features",selected,"recipient candidates",len(eligible),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS); p.add_argument("--cache",type=Path,default=CACHE)
    a=p.parse_args()
    torch.set_num_threads(3)
    make(a.cache,a.source)
