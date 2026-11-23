"""Compare the current active span with another history, clean-state PCA and isotropic spans."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.dictionary_prior.run import load_target
from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import OUT
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.run import HERE as SUBSPACE, CHECKPOINTS, FIT
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from tsfm_lens.sae.train import load_sae_checkpoint

ARMS=("active","other_history","pca","isotropic")


def run(model,source_seed,out,batch):
    target_seed=3-source_seed
    data=load_target(model)
    use=data["split"]=="extra_daily"
    ids,x,state,base=[data[k][use] for k in ("ids","x","states","base")]
    with np.load(SUBSPACE/"results"/f"{model}_seed{source_seed}.npz") as f:
        take=f["split"]=="extra_daily"
        np.testing.assert_array_equal(ids,f["ids"][take])
        codes,atoms=f["codes"][take],f["atoms"]
        ref=f["native"][take,:,:,1]
    with np.load(FIT/(model+".npz")) as f:
        pca,iso=f["pca"],f["random"]
    src=load_sae_checkpoint(str(CHECKPOINTS/f"seed{source_seed}_{model}.pt"))
    dst=load_sae_checkpoint(str(CHECKPOINTS/f"seed{target_seed}_{model}.pt"))
    support,_=active_codes(dst,state)
    d=dst.W_dec.detach().numpy().astype(float)
    perm=np.random.default_rng(181970).permutation(len(ids))
    other=np.empty_like(perm)
    other[perm]=np.roll(perm,1)
    assert np.all(other!=np.arange(len(ids)))
    # The right singular rows provide an orthonormal basis for the same active span.
    _,singular,basis=np.linalg.svd(d[support],full_matrices=False)
    if np.any(singular[:,-1]<1e-7*singular[:,0]):
        raise ValueError("Rank-deficient active span needs an explicit lower-rank control")
    row,col=np.nonzero(codes>0)
    v=-codes[row,col,None]*src.W_dec.detach().numpy()[atoms[col]].astype(float)
    outdata=dict(ids=ids,mask=codes>0,rows=row,atom_index=col,source=ref[row,col],
                 control_history_index=other,overlap=np.array([len(np.intersect1d(a,b)) for a,b in zip(support,support[other])]))
    enc=adapter_for(model)
    sel=last_window_slice(enc,data["positions"])
    for arm in ARMS:
        effects=[]
        fractions=[]
        clips=[]
        for start in range(0,len(v),batch):
            ix=slice(start,start+batch)
            r=row[ix]
            mat={"active":basis[r],"other_history":basis[other[r]],"pca":pca,"isotropic":iso}[arm]
            if mat.ndim==2:
                vec=(v[ix]@mat.T)@mat
            else:
                vec=np.einsum("nk,nkd->nd",np.einsum("nkd,nd->nk",mat,v[ix]),mat)
            length=np.linalg.norm(vec,axis=1)
            scale=np.minimum(1.,.2*data["radius"]/np.maximum(length,1e-15))
            y=forecast16(enc,MODELS[model][-1],x[r],state[r],(vec*scale[:,None]).astype("f4"),sel,batch)
            effects.append(y-base[r])
            fractions.extend((length/np.linalg.norm(v[ix],axis=1))**2)
            clips.extend(scale<1.)
        outdata[arm]=np.concatenate(effects)
        outdata[arm+"_fraction"]=np.array(fractions)
        outdata[arm+"_capped"]=np.array(clips)
    with np.load(out/f"{model}_{source_seed}_{target_seed}.npz") as previous:
        pick=previous["split"][previous["rows"]]=="extra_daily"
        np.testing.assert_array_equal(previous["ids"][previous["rows"][pick]],ids[row])
        np.testing.assert_array_equal(previous["atom_index"][pick],col)
        np.testing.assert_allclose(outdata["active"],previous["hidden_span"][pick],rtol=.004,atol=4e-4)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"span_{model}_{source_seed}.npz",**outdata)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("seed",type=int,choices=(1,2))
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.model,a.seed,a.out,a.batch)
