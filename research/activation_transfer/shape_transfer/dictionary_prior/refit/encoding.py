"""Feed the identical physical hidden knockout to another fit's hard and clean-locked encoders."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.boundary.smooth import smooth_codes
from research.activation_transfer.shape_transfer.dictionary_prior.run import load_target
from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import OUT
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.run import HERE as SUBSPACE, CHECKPOINTS
from tsfm_lens.sae.train import load_sae_checkpoint

ARMS=("true","hard","locked","hard_eighth","locked_eighth")


def endpoint(sae,state,change,fraction):
    clean=sae.encode(state)
    pre=sae.pre_activations(state+fraction*change)
    edited=sae.sparsify(pre)
    locked=pre.relu()*(clean>0)
    switch=((edited>0)&(clean==0)).sum(1)
    hard=(edited-clean)@sae.W_dec/fraction
    locked=(locked-clean)@sae.W_dec/fraction
    return hard,locked,switch


def regularized(sae,state,change,hard,locked):
    clean,_=smooth_codes(sae.pre_activations(state),sae.k)
    edited,_=smooth_codes(sae.pre_activations(state+change),sae.k)
    shoulder=(edited["shoulder"]-clean["shoulder"])@sae.W_dec
    length=change.norm(dim=-1,keepdim=True)
    moves={"true":change,"hard":hard,"locked":locked,"shoulder":shoulder}
    for name in ("hard","locked","shoulder"):
        vec=moves[name]
        moves[name+"_equal"]=vec*(length/vec.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    return moves


def run(model,source_seed,out,batch,compare_norms=False):
    target_seed=3-source_seed
    target=load_target(model)
    use=target["split"]=="extra_daily"
    ids,x,state,base=[target[k][use] for k in ("ids","x","states","base")]
    with np.load(SUBSPACE/"results"/f"{model}_seed{source_seed}.npz") as f:
        keep=f["split"]=="extra_daily"
        np.testing.assert_array_equal(f["ids"][keep],ids)
        codes,atoms=f["codes"][keep],f["atoms"]
        ref=f["native"][keep,:,:,1]
    row,col=np.nonzero(codes>0)
    src=load_sae_checkpoint(str(CHECKPOINTS/f"seed{source_seed}_{model}.pt"))
    dst=load_sae_checkpoint(str(CHECKPOINTS/f"seed{target_seed}_{model}.pt"))
    v=-codes[row,col,None]*src.W_dec.detach().numpy()[atoms[col]]
    enc=adapter_for(model)
    layer=MODELS[model][-1]
    sel=last_window_slice(enc,target["positions"])
    rec=dict(ids=ids,atoms=atoms,rows=row,atom_index=col,source=ref[row,col],active=codes>0,
             switch=np.empty(len(row),int),switch_eighth=np.empty(len(row),int))
    methods=("true","hard","locked","shoulder","hard_equal","locked_equal","shoulder_equal") if compare_norms else ARMS
    for arm in methods:
        rec[arm]=np.empty((len(row),16),dtype="f4")
        for field in ("length_ratio","hidden_error","capped","cosine"):
            rec[arm+"_"+field]=np.empty(len(row),dtype=bool if field=="capped" else "f4")
    for start in range(0,len(row),batch):
        ix=slice(start,start+batch)
        r=row[ix]
        h=torch.from_numpy(state[r].mean(1))
        actual=torch.from_numpy(v[ix])
        arms={"true":actual}
        with torch.inference_mode():
            for frac,suffix in ((1.,""),(.125,"_eighth")):
                hard,locked,switch=endpoint(dst,h,actual,frac)
                arms["hard"+suffix]=hard
                arms["locked"+suffix]=locked
                rec["switch"+suffix][ix]=switch.numpy()
                same=switch==0
                if same.any():
                    torch.testing.assert_close(hard[same],locked[same],rtol=1e-5,atol=5e-5)
            if compare_norms:
                arms=regularized(dst,h,actual,arms["hard"],arms["locked"])
        for arm,vec in arms.items():
            raw=vec.numpy()
            lengths=np.linalg.norm(raw,axis=1)
            fraction=np.minimum(1.,.2*target["radius"]/np.maximum(lengths,1e-15))
            write=raw*fraction[:,None]
            rec[arm+"_capped"][ix]=fraction<1.
            true_len=np.linalg.norm(v[ix],axis=1)
            rec[arm+"_cosine"][ix]=np.sum(raw*v[ix],axis=1)/np.maximum(lengths*true_len,1e-15)
            rec[arm+"_length_ratio"][ix]=lengths/true_len
            rec[arm+"_hidden_error"][ix]=np.linalg.norm(write-v[ix],axis=1)/true_len
            y=forecast16(enc,layer,x[r],state[r],write.astype("f4"),sel,batch)
            rec[arm][ix]=y-base[r]
    u=~rec["true_capped"]
    np.testing.assert_allclose(rec["true"][u],rec["source"][u],rtol=.004,atol=4e-4)
    out.mkdir(exist_ok=True,parents=True)
    if not compare_norms:
        for arm in methods:
            del rec[arm+"_cosine"]
    prefix="normgate" if compare_norms else "encode"
    np.savez_compressed(out/f"{prefix}_{model}_{source_seed}.npz",**rec)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("seed",type=int,choices=(1,2))
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=24)
    p.add_argument("--compare-norms",action="store_true")
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.model,a.seed,a.out,a.batch,a.compare_norms)
