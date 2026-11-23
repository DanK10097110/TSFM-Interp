"""Measure how the same observed MOMENT SAE feature transfers across forecast requests."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.horizon_identity.train import HERE,SEEDS
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,layers,load
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint


def forecast(adapter,x,horizon,position,state,delta,batch):
    result=[]
    def slot(n):
        if n!=64:raise ValueError(n)
        return slice(position,position+1)
    site=layers(adapter)["middle"]
    for lo in range(0,len(x),batch):
        ix=slice(lo,lo+batch)
        edited=(state[ix]+delta[ix]).astype("f4")[:,None,:]
        with token_patch(adapter.module,site,slot,torch.from_numpy(edited)):
            result.append(adapter.predict(x[ix],horizon,[.5])["point"])
    return np.concatenate(result)


def run(size,seed,batch,out):
    adapter=load(size)
    x,ids,split=histories(False)
    sae=load_sae_checkpoint(str(CACHE/size/f"sae_{seed}.pt"))
    with np.load(HERE/"results"/f"{size}_{seed}_dictionary.npz") as d:
        atoms=d["ids"];directions=d["decoder"]
    with np.load(CACHE/size/"factors_hidden.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"])
        hs={k:f[k] for k in ("native8","native16","position8","one_mask","both8")}
    with np.load(HERE/"results"/(size+"_factors.npz")) as f:
        ordinary={h:f["native"+str(h)] for h in (8,16)}
    with torch.inference_mode():
        codes={k:sae.encode(torch.from_numpy(hidden[:,0])).numpy() for k,hidden in hs.items()}
    z8=codes["native8"][:,atoms]
    z16=codes["native16"][:,atoms]
    row,col=np.nonzero(z8>0)
    direction=directions[col]
    direct=-z8[row,col,None]*direction
    out.mkdir(exist_ok=True,parents=True)
    rec=dict(ids=ids,split=split,row=row,col=col,atoms=atoms,z8=z8,z16=z16)
    for cell,c in codes.items():
        rec["support_"+cell]=np.argsort(-c,axis=1)[:,:sae.k].astype("i4")
        rec["code_"+cell]=c[:,atoms]
        rec["norm_"+cell]=np.linalg.norm(hs[cell][:,0],axis=1)
        rec["distance_"+cell]=np.linalg.norm(hs[cell][:,0]-hs["native8"][:,0],axis=1)
    for name,h,index,tok,delta in (
        ("native",8,62,0,direct),("same_vector",16,61,0,direct),
        ("same_feature",16,61,0,-z16[row,col,None]*direction),
        ("wrong_position",16,62,1,direct)):
        state=hs["native"+str(h)][row,tok]
        # Persist independently reproducible blocks for the CPU-heavy large encoder.
        parts=CACHE/size/f"horizon_ablate_{seed}"/name
        parts.mkdir(exist_ok=True,parents=True)
        blocks=[]
        for lo in range(0,len(row),96):
            seq=slice(lo,lo+96)
            fname=parts/f"{lo}.npy"
            if not fname.exists():
                zero=forecast(adapter,x[row[seq]],h,index,state[seq],np.zeros_like(delta[seq]),batch)
                np.testing.assert_allclose(zero,ordinary[h][row[seq]],atol=3e-5,rtol=.0003)
                wave=forecast(adapter,x[row[seq]],h,index,state[seq],delta[seq],batch)-zero
                temp=fname.with_suffix(".tmp.npy")
                np.save(temp,wave)
                temp.replace(fname)
            blocks.append(np.load(fname))
        rec[name]=np.concatenate(blocks)
        print(size,seed,name,len(row),flush=True)
    np.savez_compressed(out/f"{size}_{seed}_ablate.npz",**rec)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("size");p.add_argument("seed",type=int,choices=SEEDS)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--batch",type=int,default=16)
    p.add_argument("--out",type=Path,default=HERE/"results")
    a=p.parse_args();torch.set_num_threads(a.threads);run(a.size,a.seed,a.batch,a.out)
