"""Regenerate a terminal design matrix in restartable blocks, checking a concurrent full replay."""
import argparse
import os
import numpy as np
import torch
from research.activation_transfer.moment.head_test.data import SCRATCH
from research.activation_transfer.moment.head_test.capture import adapter,collect


def run(size,batch):
    folder=SCRATCH/(size+"_0")
    parts=folder/"terminal_parts";parts.mkdir(exist_ok=True)
    dest=folder/"terminal.npz"
    if dest.exists():return
    a=adapter(size,0)
    with np.load(SCRATCH/"train.npz") as f:x,ids,split=f["context"],f["ids"],f["split"]
    for lo in range(0,len(x),96):
        if dest.exists():return
        segment=parts/f"{lo}.npz"
        if segment.exists():continue
        vals=x[lo:lo+96]
        r=collect(a,vals[:,:-16],None,batch)
        r["y"]=(vals[:,-16:].reshape(-1,2,8)-r["mu"][:,None,None])/r["sd"][:,None,None]
        temp=segment.with_suffix(".tmp.npz")
        np.savez_compressed(temp,**r)
        temp.replace(segment)
        print(size,lo,flush=True)
    blocks=[]
    for lo in range(0,len(x),96):
        with np.load(parts/f"{lo}.npz") as f:blocks.append(dict(f))
    data={k:np.concatenate([r[k] for r in blocks]) for k in blocks[0]}
    w=a.module.head.linear.weight.detach().numpy();b=a.module.head.linear.bias.detach().numpy()
    pred=(data["h"]@w.T+b)*data["sd"][:,None,None]+data["mu"][:,None,None]
    np.testing.assert_allclose((pred-data["pub"])/data["sd"][:,None,None],0.,atol=5e-5)
    tmp=folder/f"terminal.{os.getpid()}.npz"
    np.savez_compressed(tmp,ids=ids,split=split,weight=w,bias=b,**data)
    if dest.exists():
        with np.load(dest) as f:np.testing.assert_allclose(f["h"],data["h"],atol=2e-5,rtol=2e-5)
        tmp.unlink()
    else:tmp.replace(dest)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size");p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);run(a.size,a.batch)
