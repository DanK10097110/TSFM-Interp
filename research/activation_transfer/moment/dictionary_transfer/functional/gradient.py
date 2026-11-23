"""Record ordinary forecast Jacobians on separate calibration histories, bounded into restartable blocks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,load,layers

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def run(size,batch):
    train,ids,_=histories(True);_,test_ids,_=histories(False)
    assert set(ids).isdisjoint(test_ids)
    rows=np.random.default_rng(22112306).permutation(len(ids))[:128]
    x=train[rows];ids=ids[rows]
    a=load(size);site=layers(a)["middle"]
    folder=CACHE/"functional"/size;folder.mkdir(exist_ok=True,parents=True)
    chunks=[]
    for start in range(0,len(x),batch):
        dest=folder/f"{start}.npz"
        if not dest.exists():
            obs=x[start:start+batch];saved={}
            def tap(_,args,out):
                v=out[0] if isinstance(out,tuple) else out
                v.requires_grad_(True);saved["hidden"]=v
            handle=a.module.get_submodule(site).register_forward_hook(tap)
            try:
                with torch.enable_grad():
                    y=a.predict_differentiable(obs,16)
                    jac=np.stack([torch.autograd.grad(y[:,lead].sum(),saved["hidden"],retain_graph=lead<7)[0][:,61].detach().numpy()
                                  for lead in range(8)],axis=1)
            finally:handle.remove()
            temp=dest.with_suffix(".tmp.npz")
            np.savez_compressed(temp,ids=ids[start:start+batch],jac=jac);temp.replace(dest)
            print(size,start,flush=True)
        with np.load(dest) as f:np.testing.assert_array_equal(f["ids"],ids[start:start+batch]);chunks.append(f["jac"])
    OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(OUT/f"{size}_gradient.npz",ids=ids,split=["fit"]*96+["cal"]*32,jac=np.concatenate(chunks))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size");p.add_argument("--batch",type=int,default=8)
    a=p.parse_args();torch.set_num_threads(2);run(a.size,a.batch)
