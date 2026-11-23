"""Freeze genuinely disjoint evaluation histories and collect current first-eight Jacobians."""
import argparse
from hashlib import sha256, file_digest
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import adapter_for, capture, contexts, MODELS, CACHE as ATOMS
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as REP, HUB_REVISION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.continuation.probe import SCRATCH as CAL
from research.activation_transfer.shape_transfer.fresh_cohort.data import exclusions, PUBLIC, VERSION, CACHE as OLD, OUT
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from tsfm_lens.extraction.hooks import token_patch

HERE=Path(__file__).resolve().parent
CACHE=Path("/tmp/tsfm_blind_frequency_20261121")


def prepare():
    sets,_=exclusions()
    seen=set()
    for path in (ATOMS,REP,OLD):
        with np.load(path/"data.npz") as f:
            seen.update(sha256(x.tobytes()).digest() for x in f["x"])
            for v in f["ids"]:
                domain,_,item=str(v).partition("/")
                if domain in sets:
                    sets[domain].add(item)
    home=Path.home()/".cache/huggingface"
    paths=[home/"hub/datasets--Salesforce--GiftEval/snapshots"/HUB_REVISION/"m4_daily/data-00000-of-00001.arrow",
           home/"datasets"/PUBLIC/"monash_traffic"/VERSION]
    ids, xs, split, meta=[],[],[],{}
    for domain,seed,path in zip(("daily","monash_traffic"),(20261121071,20261121072),paths):
        items,x=contexts(path,128,seed,seen,sets[domain])
        keys=[domain+"/"+v for v in items]
        with path.open("rb") as f:
            source_hash=file_digest(f,"sha256").hexdigest()
        meta[domain]=dict(seed=seed,items=keys,source=str(path),source_sha256=source_hash,
                          excluded=len(sets[domain]))
        ids.extend(keys); xs.extend(x); split.extend([domain]*len(x))
    x=np.stack(xs)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE/"data.npz",x=x,ids=ids,split=split)
    (HERE/"results"/"manifest.json").write_text(json.dumps(dict(domains=meta,
        input_sha256=sha256(x.tobytes()).hexdigest()),indent=2)+"\n")
    print("frozen",x.shape,flush=True)


def probe(model,batch=8):
    capture(model,CACHE)
    adapter=adapter_for(model)
    h,pos,ids=read_materials(CACHE,model)
    with np.load(CACHE/"data.npz") as f:
        x=f["x"]
        np.testing.assert_array_equal(ids,f["ids"])
    sel=last_window_slice(adapter,pos)
    layer=MODELS[model][-1]
    jac=[]
    for i in range(0,len(x),batch):
        state=torch.from_numpy(h[i:i+batch])
        delta=torch.zeros_like(state[:,0],requires_grad=True)
        with torch.enable_grad(),token_patch(adapter.module,layer,sel,state+delta[:,None]):
            y=adapter.predict_differentiable(x[i:i+batch],8)
            rows=[torch.autograd.grad(y[:,k].sum(),delta,retain_graph=k<7)[0] for k in range(8)]
        jac.append(torch.stack(rows,dim=1).detach().numpy())
    jac=np.concatenate(jac)
    base=forecast16(adapter,layer,x,h,None,sel,batch)
    with np.load(CAL/(model+"_jac16.npz")) as f:
        radius=float(f["radius"])
    idx=np.r_[0:8,128:136]
    v=np.random.default_rng(20261121074).normal(size=(len(idx),h.shape[-1])).astype("f4")
    v*= (.002*radius/np.linalg.norm(v,axis=1))[:,None]
    plus,minus=[forecast16(adapter,layer,x[idx],h[idx],sign*v,sel,batch) for sign in (1,-1)]
    fd=(plus[:,:8]-minus[:,:8])/2
    pred=np.einsum("nhd,nd->nh",jac[idx],v)
    error=np.linalg.norm(fd-pred)/np.linalg.norm(fd)
    if not np.isfinite(jac).all() or error>.08:
        raise ValueError(f"Production finite difference error {error}")
    np.savez_compressed(CACHE/(model+"_early.npz"),ids=ids,jac=jac,forecast=base,fd_error=error)
    print(model,"finite difference",error,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["data",*MODELS])
    a=p.parse_args();torch.set_num_threads(1)
    prepare() if a.stage=="data" else probe(a.stage)
