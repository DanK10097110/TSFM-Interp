"""Prospectively test full-dose dictionary gate changes on new daily and weekly items."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import CACHE as ATOMS, MODELS,adapter_for,capture,contexts
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH as BOUNDARY,HUB,excluded_ids
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as REP,HUB_REVISION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH
from research.activation_transfer.shape_transfer.dictionary_prior.refit.encoding import endpoint,regularized
from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import OUT
from research.activation_transfer.shape_transfer.fresh_cohort.data import exclusions,CACHE as EXTRA
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.run import HERE as SUBSPACE,CHECKPOINTS
from research.activation_transfer.shape_transfer.rolling.data import CACHE as ROLLING
from tsfm_lens.sae.train import load_sae_checkpoint

TMP=Path("/tmp/tsfm_gate_confirm_20261121")
ARMS=("true","hard","locked","shoulder","hard_equal","locked_equal","shoulder_equal")


def prepare(tmp,out):
    manifest=out/"confirm_manifest.json"
    if manifest.exists():
        # Use the frozen reserve list so a later experiment cannot silently change this sample.
        locked=json.loads(manifest.read_text())
        ids,windows,splits=[],[],[]
        seen=set()
        for domain,meta in locked["domains"].items():
            raw=HUB/"datasets--Salesforce--GiftEval/snapshots"/HUB_REVISION/("m4_"+domain)/"data-00000-of-00001.arrow"
            keys,obs=contexts(raw,len(meta["items"]),meta["seed"],seen,meta["excluded"])
            np.testing.assert_array_equal(keys,meta["items"])
            ids.extend(f"{domain}/{k}" for k in keys)
            windows.extend(obs)
            splits.extend([domain]*len(keys))
        x=np.stack(windows)
        if ids!=locked["ids"] or sha256(x.tobytes()).hexdigest()!=locked["input_sha256"]:
            raise ValueError("Raw source does not reproduce the locked input panel")
        tmp.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(tmp/"data.npz",x=x,ids=ids,split=splits)
        return
    exclude,_=exclusions()
    skip={"daily":set(exclude["daily"]),"weekly":set(excluded_ids("weekly"))}
    seen=set()
    for root in (ATOMS,REP,EXTRA,BLIND,ROLLING):
        with np.load(root/"data.npz") as f:
            for id_,x in zip(f["ids"],f["x"]):
                domain,key=str(id_).split("/")[-2:]
                if domain in skip:
                    skip[domain].add(key)
                seen.add(sha256(x.tobytes()).digest())
    with np.load(BOUNDARY/"weekly_bolt.npz") as f:
        skip["weekly"].update(map(str,f["ids"]))
    ids,x,split,meta=[],[],[],{}
    for domain,seed in (("daily",1102189),("weekly",1102191)):
        raw=HUB/"datasets--Salesforce--GiftEval/snapshots"/HUB_REVISION/("m4_"+domain)/"data-00000-of-00001.arrow"
        prior=seen.copy()
        try:
            keys,obs=contexts(raw,96,seed,seen,skip[domain])
        except ValueError as exc:
            import re
            match=re.match(r"Only (\d+) eligible",str(exc))
            if not match or int(match[1])==0:
                raise
            seen=prior
            keys,obs=contexts(raw,int(match[1]),seed,seen,skip[domain])
        assert not (set(keys)&skip[domain])
        ids.extend(f"{domain}/{k}" for k in keys)
        x.extend(obs)
        split.extend([domain]*len(keys))
        meta[domain]=dict(seed=seed,source=str(raw),excluded=sorted(skip[domain]),items=keys)
    x=np.stack(x)
    out.mkdir(exist_ok=True,parents=True)
    tmp.mkdir(exist_ok=True,parents=True)
    doc=dict(domains=meta,ids=ids,input_sha256=sha256(x.tobytes()).hexdigest())
    manifest.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(tmp/"data.npz",x=x,ids=ids,split=split)
    print({k:len(d["items"]) for k,d in meta.items()},flush=True)


def run(model,seed,tmp,out,batch):
    with np.load(tmp/"data.npz") as f:
        ids,x,split=f["ids"],f["x"],f["split"]
    h,pos,check=read_materials(tmp,model)
    np.testing.assert_array_equal(ids,check)
    with np.load(DEPTH/f"{model}_{3 if model=='bolt' else 10}.npz") as f:
        radius=float(f["radius"])
    with np.load(SUBSPACE/"results"/f"{model}_seed{seed}.npz") as f:
        atoms=f["atoms"]
    src=load_sae_checkpoint(str(CHECKPOINTS/f"seed{seed}_{model}.pt"))
    dst=load_sae_checkpoint(str(CHECKPOINTS/f"seed{3-seed}_{model}.pt"))
    clean=torch.from_numpy(h.mean(axis=1))
    with torch.inference_mode():
        codes=src.encode(clean).numpy()[:,atoms]
    rows,cols=np.nonzero(codes>0)
    vec=-codes[rows,cols,None]*src.W_dec.detach().numpy()[atoms[cols]]
    enc=adapter_for(model)
    sel=last_window_slice(enc,pos)
    site=MODELS[model][-1]
    base=forecast16(enc,site,x,h,None,sel,batch)
    data=dict(ids=ids,split=split,rows=rows,atom_index=cols,atoms=atoms,active=codes>0,switch=np.empty(len(rows),int))
    for name in ("source",*ARMS):
        data[name]=np.empty((len(rows),16),dtype="f4")
        data[name+"_capped"]=np.empty(len(rows),dtype=bool)
    for start in range(0,len(rows),batch):
        ix=slice(start,start+batch)
        r=rows[ix]
        v=torch.from_numpy(vec[ix])
        with torch.inference_mode():
            hard,locked,switch=endpoint(dst,clean[r],v,1.)
            data["switch"][ix]=switch.numpy()
            arms=regularized(dst,clean[r],v,hard,locked)
        for name,value in {"source":v,**arms}.items():
            raw=value.numpy()
            length=np.linalg.norm(raw,axis=1)
            scale=np.ones_like(length) if name=="source" else np.minimum(1.,.2*radius/np.maximum(length,1e-12))
            data[name+"_capped"][ix]=scale<1
            data[name][ix]=forecast16(enc,site,x[r],h[r],(raw*scale[:,None]).astype("f4"),sel,batch)-base[r]
    ok=~data["true_capped"]
    np.testing.assert_array_equal(data["true"][ok],data["source"][ok])
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"confirm_{model}_{seed}.npz",**data)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("data","capture","run"))
    p.add_argument("model",nargs="?",choices=MODELS)
    p.add_argument("--seed",type=int,choices=(1,2),default=1)
    p.add_argument("--scratch",type=Path,default=TMP)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=="data": prepare(a.scratch,a.out)
    elif a.stage=="capture": capture(a.model,a.scratch)
    else: run(a.model,a.seed,a.scratch,a.out,a.batch)
