"""Lock untouched Bitbrains machines and derive live source/target geometry for full-dose feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.capture import HUB, ROOT
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as REP, HUB_REVISION, HERE as REP_RESULTS
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH, extract
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import apply
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
SCRATCH=Path("/tmp/tsfm_depth_feedback_20261121")
OUT=HERE/"results"


def data(scratch,out):
    out.mkdir(exist_ok=True,parents=True)
    scratch.mkdir(exist_ok=True,parents=True)
    manifest=out/"manifest.json"
    if manifest.exists():
        excluded=set(json.loads(manifest.read_text(encoding="utf-8"))["excluded"])
    else:
        excluded=set()
        # The raw item namespace is unique to this archive. Restrict scanning to public studies.
        research=ROOT/"research/activation_transfer"
        files=[p for p in research.rglob("*.json") if HERE not in p.parents]
        files.extend((ROOT/"benchmark_v3/external_real").glob("*.jsonl"))
        for file in files:
            for hit in re.findall(rb"fastStorage_[0-9]+",file.read_bytes()):
                excluded.add(hit.decode("ascii"))
    path=HUB/"datasets--Salesforce--GiftEval/snapshots"/HUB_REVISION/"bitbrains_fast_storage/H/data-00000-of-00001.arrow"
    with pa.memory_map(str(path)) as f:
        table=pa.ipc.open_stream(f).read_all()
    x,ids,seen=[],[],set()
    for raw in np.random.default_rng(6311411).permutation(len(table)):
        i=int(raw)
        key=str(table["item_id"][i].as_py())
        if key in excluded:
            continue
        row=table["target"][i].values[0].values
        if len(row)<488:
            continue
        v=row.slice(len(row)-488,480).to_numpy(zero_copy_only=False).astype(float)
        if not np.isfinite(v).all() or v.std()<1e-8:
            continue
        v=((v-v.mean())/v.std()).astype("f4")
        digest=hashlib.sha256(v.tobytes()).hexdigest()
        if digest in seen:
            continue
        x.append(v); ids.append("machines/"+key); seen.add(digest)
        if len(x)==256:
            break
    if len(x)!=256:
        raise ValueError(f"Only {len(x)} usable new machines")
    new=np.stack(x)
    with path.open("rb") as f:
        file_hash=hashlib.file_digest(f,"sha256").hexdigest()
    locked=dict(seed=6311411,items=ids,excluded=sorted(excluded),file_sha256=file_hash,
                input_sha256=hashlib.sha256(new.tobytes()).hexdigest(),revision=HUB_REVISION)
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))!=locked:
        raise ValueError("Cohort no longer reproduces")
    manifest.write_text(json.dumps(locked,indent=2)+"\n",encoding="utf-8")
    with np.load(REP/"data.npz") as f:
        old=f["split"]=="machines"
        if not set(f["ids"][old]).isdisjoint(ids):
            raise ValueError("Old/new machine collision")
        np.savez_compressed(scratch/"data.npz", x=np.concatenate((f["x"][old],new)),ids=np.r_[f["ids"][old],ids],
                            split=np.repeat(["original","new"],[sum(old),len(ids)]))
    print("locked",len(ids),"new machines after",len(excluded),"exclusions",flush=True)


def collect(name,scratch,batch):
    with np.load(scratch/"data.npz") as f:
        x,ids,split=f["x"],f["ids"],f["split"]
    enc=adapter_for(name)
    for p in enc.module.parameters():
        p.requires_grad_(False)
    spans=enc.token_time_spans()
    pos=np.flatnonzero(spans[:,0]>=448)
    sel=last_window_slice(enc,pos)
    layers=enc.all_layer_names()
    depths=(3,) if name=="bolt" else (0,10,19)
    sites=[layers[d] for d in depths]
    jacs={l:[] for l in sites}
    states={l:[] for l in sites}
    ordinary=[]
    for start in range(0,len(x),batch):
        inp=x[start:start+batch]
        h,j,y=extract(enc,sites,pos,inp,True)
        b=enc.predict(inp,16,[.5])["quantiles"][...,0]
        np.testing.assert_allclose(b[:,:8],y,atol=8e-5,rtol=3e-5)
        ordinary.append(b)
        for l in sites:
            jacs[l].append(j[l]); states[l].append(h[l])
    base=np.concatenate(ordinary)
    for dep,l in zip(depths,sites):
        h,j=np.concatenate(states[l]),np.concatenate(jacs[l])
        with np.load(DEPTH/f"{name}_{dep}.npz") as old:
            use=old["split"]=="machines"
            np.testing.assert_array_equal(ids[split=="original"],old["ids"][use])
            np.testing.assert_allclose(j[split=="original"],old["jac"][use],atol=5e-5,rtol=.002)
            radius=float(old["radius"])
        result=dict(ids=ids,split=split,jac=j,states=h,base=base,radius=radius,layer=l,pos=pos)
        if name=="bolt":
            with np.load(REP_RESULTS/"results/bolt_seed0.npz") as f:
                atoms=f["atoms"]
            sae=load_sae_checkpoint(str(dictionary_path(name)))
            with torch.inference_mode():
                codes=sae.encode(torch.from_numpy(h.mean(axis=1))).numpy()[:,atoms]
            dec=sae.W_dec.detach().numpy()[atoms].T
            native,_=apply(enc,l,sel,x,h,base,codes,dec,np.inf,32,forecast16)
            result.update(codes=codes,atoms=atoms,dec=dec,native=native)
        np.savez_compressed(scratch/f"{name}_{dep}.npz",**result)
        print(name,dep,"done",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("data","bolt","timesfm"))
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=8)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=="data":
        data(a.scratch,a.out)
    else:
        collect(a.stage,a.scratch,a.batch)
