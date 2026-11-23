"""Fix the real-item split and retrieve aligned clean window states without forecast labels."""
import argparse
from hashlib import sha256, file_digest
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

HERE=Path(__file__).resolve().parent
from research.activation_transfer.boundary.capture import adapter_for, dictionary_path, MODELS
from tsfm_lens.extraction.extract import capture_raw_tokens

HUB=Path(os.environ.get("HF_HUB_CACHE", Path.home()/".cache/huggingface/hub"))
CACHE=Path(os.environ.get("TSFM_ATOM_CACHE", "/tmp/tsfm_atoms_20261120"))
ELECTRICITY_ARROW=Path(os.environ.get("TSFM_ELECTRICITY_ARROW",
    Path.home()/".cache/huggingface/datasets/autogluon___chronos_datasets/monash_electricity_hourly"
    /"0.0.0/eeecad0b82a8c237e212ce6f8d1abecb513e2cec/chronos_datasets-train.arrow"))
NTRAIN=1024
NVAL=96


def contexts(path, n, seed, seen, excluded_ids=()):
    """One unique 480-step history per item, with an eight-step trailing buffer."""
    with pa.memory_map(str(path)) as stream:
        table=pa.ipc.open_stream(stream).read_all()
    name="item_id" if "item_id" in table.column_names else "id"
    ids, xs=[],[]
    for i in np.random.default_rng(seed).permutation(len(table)):
        row=table["target"][int(i)].values
        if len(row)<488:
            continue
        x=row.slice(len(row)-488,480).to_numpy(zero_copy_only=False).astype(float)
        if not np.isfinite(x).all() or x.std()<1e-8:
            continue
        x=((x-x.mean())/x.std()).astype("f4")
        key=sha256(x.tobytes()).digest()
        item=str(table[name][int(i)].as_py())
        if key in seen or item in ids or item in excluded_ids:
            continue
        ids.append(item)
        xs.append(x)
        seen.add(key)
        if len(xs)==n:
            break
    if len(xs)!=n:
        raise ValueError(f"Only {len(xs)} eligible in {path}")
    return ids,np.stack(xs)


def prepare(cache,out):
    files=[HUB/"datasets--Salesforce--GiftEval/snapshots/30841734ac5cfddbd0c3bad6d09d2b6b32becbb0/m4_daily/data-00000-of-00001.arrow",
           ELECTRICITY_ARROW]
    xs,ids,seen=[],[],set()
    meta={}
    names=["daily","electricity"]
    for i,(dataset,path,count) in enumerate(zip(names,files,[NTRAIN+NVAL+160,160])):
        items,x=contexts(path,count,2026112014+i,seen)
        ids.extend(dataset+"/"+item for item in items)
        xs.append(x)
        with path.open("rb") as reader:
            meta[dataset]=dict(sha256=file_digest(reader,"sha256").hexdigest(),source=path.name)
    split=np.array(["train"]*NTRAIN+["cal"]*NVAL+["daily"]*160+["electricity"]*160)
    x=np.vstack(xs)
    np.savez_compressed(cache/"data.npz",x=x,ids=ids,split=split)
    (out/"manifest.json").write_text(json.dumps(dict(files=meta,context_sha256=sha256(x.tobytes()).hexdigest(),
            split=dict(zip(ids,split.tolist()))),indent=2)+"\n")


def capture(name,cache):
    with np.load(cache/"data.npz") as z:
        x,ids=z["x"],z["ids"]
    enc=adapter_for(name)
    layer=MODELS[name][-1]
    spans=enc.token_time_spans()
    chosen=np.where(spans[:,0]>=480-32)[0]
    if (spans[chosen[0],0]!=448 or spans[chosen[-1],1]!=480 or
        not np.array_equal(spans[chosen[1:],0],spans[chosen[:-1],1])):
        raise ValueError("Patches do not cover the final 32 observations contiguously")
    out=[]
    for i in range(0,len(x),24):
        out.append(capture_raw_tokens(enc,x[i:i+24],[layer])[layer][:,chosen,:].numpy())
    with dictionary_path(name).open("rb") as reader:
        sae_hash=file_digest(reader,"sha256").hexdigest()
    np.savez_compressed(cache/(name+".npz"),tokens=np.concatenate(out),positions=chosen,ids=ids,sae_sha256=sae_hash)
    print("captured",name,layer,len(ids),"histories",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["data",*MODELS]); p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=HERE/"results")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); a.cache.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    if a.stage=="data":
        prepare(a.cache,a.out)
    else:
        capture(a.stage,a.cache)
