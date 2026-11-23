"""Domain-informed atom screening on independent electricity customers."""
import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import CACHE as ORIGINAL, ELECTRICITY_ARROW, MODELS, adapter_for, capture, contexts, dictionary_path
from research.activation_transfer.atoms.vectors import bridge, choose_atoms, coactive
from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import clean_forecast, derangements, read_materials
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/tsfm_electricity_atoms_20261120")
DATA = ELECTRICITY_ARROW


def prepare(cache, out):
    with np.load(ORIGINAL/"data.npz") as f:
        mask=f["split"]=="electricity"
        excluded={s.split("/",1)[1] for s in f["ids"][mask]}
        seen={sha256(x.tobytes()).digest() for x in f["x"]}
    ids,x=contexts(DATA, 160, 2026112082, seen, excluded)
    ids=np.array(["electricity/"+i for i in ids])
    split=np.array(["screen"]*64+["test"]*96)
    with DATA.open("rb") as reader:
        digest=file_digest(reader,"sha256").hexdigest()
    np.savez_compressed(cache/"data.npz",x=x,ids=ids,split=split)
    (out/"electricity_manifest.json").write_text(json.dumps(dict(sha256=digest,
         x_sha256=sha256(x.tobytes()).hexdigest(),split=dict(zip(ids,split))),indent=2)+"\n")


def run(source,cache,out,batch):
    target = next(m for m in MODELS if m!=source)
    with np.load(cache/"data.npz") as f:
        x,ids,split=f["x"],f["ids"],f["split"]
    screen, test = split=="screen",split=="test"
    with np.load(ORIGINAL/"data.npz") as f:
        tr = f["split"]=="train"
        mi = f["ids"][tr]
    states,tokens,positions,codes,dec,original = {},{},{},{},{},{}
    for model in [source,target]:
        tokens[model],positions[model],si=read_materials(cache,model)
        np.testing.assert_array_equal(si,ids)
        states[model]=tokens[model].mean(axis=1)
        sae=load_sae_checkpoint(str(dictionary_path(model)))
        with torch.inference_mode():
            codes[model]=sae.encode(torch.from_numpy(states[model])).numpy()
            dec[model]=sae.W_dec.numpy().copy()
        original_tokens,_,oi=read_materials(ORIGINAL,model)
        np.testing.assert_array_equal(oi[tr],mi)
        original[model]=original_tokens[tr].mean(axis=1)
    selected=choose_atoms(codes[source][screen],16,2026112083)
    source_dirs=dec[source][selected]
    z=codes[source][test][:,selected]
    active=z>0
    arrays=dict(ids=ids[test],atoms=selected,codes=z,active=active,
                screen_counts=np.sum(codes[source][screen][:,selected]>0,axis=0))
    a=adapter_for(source)
    layer=MODELS[source][-1]
    select=last_window_slice(a,positions[source])
    src_tokens=tokens[source][test]
    base=clean_forecast(a,layer,x[test],src_tokens,select,batch)
    arrays["native"]=run_arm(a,layer,select,x[test],src_tokens,base,active,z,source_dirs,batch)
    del a
    a=adapter_for(target)
    layer=MODELS[target][-1]
    select=last_window_slice(a,positions[target])
    t=tokens[target][test]
    base=clean_forecast(a,layer,x[test],t,select,batch)
    subset=np.random.default_rng(2026112084).permutation(len(mi))[:64]
    pairs={"m4_full": (original[source],original[target]), "m4_small":(original[source][subset],original[target][subset]),
           "electric":(states[source][screen],states[target][screen])}
    for name,(xx,yy) in pairs.items():
        dirs,_,_=bridge(xx,yy,source_dirs,2026112018)
        norm=np.linalg.norm(dirs,axis=1,keepdims=True)
        arrays[name]=run_arm(a,layer,select,x[test],t,base,active,z,dirs,batch)
        for j, perm in enumerate(derangements(len(selected))):
            wrong=dirs[perm]*norm/np.maximum(norm[perm],1e-14)
            arrays[name+"_wrong"+str(j)]=run_arm(a,layer,select,x[test],t,base,active,z,wrong,batch)
        print(source,name,"done",flush=True)
    eligible=np.flatnonzero((codes[target][screen]>0).mean(axis=0)>=.01)
    match,r=coactive(codes[source][screen][:,selected],codes[target][screen],eligible)
    arrays["coactive"]=run_arm(a,layer,select,x[test],t,base,active,codes[target][test][:,match],dec[target][match],batch)
    arrays.update(matches=match,target_match_codes=codes[target][test][:,match],matching_correlations=r,m4_subset_ids=mi[subset])
    np.savez_compressed(out/("electricity_"+source+".npz"),**arrays)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["data","capture","run"])
    p.add_argument("model",nargs="?",choices=MODELS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    a.cache.mkdir(exist_ok=True,parents=True)
    a.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(3)
    if a.stage=="data":
        prepare(a.cache,a.out)
    elif a.stage=="capture":
        capture(a.model,a.cache)
    else:
        run(a.model,a.cache,a.out,a.batch)
