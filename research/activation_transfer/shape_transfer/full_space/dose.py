"""Separate clipping from local approximation error for natural atom removals."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.transfer import HERE, SCRATCH, apply, boost_shape, solve_adaptive
from tsfm_lens.sae.train import load_sae_checkpoint


def run(source,cache,scratch,out,batch):
    target = next(m for m in MODELS if m != source)
    with np.load(cache/"data.npz") as f:
        keep = np.isin(f["split"],["daily","machines"])
        x,ids = f["x"][keep],f["ids"][keep]
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as f:
        ev = f["split"]!="cal"
        np.testing.assert_array_equal(ids,f["ids"][ev])
        codes, atoms, split = f["source_codes"][ev],f["atoms"],f["split"][ev]
    with np.load(scratch/(source+"_jac.npz")) as s, np.load(scratch/(target+"_jac.npz")) as t:
        np.testing.assert_array_equal(ids,s["ids"][ev])
        np.testing.assert_array_equal(s["ids"],t["ids"])
        js,jt = s["jac"][ev].astype(float),t["jac"][ev].astype(float)
        bases = {source:s["forecast"][ev],target:t["forecast"][ev]}
        radius = float(t["radius"])
    with np.load(out/(source+".npz")) as original:
        np.testing.assert_array_equal(ids,original["ids"])
        weight = float(original["shape_weight"])
        clipped = original["adaptive_clipped"]
    directions = load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float).T
    predicted = js @ directions
    write = solve_adaptive(boost_shape(jt,weight),boost_shape(predicted,weight))
    norms = codes*np.linalg.norm(write,axis=1)
    scale = np.minimum(1., .2*radius/np.maximum(norms,1e-15))
    np.testing.assert_array_equal(clipped,scale<1.)
    results = dict(ids=ids,split=split,atoms=atoms,active=codes>0,raw_fraction=norms/radius,
                   linear_target=(-codes[:,None,:]*(jt @ write)*scale[:,None,:]).transpose(0,2,1).astype("f4"))
    for name in (source,target):
        adapter = adapter_for(name)
        tok,pos,ident = read_materials(cache,name)
        np.testing.assert_array_equal(ids,ident[keep])
        args = [adapter,MODELS[name][-1],last_window_slice(adapter,pos),x,tok[keep],bases[name]]
        if name == source:
            results["source_quarter"],_ = apply(*args,.25*codes,directions,np.inf,batch)
        else:
            # Dividing both dose and cap leaves every relative attenuation factor intact.
            results["quarter"],quarter_clipped = apply(*args,.25*codes,write,radius,batch,max_fraction=.05)
            np.testing.assert_array_equal(quarter_clipped,clipped)
            results["larger_cap"],results["larger_cap_clipped"] = apply(*args,codes,write,radius,batch,max_fraction=.4)
    np.savez_compressed(out/(source+"_dose.npz"),**results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=32)
    args=p.parse_args()
    torch.set_num_threads(2)
    run(args.source,args.cache,args.scratch,args.out,args.batch)
