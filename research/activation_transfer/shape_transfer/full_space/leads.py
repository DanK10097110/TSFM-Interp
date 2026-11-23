"""Replay natural SAE removals and partially constrained target writes out to 16 leads."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.transfer import HERE, SCRATCH, apply, boost_shape, solve_adaptive
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint


def forecast16(adapter, layer, x, tokens, delta, select, batch):
    if len(x) > batch:
        return np.concatenate([forecast16(adapter,layer,x[i:i+batch],tokens[i:i+batch],
            None if delta is None else delta[i:i+batch],select,batch) for i in range(0,len(x),batch)])
    if delta is None:
        return adapter.predict(x,16,[.5])["quantiles"][...,0]
    with token_patch(adapter.module,layer,select,torch.from_numpy(tokens + delta[:,None])):
        return adapter.predict(x,16,[.5])["quantiles"][...,0]


def partial_writes(target_jac, source_response, source_calibration, leads):
    calibration = source_calibration[:,leads]
    mean = calibration.mean(axis=1,keepdims=True)
    weight = np.sqrt(np.mean(mean**2)/np.mean((calibration-mean)**2))
    return solve_adaptive(boost_shape(target_jac[:,leads],weight),boost_shape(source_response[:,leads],weight))


def run(source, cache, scratch, out, batch):
    target = next(m for m in MODELS if m != source)
    with np.load(cache/"data.npz") as f:
        keep = np.isin(f["split"],["daily","machines"])
        x, ids = f["x"][keep], f["ids"][keep]
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as f:
        ev = f["split"]!="cal"
        np.testing.assert_array_equal(ids,f["ids"][ev])
        codes, atoms, reference = f["source_codes"][ev], f["atoms"], f["native"][ev]
        split = f["split"][ev]
    with np.load(scratch/(source+"_jac.npz")) as s, np.load(scratch/(target+"_jac.npz")) as t:
        np.testing.assert_array_equal(s["ids"],t["ids"])
        np.testing.assert_array_equal(ids,s["ids"][ev])
        source_cal = s["jac"][~ev].astype(float)
        source_jac = s["jac"][ev].astype(float)
        target_jac,target_radius = t["jac"][ev].astype(float),float(t["radius"])
    directions = load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float).T
    jsu = source_jac @ directions
    linear = -codes[:,:,None]*jsu.transpose(0,2,1)
    record = dict(ids=ids,split=split,active=codes>0,atoms=atoms,
                  last8_tile=np.tile(linear,(1,1,2)).astype("f4"))
    # Interpolation is within each history and feature, using only the four disclosed source leads.
    odd = np.arange(0,8,2)
    record["odd_interpolation"] = np.stack([np.interp(np.arange(16),odd,row[odd])
        for row in linear.reshape(-1,8)]).reshape(len(ids),len(atoms),16).astype("f4")
    for name in (source,target):
        adapter = adapter_for(name)
        tokens,pos,item_ids = read_materials(cache,name)
        np.testing.assert_array_equal(item_ids[keep],ids)
        layer = MODELS[name][-1]
        tokens,select = tokens[keep],last_window_slice(adapter,pos)
        base = forecast16(adapter,layer,x,tokens,None,select,batch)
        if name==source:
            native,_ = apply(adapter,layer,select,x,tokens,base,codes,directions,np.inf,batch,forecast16)
            np.testing.assert_allclose(native[:,:,:8],reference,atol=3e-4,rtol=.003)
            record["source"] = native
            record["source_replay_max"] = np.max(np.abs(native[:,:,:8]-reference))
            continue
        for arm,leads in [("eight",np.arange(8)),("odd",odd)]:
            write = partial_writes(target_jac,jsu,source_cal,leads)
            record[arm],record[arm+"_clipped"] = apply(adapter,layer,select,x,tokens,base,codes,
                                                       write,target_radius,batch,forecast16)
            if arm=="eight":
                with np.load(out/(source+".npz")) as old:
                    np.testing.assert_array_equal(old["ids"],ids)
                    prev = old["adaptive"]
                np.testing.assert_allclose(record[arm][:,:,:8],prev,atol=3e-4,rtol=.003)
                record["target_replay_max"] = np.max(np.abs(record[arm][:,:,:8]-prev))
    np.savez_compressed(out/(source+"_leads.npz"),**record)


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
