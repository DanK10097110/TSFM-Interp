"""Factor dependence on source versus receiving model derivatives at the current history."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.transfer import HERE, SCRATCH, apply, boost_shape, solve_fixed
from tsfm_lens.sae.train import load_sae_checkpoint


def run(source,cache,scratch,out,batch):
    target = next(n for n in MODELS if n!=source)
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as f:
        cal = f["split"]=="cal"
        ids,codes,atoms,labels = f["ids"],f["source_codes"][~cal],f["atoms"],f["split"][~cal]
    with np.load(scratch/(target+"_jac.npz")) as t,np.load(scratch/(source+"_jac.npz")) as s:
        np.testing.assert_array_equal(ids,t["ids"])
        np.testing.assert_array_equal(ids,s["ids"])
        # Receiver test derivatives are excluded from the reusable operator.
        tc,base,radius = t["jac"][cal].astype(float),t["forecast"][~cal],float(t["radius"])
        source_dirs = load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float).T
        response = s["jac"].astype(float) @ source_dirs
    with np.load(out/(source+".npz")) as old:
        np.testing.assert_array_equal(old["ids"],ids[~cal])
        weight = float(old["shape_weight"])
    identity = np.broadcast_to(np.eye(8),(int(cal.sum()),8,8))
    inverse = solve_fixed(boost_shape(tc,weight),boost_shape(identity,weight))
    adapter = adapter_for(target)
    tokens,pos,all_ids = read_materials(cache,target)
    with np.load(cache/"data.npz") as f:
        keep = np.isin(f["split"],["daily","machines"])
        x = f["x"][keep]
        np.testing.assert_array_equal(all_ids,f["ids"])
        np.testing.assert_array_equal(all_ids[keep],ids[~cal])
    args = [adapter,MODELS[target][-1],last_window_slice(adapter,pos),x,tokens[keep],base,codes]
    result = dict(ids=ids[~cal],split=labels,atoms=atoms,active=codes>0,receiver_library=inverse,
                  receiver=target,donor=source,layer=MODELS[target][-1],radius=radius,
                  calibration_ids=ids[cal],target_positions=pos,shape_weight=weight)
    for key,pred in [("current_source_fixed_target",response[~cal]),
                     ("mean_source_fixed_target",response[cal].mean(axis=0))]:
        result[key],result[key+"_clipped"] = apply(*args,inverse @ pred,radius,batch)
    np.savez_compressed(out/(source+"_access.npz"),**result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.source,a.cache,a.scratch,a.out,a.batch)
