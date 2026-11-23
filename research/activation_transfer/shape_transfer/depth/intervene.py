"""Run fixed, local and wrong-order target writes while the source SAE site stays fixed."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.depth.geometry import FRESH,HERE,SCRATCH,histories
from research.activation_transfer.shape_transfer.fresh_cohort.data import OUT as FRESH_RESULTS
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL, SCRATCH as OLD_JAC,apply,boost_shape,solve_fixed,solve_adaptive
from tsfm_lens.sae.train import load_sae_checkpoint


def source_material(source):
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as old,np.load(FRESH_RESULTS/(source+"_native.npz")) as new:
        np.testing.assert_array_equal(old["atoms"],new["atoms"])
        ids=np.r_[old["ids"],new["ids"]]
        codes=np.concatenate([old["source_codes"],new["codes"]])
        atoms=new["atoms"]
    with np.load(OLD_JAC/(source+"_jac.npz")) as old,np.load(FRESH/(source+"_geometry.npz")) as new:
        np.testing.assert_array_equal(ids,np.r_[old["ids"],new["ids"]])
        jac=np.concatenate([old["jac"],new["jac"]]).astype(float)
    dec=load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float).T
    with np.load(FULL/"results"/(source+".npz")) as record:
        weight=float(record["shape_weight"])
    return ids,codes,atoms,dec,jac @ dec,weight


def source16(source,out,batch):
    ids,codes,atoms,dec,_,_=source_material(source)
    with np.load(FRESH/"data.npz") as f:
        x,news=f["x"],f["ids"]
    h,pos,i=read_materials(FRESH,source)
    np.testing.assert_array_equal(i,news)
    np.testing.assert_array_equal(i,ids[-len(i):])
    adapter=adapter_for(source)
    layer=MODELS[source][-1]
    select=last_window_slice(adapter,pos)
    base=forecast16(adapter,layer,x,h,None,select,batch)
    y,_=apply(adapter,layer,select,x,h,base,codes[-len(i):],dec,np.inf,batch,forecast16)
    with np.load(FRESH_RESULTS/(source+"_native.npz")) as f:
        np.testing.assert_allclose(y[:,:,:8],f["native"],atol=3e-4,rtol=.003)
    with np.load(FULL/"results"/(source+"_leads.npz")) as f:
        all_ids=np.r_[f["ids"],news]
        effects=np.concatenate([f["source"],y])
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(source+"_reference.npz"),ids=all_ids,effects=effects,atoms=atoms)


def run(target,depth,scratch,out,batch):
    source=next(n for n in MODELS if n!=target)
    ids,codes,atoms,_,response,weight=source_material(source)
    all_ids,x,labels=histories()
    np.testing.assert_array_equal(all_ids[labels!="train"],ids)
    x=x[~np.isin(labels,["cal","train"])]
    with np.load(scratch/f"{target}_{depth}.npz") as f:
        np.testing.assert_array_equal(f["ids"],ids)
        split=f["split"]
        ev=split!="cal"
        jt=f["jac"].astype(float)
        h,pos,base,radius,layer=f["states"][ev],f["positions"],f["base"][ev],float(f["radius"]),str(f["layer"])
    adapter=adapter_for(target)
    select=last_window_slice(adapter,pos)
    identity=np.broadcast_to(np.eye(8),(int((~ev).sum()),8,8))
    library=solve_fixed(boost_shape(jt[~ev],weight),boost_shape(identity,weight))
    local=solve_adaptive(boost_shape(jt[ev],weight),boost_shape(response[ev],weight))
    wrong=solve_adaptive(boost_shape(jt[ev][:,[3,6,1,7,0,2,5,4]],weight),boost_shape(response[ev],weight))
    record=dict(ids=ids[ev],split=split[ev],active=codes[ev]>0,atoms=atoms,radius=radius,layer=layer,
                singular_values=np.linalg.svd(jt[ev],compute_uv=False),receiver_library=library)
    for arm,w in [("local",local),("fixed",library@response[ev]),("wrong",wrong)]:
        record[arm+"_raw_norm_fraction"]=codes[ev]*np.linalg.norm(w,axis=1)/radius
        for fraction in ((.20,) if arm=="wrong" else (.05,.20)):
            key=f"{arm}_{int(100*fraction)}"
            record[key],record[key+"_clipped"]=apply(adapter,layer,select,x,h,base,codes[ev],w,radius,
                                                    batch,forecast16,max_fraction=fraction)
    if layer==MODELS[target][-1]:
        with np.load(FULL/"results"/(source+"_leads.npz")) as old:
            np.testing.assert_array_equal(ids[ev][:len(old["ids"])],old["ids"])
            np.testing.assert_allclose(record["local_20"][:len(old["ids"])],old["eight"],atol=4e-4,rtol=.004)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"{target}_{depth}.npz",**record)
    print(target,depth,"done",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("--depth",type=int)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.depth is None:
        source16(a.model,a.out,a.batch)
    else:
        run(a.model,a.depth,a.scratch,a.out,a.batch)
