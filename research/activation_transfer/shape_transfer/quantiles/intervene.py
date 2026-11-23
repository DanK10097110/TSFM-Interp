"""Translate donor median, interval half-width and asymmetry, then score actual forecasts."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.fresh_cohort.data import OUT as FRESH_RESULTS
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL, apply, boost_shape,solve_adaptive,solve_fixed
from research.activation_transfer.shape_transfer.quantiles.geometry import HERE,SCRATCH,components,material,ordinary_or_patch
from research.activation_transfer.replication.data import HERE as REPLICATION


def channels(jac):
    """[series, lead, quantile, hidden] -> [series, component, lead, hidden]."""
    return np.moveaxis(components(np.moveaxis(jac,2,-1)),-1,1)


def weights(source_cal):
    dc=source_cal.mean(axis=2,keepdims=True)
    def rms(a):
        return np.sqrt(np.mean(a*a,axis=(0,2,3)))
    shape_factor=rms(dc)/rms(source_cal-dc)
    balanced=dc+shape_factor[None,:,None,None]*(source_cal-dc)
    sizes=rms(balanced)
    if np.any(~np.isfinite(shape_factor)) or np.min(sizes)<=0:
        raise ValueError("Degenerate calibration channel")
    return shape_factor,sizes[0]/sizes


def stack(values,balance,gains):
    mean=values.mean(axis=2,keepdims=True)
    result=gains[None,:,None,None]*(mean+balance[None,:,None,None]*(values-mean))
    return result.reshape(len(values),-1,values.shape[-1])


def run(source,scratch,out,batch):
    target=next(n for n in MODELS if n!=source)
    ids,codes,atoms,dec,_,old_weight=source_material(source)
    with np.load(scratch/(source+".npz")) as s,np.load(scratch/(target+".npz")) as t:
        np.testing.assert_array_equal(ids,s["ids"])
        np.testing.assert_array_equal(ids,t["ids"])
        split=s["split"]
        ev=split!="cal"
        source_j=channels(s["jac"].astype(float))
        target_j=channels(t["jac"].astype(float))
        balance,gain=weights(source_j[~ev])
        template=source_j[ev] @ dec
        radii={source:float(s["radius"]),target:float(t["radius"])}
        bases={source:s["base"][ev],target:t["base"][ev]}
        audit={source:s["fd_error"].tolist(),target:t["fd_error"].tolist()}
    np.testing.assert_allclose(balance[0],old_weight,rtol=2e-4)
    demand=stack(template,balance,gain)
    receiver=stack(target_j[ev],balance,gain)
    cal=stack(target_j[~ev],balance,gain)
    inverse=solve_fixed(cal,np.broadcast_to(np.eye(24),(len(cal),24,24)))
    record=dict(ids=ids[ev],split=split[ev],atoms=atoms,active=codes[ev]>0,balance=balance,gains=gain,
                donor_baseline=bases[source],target_baseline=bases[target],donor_fd=audit[source],target_fd=audit[target],
                library=inverse,source_linear=(-codes[ev,:,None,None]*template.transpose(0,3,2,1)).astype("f4"))
    for name in (source,target):
        enc=adapter_for(name)
        layer=MODELS[name][-1]
        full_ids,x,labels,h,pos=material(name)
        keep=~np.isin(labels,["train","cal"])
        np.testing.assert_array_equal(full_ids[keep],ids[ev])
        select=last_window_slice(enc,pos)
        args=(enc,layer,select,x[keep],h[keep],bases[name].reshape(sum(ev),24),codes[ev])
        if name==source:
            y,_=apply(*args,dec,np.inf,batch,ordinary_or_patch)
            record["source"]=y.reshape(len(y),len(atoms),8,3)
            continue
        med=solve_adaptive(boost_shape(target_j[ev,0],balance[0]),boost_shape(template[:,0],balance[0]))
        alt=target_j[ev].copy()
        alt[:,1]*=-1
        for arm,w in [("median_only",med),("joint_local",solve_adaptive(receiver,demand)),
                      ("joint_fixed",inverse@demand),("wrong_width",solve_adaptive(stack(alt,balance,gain),demand))]:
            y,record[arm+"_clipped"]=apply(*args,w,radii[target],batch,ordinary_or_patch)
            record[arm]=y.reshape(len(y),len(atoms),8,3)
            record[arm+"_raw_fraction"]=codes[ev]*np.linalg.norm(w,axis=1)/radii[target]
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as old,np.load(FRESH_RESULTS/(source+"_native.npz")) as new:
        actual=np.concatenate([old["native"][old["split"]!="cal"],new["native"]])
    with np.load(FULL/"results"/(source+".npz")) as old,np.load(FRESH_RESULTS/(source+"_writes.npz")) as new:
        np.testing.assert_array_equal(ids[ev],np.r_[old["ids"],new["ids"]])
        previous=np.concatenate([old["adaptive"],new["adaptive"]])
    np.testing.assert_allclose(record["source"][:,:,:,1],actual,atol=4e-4,rtol=.004)
    np.testing.assert_allclose(record["median_only"][:,:,:,1],previous,atol=4e-4,rtol=.004)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(source+".npz"),**record)
    print(source,"finished",audit,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=24)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.source,a.scratch,a.out,a.batch)
