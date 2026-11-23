"""Write each inferred donor signature using one frozen target library and score only later."""
import argparse
import gc

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as OLD,HERE as REP
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.depth.geometry import FRESH,HERE as DEPTH,histories
from research.activation_transfer.shape_transfer.fresh_cohort.data import OUT as FRESH_OUT
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL,SCRATCH
from research.activation_transfer.shape_transfer.full_space.transfer import apply
from research.activation_transfer.shape_transfer.state_predictor.fit import KEYS
from research.activation_transfer.shape_transfer.state_predictor.memory import CACHE,HERE


def material(model,ids,eval_rows):
    h,p,i=read_materials(OLD,model);new,other,j=read_materials(FRESH,model)
    np.testing.assert_array_equal(p,other);np.testing.assert_array_equal(np.r_[i,j],ids)
    with np.load(SCRATCH/(model+"_jac.npz")) as a,np.load(FRESH/(model+"_geometry.npz")) as b:
        k=a["split"]!="cal";np.testing.assert_array_equal(np.r_[a["ids"][k],b["ids"]],ids[eval_rows])
        base=np.concatenate((a["forecast"][k],b["forecast"]))
    return np.concatenate((h,new))[eval_rows],p,base


def run(source,seed,batch=32):
    target=next(k for k in MODELS if k!=source)
    ids,x,labels=histories()
    ev=~np.isin(labels,["train","cal"]);eval_x=x[ev];domain=labels[ev]
    key_order=[k for t in KEYS for k in (t,t+"_shuffled")]
    with np.load(CACHE/f"{source}_seed{seed}_fit.npz") as f:
        np.testing.assert_array_equal(ids[ev],f["ids"])
        codes,atoms,dec,weight=f["codes"],f["atoms"],f["decoder"],float(f["weight"])
        predictions={k:f[k].astype(float) for k in ("mean",*key_order)}
        calibration=f["cal_ids"]
    # This is the only use of current donor derivatives. They supply an oracle arm and scoring diagnostic.
    with np.load(SCRATCH/(source+"_jac.npz")) as a,np.load(FRESH/(source+"_geometry.npz")) as b:
        use=a["split"]!="cal"
        np.testing.assert_array_equal(np.r_[a["ids"][use],b["ids"]],ids[ev])
        predictions["current"]=np.concatenate((a["jac"][use],b["jac"])).astype(float)@dec
    with np.load(FULL/"results"/(source+"_access.npz")) as f:
        library=f["receiver_library"].astype(float);radius=float(f["radius"])
        np.testing.assert_allclose(weight,f["shape_weight"])
        np.testing.assert_array_equal(calibration,f["calibration_ids"])
        assert str(f["receiver"])==target and str(f["layer"])==MODELS[target][-1]
    results=dict(ids=ids[ev],split=domain,source=source,target=target,atoms=atoms,codes=codes,
                 active=codes>0,weight=weight,receiver_radius=radius,reuse=np.array([],dtype="U40"))
    old=np.isin(domain,["daily","machines"])
    with np.load(REP/"results"/f"{source}_seed{seed}.npz") as orig:
        m=orig["split"]!="cal";np.testing.assert_array_equal(ids[ev][old],orig["ids"][m])
        prior=orig["native"][m]
    hs,ps,base_s=material(source,ids,ev)
    if seed==0:
        with np.load(FRESH_OUT/(source+"_native.npz")) as fresh:
            np.testing.assert_array_equal(ids[ev][~old],fresh["ids"])
            np.testing.assert_array_equal(atoms,fresh["atoms"])
            new_native=fresh["native"]
    else:
        src=adapter_for(source)
        new_native,_=apply(src,MODELS[source][-1],last_window_slice(src,ps),eval_x[~old],hs[~old],base_s[~old],
                           codes[~old],dec,np.inf,batch)
        del src;gc.collect()
    native=np.zeros((*codes.shape,8),dtype="f4")
    native[old],native[~old]=prior,new_native
    results["native"]=native
    ht,pt,base_t=material(target,ids,ev)
    tgt=adapter_for(target);sel=last_window_slice(tgt,pt); reused=[]
    for key in ("mean","current",*key_order):
        val=predictions[key]
        results["forecast_"+key]=(-codes[:,:,None]*val.transpose(0,2,1)).astype("f4")
        if key!="mean" and np.array_equal(val,predictions["mean"]):
            for prefix in ("effect_","cap_","fraction_"): results[prefix+key]=results[prefix+"mean"]
            reused.append(key+"=mean")
            continue
        directions=library@val
        effect,cap=apply(tgt,MODELS[target][-1],sel,eval_x,ht,base_t,codes,directions,radius,batch)
        fraction=np.linalg.norm(directions,axis=1)*codes/radius
        np.testing.assert_array_equal(cap,fraction>.2)
        results["effect_"+key]=effect;results["cap_"+key]=cap;results["fraction_"+key]=fraction.astype("f4")
        print(source,seed,key,"active",int((codes>0).sum()),"caps",int(cap.sum()),flush=True)
    results["reuse"]=np.array(reused,dtype="U40")
    if seed==0:
        middle=3 if target=="bolt" else 10
        with np.load(DEPTH/"results"/f"{target}_{middle}.npz") as old:
            np.testing.assert_array_equal(ids[ev],old["ids"])
            diff=np.max(np.abs(old["fixed_20"][:,:,:8]-results["effect_current"]))
            np.testing.assert_allclose(old["fixed_20"][:,:,:8],results["effect_current"],atol=4e-4,rtol=.003)
        results["max_archived_oracle_replay"]=diff
    np.savez_compressed(HERE/"results"/f"{source}_seed{seed}.npz",**results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=MODELS)
    p.add_argument("seed",type=int,choices=[0,1]);a=p.parse_args();torch.set_num_threads(1);run(a.source,a.seed)
