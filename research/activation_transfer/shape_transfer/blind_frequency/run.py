"""Fit controls on historical derivatives, then physically compare blind 16-lead continuations."""
import argparse
import gc

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as OLD_DATA, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import HERE,CACHE,CAL
from research.activation_transfer.shape_transfer.blind_frequency.geometry import blinded,permute_histories,within_model
from research.activation_transfer.shape_transfer.continuation.probe import ORIGINAL
from research.activation_transfer.shape_transfer.continuation.run import corrected
from research.activation_transfer.shape_transfer.continuation.probe import HERE as CONT
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape, solve_adaptive, solve_fixed
from tsfm_lens.sae.train import load_sae_checkpoint


def load_model_inputs(model,cohort,keep,ids):
    root=OLD_DATA if cohort=="old" else CACHE
    h,pos,item_ids=read_materials(root,model)
    np.testing.assert_array_equal(item_ids[keep],ids)
    with np.load(CAL/(model+"_jac16.npz")) as f:
        is_cal=f["split"]=="cal"
        assert is_cal.sum()==96
        calibration=f["jac"][is_cal].astype(float)
        cal_ids=f["ids"][is_cal]
        radius=float(f["radius"])
        if cohort=="old":
            ordinary=f["forecast"][~is_cal]
            np.testing.assert_array_equal(f["ids"][~is_cal],ids)
    if cohort=="old":
        with np.load(ORIGINAL/(model+"_jac.npz")) as f:
            idx=f["split"]!="cal"
            np.testing.assert_array_equal(f["ids"][idx],ids)
            jac=f["jac"][idx].astype(float)
    else:
        with np.load(CACHE/(model+"_early.npz")) as f:
            np.testing.assert_array_equal(f["ids"],ids)
            jac=f["jac"].astype(float)
            ordinary=f["forecast"]
    assert jac.shape[1]==8
    return h[keep],pos,jac,ordinary,calibration,cal_ids,radius


def run(source,cohort,batch=32):
    target=next(k for k in MODELS if k!=source)
    with np.load((OLD_DATA if cohort=="old" else CACHE)/"data.npz") as f:
        keep=np.isin(f["split"],["daily","machines","monash_traffic"])
        x,ids,split=f["x"][keep],f["ids"][keep],f["split"][keep]
    hs,ps,js,base_s,cs,cs_ids,_=load_model_inputs(source,cohort,keep,ids)
    ht,pt,jt,base_t,ct,ct_ids,radius=load_model_inputs(target,cohort,keep,ids)
    np.testing.assert_array_equal(cs_ids,ct_ids)
    assert set(cs_ids).isdisjoint(ids)
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as f:
        atoms=f["atoms"]
    sae=load_sae_checkpoint(str(dictionary_path(source)))
    decoder=sae.W_dec.detach().numpy()[atoms].astype(float).T
    with torch.inference_mode():
        codes=sae.encode(torch.from_numpy(hs.mean(axis=1))).numpy()[:,atoms]
    ev_early=js@decoder
    with np.load(CONT/"results"/(source+".npz")) as ref:
        weight=float(ref["weight"])
        if cohort=="old":
            np.testing.assert_array_equal(ids,ref["ids"])
            np.testing.assert_array_equal(codes>0,ref["active"])
            np.testing.assert_array_equal(atoms,ref["atoms"])
            ref_effect={k:ref[k] for k in ("source","minimum","paired_early")}
    record=dict(ids=ids,split=split,active=codes>0,codes=codes,atoms=atoms,weight=weight,radius=radius)
    if cohort=="fresh":
        src=adapter_for(source)
        sel=last_window_slice(src,ps)
        record["source"],_=apply(src,MODELS[source][-1],sel,x,hs,base_s,codes,decoder,np.inf,batch,forecast16)
        del src
        gc.collect()
    else:
        record["source"]=ref_effect["source"]

    def as_wave(late):
        return (-codes[:,:,None]*np.concatenate((ev_early,late),axis=1).transpose(0,2,1)).astype("f4")
    donor_late=within_model(cs,js)@decoder
    target_late=within_model(ct,jt)
    record["donor_regression"]=as_wave(donor_late)
    template=cs[:,8:].mean(0)@decoder
    record["donor_static"]=as_wave(np.broadcast_to(template,donor_late.shape))
    w_source=boost_shape(ev_early,weight)
    w_target=boost_shape(jt,weight)
    adapter=adapter_for(target)
    target_sel=last_window_slice(adapter,pt)
    def patch(key,direction):
        norms=np.linalg.norm(direction,axis=-2)*codes/radius
        record[key+"_norm"]=norms.astype("f4")
        if cohort=="old" and key in ("paired","minimum"):
            ref=ref_effect["paired_early" if key=="paired" else key]
            # Actual fresh subset replay; remaining old entries are the identical measured arm.
            idx=np.r_[0:8,192:200]
            fx,_=apply(adapter,MODELS[target][-1],target_sel,x[idx],ht[idx],base_t[idx],
                       codes[idx],direction[idx],radius,batch,forecast16)
            np.testing.assert_allclose(fx,ref[idx],atol=3e-4,rtol=.003)
            record[key+"_replay_max"]=np.max(np.abs(fx-ref[idx]))
            record[key]=ref
        else:
            record[key],_=apply(adapter,MODELS[target][-1],target_sel,x,ht,base_t,codes,
                                direction,radius,batch,forecast16)
        print(source,cohort,key,int((norms>.2).sum()),flush=True)
    patch("minimum",solve_adaptive(w_target,w_source))
    perm=permute_histories(len(cs))
    record["cal_derangement"]=perm
    for key,source_cal,target_cal in (("paired",cs,ct),("shuffled",cs[perm],ct),("blind_high",blinded(cs),blinded(ct))):
        prior=solve_fixed(boost_shape(target_cal,weight),boost_shape(source_cal@decoder,weight))
        patch(key,corrected(w_target,w_source,prior))
    target16=np.concatenate((jt,target_late),axis=1)
    donor16=np.concatenate((ev_early,donor_late),axis=1)
    patch("independent",solve_adaptive(boost_shape(target16,weight),boost_shape(donor16,weight)))
    np.savez_compressed(HERE/"results"/(source+"_"+cohort+".npz"),**record)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("cohort",choices=["old","fresh"])
    a=p.parse_args(); torch.set_num_threads(1);run(a.source,a.cohort)
