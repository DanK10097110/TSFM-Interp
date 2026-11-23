"""Allocate equal weight to all kth-distance ties, CAL-tune, and physically replay the fixed library."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.score import scores
from research.activation_transfer.shape_transfer.depth.geometry import histories
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL
from research.activation_transfer.shape_transfer.full_space.transfer import apply
from research.activation_transfer.shape_transfer.state_predictor.fit import K_VALUES,key_material,rank_neighbours
from research.activation_transfer.shape_transfer.state_predictor.intervene import material
from research.activation_transfer.shape_transfer.state_predictor.memory import HERE,CACHE
from research.activation_transfer.shape_transfer.state_predictor.score import settings_interval


def fractional(dist,k):
    threshold=np.partition(dist,k-1,axis=1)[:,k-1,None]
    near=dist<threshold
    at=dist==threshold
    remainder=k-near.sum(1)
    weight=(near+at*(remainder/at.sum(1))[:,None])/k
    np.testing.assert_allclose(weight.sum(1),1,atol=3e-14)
    return weight


def tune(dist,gram,cross,energy):
    best=None
    for k in K_VALUES:
        w=fractional(dist,k)
        corr=np.mean(np.einsum("nt,nt->n",cross,w))
        var=np.mean(np.einsum("nt,nt->n",w@gram,w))
        mix=0. if var<1e-14*energy else float(np.clip(corr/var,0,1))
        rel=float((energy-2*mix*corr+mix*mix*var)/energy)
        if best is None or rel<best[0]:best=(rel,k,mix)
    return best


def estimate(dist,projected,k,mix):
    w=fractional(dist,k)
    avg=projected.mean(0)
    near=(w@projected.reshape(len(projected),-1)).reshape(len(w),*avg.shape)
    return avg+mix*(near-avg),w


def evaluate(ref,extra):
    output={}
    z=ref["codes"]
    for domain in np.unique(ref["split"]):
        use=ref["split"]==domain;keep=ref["active"][use].sum(0)>=8
        mask=ref["active"][use][:,keep]
        def select(a):return a[use][:,keep].astype(float)
        n=int(keep.sum())
        if n==0:
            output[domain]=dict(n=len(mask),atoms=[]);continue
        hist=resampling_counts(len(mask),4000)
        draws=np.random.default_rng(20261121084).multinomial(n,np.full(n,1/n),size=len(hist))
        rd=dict(n=len(mask),atoms=ref["atoms"][keep].tolist(),n_active=int(mask.sum()),
                stable_caps=int(select(ref["cap_support"])[mask].sum()),fair_caps=int(select(extra["cap"])[mask].sum()))
        truth=select(ref["native"])
        for measure,prefix in (("physical","effect"),("forecast","forecast")):
            for scale in ("centered","full"):
                get=lambda a:a-a.mean(-1,keepdims=True) if scale=="centered" else a
                y=get(truth)
                raw={}
                for k,p in (("stable",ref[prefix+"_support"]),("fair",extra[prefix]),("mean",ref[prefix+"_mean"])):
                    raw[k]=scores(get(select(p)),y,mask,hist,draws)
                out={k:settings_interval(*v[:3]) for k,v in raw.items()}
                for ctrl in ("stable","mean"):
                    point,ss,ff=[raw["fair"][j]-raw[ctrl][j] for j in range(3)]
                    stat=settings_interval(point,ss,ff)
                    if ctrl=="stable" and measure=="physical" and scale=="centered" and domain!="machines":
                        stat["primary_lower"]=float(np.nanquantile(ff,.05/6))
                    out["fair_minus_"+ctrl]=stat
                rd[measure+"_"+scale]=out
        output[domain]=rd
    return output


def run(source):
    target=next(n for n in MODELS if n!=source)
    ids,labels,features,atoms,decoder,_=key_material(source,0)
    train=labels=="train";cal=labels=="cal";ev=~(train|cal)
    _,x,lab=histories();np.testing.assert_array_equal(lab,labels)
    with np.load(CACHE/(source+"_train.npz")) as f:
        np.testing.assert_array_equal(ids[train],f["ids"]);projected=f["jac"].astype(float)@decoder
    with np.load(CACHE/(source+"_generic_gram.npz")) as f:
        gram=f["gram"];cross=f["truth_dot"];energy=float(f["truth_energy"])
    q=features["support"]
    _,d=rank_neighbours(q[train],q[~train],sparse=True)
    iscal=cal[~train]
    loss,k,mix=tune(d[iscal],gram,cross,energy)
    y,w=estimate(d[~iscal],projected,k,mix)
    perm=np.random.default_rng(20261121087).permutation(train.sum())
    alt=tune(d[iscal][:,perm],gram[perm][:,perm],cross[:,perm],energy)
    np.testing.assert_allclose(alt,[loss,k,mix],atol=3e-13,rtol=3e-13)
    after,wo=estimate(d[~iscal][:,perm],projected[perm],alt[1],alt[2])
    weight_error=float(np.max(np.abs(w-wo[:,np.argsort(perm)])))
    pred_error=float(np.max(np.abs(after-y)))
    if weight_error or pred_error>1e-11:raise ValueError("Joint row permutation changed tie-fair predictions")
    with np.load(HERE/"results"/(source+"_seed0.npz")) as f:ref=dict(f)
    np.testing.assert_array_equal(ref["atoms"],atoms);np.testing.assert_array_equal(ref["ids"],ids[ev])
    with np.load(FULL/"results"/(source+"_access.npz")) as f:lib=f["receiver_library"];radius=float(f["radius"])
    ht,pt,base=material(target,ids,ev)
    enc=adapter_for(target)
    z=ref["codes"]
    effect,cap=apply(enc,MODELS[target][-1],last_window_slice(enc,pt),x[ev],ht,base,z,lib@y,radius,32)
    extra=dict(ids=ids[ev],atoms=atoms,cap=cap,effect=effect,forecast=(-z[:,:,None]*y.transpose(0,2,1)).astype("f4"))
    np.savez_compressed(HERE/"results"/(source+"_ties.npz"),**extra)
    info=json.loads((HERE/"results"/(source+"_seed0_fit.json")).read_text())["fits"]["support"]
    report=dict(fair=dict(k=k,mix=mix,relative_generic_cal_loss=loss),stable={v:info[v] for v in ("k","mix","relative_generic_cal_loss")},
                max_permutation_weight_error=weight_error,max_permutation_unit_prediction_error=pred_error,
                domains=evaluate(ref,extra))
    (HERE/"results"/(source+"_ties.json")).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(source,"fair",k,mix,loss,"order error",pred_error,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=MODELS)
    a=p.parse_args();torch.set_num_threads(1);run(a.source)
