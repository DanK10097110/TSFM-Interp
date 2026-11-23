"""Independent neighborhood/scale replay and fresh physical checks on sampled held-out histories."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from research.activation_transfer.atoms.materials import adapter_for,MODELS
from research.activation_transfer.shape_transfer.depth.geometry import histories
from research.activation_transfer.shape_transfer.full_space.audit import direct_scores,ordinary
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL,SCRATCH
from research.activation_transfer.shape_transfer.state_predictor.fit import key_material
from research.activation_transfer.shape_transfer.state_predictor.memory import CACHE,HERE

KEYS=("raw","dense","amplitude","support")


def one(model,seed):
    target=next(n for n in MODELS if n!=model)
    ids,labels,features,atoms,D,sha=key_material(model,seed)
    _,x,_=histories()
    train=labels=="train";cal=labels=="cal";test=~(train|cal)
    ref=ids[test]
    with np.load(HERE/"results"/f"{model}_seed{seed}.npz") as f:r=dict(f)
    with np.load(CACHE/f"{model}_seed{seed}_fit.npz") as f:fit=dict(f)
    metadata=json.loads((HERE/"results"/f"{model}_seed{seed}_fit.json").read_text())
    np.testing.assert_array_equal(ref,r["ids"])
    np.testing.assert_array_equal(atoms,r["atoms"])
    assert not(set(ids[train])&set(ids[cal]) or set(ids[train|cal])&set(ref))
    with np.load(CACHE/(model+"_train.npz")) as f:
        train_j=f["jac"].astype(float)
        np.testing.assert_array_equal(ids[train],f["ids"])
    with np.load(SCRATCH/(model+"_jac.npz")) as f:cal_j=f["jac"][f["split"]=="cal"].astype(float)
    with np.load(FULL/"results"/(model+"_access.npz")) as f:
        lib=f["receiver_library"];rad=float(f["radius"])
    w=metadata["weight"]
    I=np.eye(8);M=np.ones((8,8))/8;Q=M+w*(I-M)
    mean=train_j.mean(0)
    y=Q@(cal_j-mean);E=float(np.sum(y*y))
    pr=np.random.default_rng(20261121081).permutation(train.sum())
    idx=[]
    rng=np.random.default_rng(20261121086)
    for domain in ("daily","extra_monash_traffic","extra_daily","machines"):
        if len(idx)==2:break
        m=r["split"]==domain;cols=r["active"][m].sum(0)>=8
        cases=np.argwhere(m[:,None]&r["active"]&cols)
        if len(cases):idx.append(cases[rng.integers(len(cases))])
    rows=np.array([i for i,_ in idx],int)
    arms={};point_errors=[]
    for key in KEYS:
        val=features[key]
        q=np.concatenate((val[cal],val[test][rows]))
        # scipy computes squared Euclidean from explicit differences, independent of norm/dot-product expansion.
        dist=cdist(q.astype(float),val[train].astype(float),metric="sqeuclidean")
        order=np.argsort(dist,axis=1,kind="stable")
        for sham in (False,True):
            arm=key+("_shuffled" if sham else "")
            st=metadata["fits"][arm];k=st["k"]
            selected=(pr[order[:,:k]] if sham else order[:,:k])
            np.testing.assert_array_equal(selected[len(y):],fit[arm+"_neighbors"][rows])
            # Compute validation errors directly in the generic Jacobian, without using precomputed Gram products.
            local=np.stack([train_j[neighbors].mean(0) for neighbors in selected[:len(y)]])
            pred=Q@(local-mean)
            mix=float(np.clip(np.sum(y*pred)/np.sum(pred*pred),0,1))
            loss=float(np.sum((y-mix*pred)**2)/E)
            np.testing.assert_allclose([mix,loss],[st["mix"],st["relative_generic_cal_loss"]],rtol=3e-8,atol=3e-10)
            actual=np.stack([(mean+mix*(train_j[ind].mean(0)-mean))@D for ind in selected[len(y):]])
            mx=float(np.max(np.abs(actual-fit[arm][rows])))
            point_errors.append(mx)
            arms[arm]=dict(k=k,mix=mix,direct_validation_loss_relative=loss,max_unit_projection_error=mx)
    if max(point_errors)>2e-7:raise AssertionError("Neighbor readout replay mismatch")
    src,tgt=adapter_for(model),adapter_for(target)
    physical=[]
    for i,j in idx:
        h=x[test][i:i+1];z=r["codes"][i,j]
        b0=ordinary(src,model,h,None,8);t0=ordinary(tgt,target,h,None,8)
        donor=ordinary(src,model,h,-z*D[:,j],8)-b0
        max_native=float(np.max(np.abs(donor-r["native"][i,j])))
        if max_native>3e-5:raise ValueError("Independent source removal discrepancy")
        out={}
        for key in ("mean","dense","amplitude","support","current"):
            unit=-r["forecast_"+key][i,j].astype(float)/z
            delta=-z*(lib@unit)
            frac=np.linalg.norm(delta)/rad
            actual=ordinary(tgt,target,h,delta*min(1,.2/max(frac,1e-30)),8)-t0
            archived=r["effect_"+key][i,j]
            diff=float(np.max(np.abs(actual-archived)))
            if diff>3e-5:raise ValueError("Independent target intervention discrepancy")
            out[key]=dict(max_forecast_error=diff,raw_fraction=float(frac))
        physical.append(dict(id=str(ref[i]),domain=str(r["split"][i]),atom=int(atoms[j]),source_code=float(z),
                             max_native_loss_error=max_native,target_arms=out))
    published=json.loads((HERE/"results"/f"{model}_seed{seed}_scores.json").read_text())["domains"]
    max_score_error=0.
    for domain,rec in published.items():
        use=r["split"]==domain
        valid=r["active"][use].sum(0)>=8
        if not np.any(valid):continue
        mask=r["active"][use][:,valid]
        target_truth=r["native"][use][:,valid].astype(float)
        for key in ("mean","dense","amplitude","support","current"):
            for prefix,measure in (("effect_","physical"),("forecast_","forecast")):
                est=r[prefix+key][use][:,valid].astype(float)
                for centered in (False,True):
                    ref=target_truth-target_truth.mean(-1,keepdims=True) if centered else target_truth
                    pred=est-est.mean(-1,keepdims=True) if centered else est
                    test_score=direct_scores(ref,pred,mask)["skill"]
                    expected=rec[measure+("_centered" if centered else "_full")][key]["value"]
                    max_score_error=max(max_score_error,abs(test_score-expected))
    if max_score_error>2e-8:raise AssertionError("Population score replay mismatch")
    return dict(checkpoint=sha,all_ids_disjoint=True,max_population_point_error=max_score_error,source_width=D.shape[0],train_histories=int(train.sum()),
                validation_histories=int(cal.sum()),fits=arms,physical=physical)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=MODELS);p.add_argument("seed",type=int,choices=[0,1])
    a=p.parse_args();torch.set_num_threads(1)
    (HERE/"results"/f"{a.source}_seed{a.seed}_audit.json").write_text(json.dumps(one(a.source,a.seed),indent=2,allow_nan=False)+"\n")
