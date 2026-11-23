"""Choose a generic derivative memory lookup using only training and calibration derivatives."""
import argparse
from hashlib import file_digest,sha256
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import torch

from research.activation_transfer.atoms.materials import dictionary_path,MODELS
from research.activation_transfer.replication.data import CACHE as OLD,HERE as REP
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.depth.geometry import histories,FRESH
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as HISTORICAL
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.state_predictor.memory import CACHE,HERE
from tsfm_lens.sae.train import load_sae_checkpoint

KEYS=("raw","dense","amplitude","support")
K_VALUES=(1,4,16,64,256)


def rank_neighbours(train,query,sparse=False):
    """Return training indices with stable ties; binary values yield symmetric-difference count."""
    t=np.asarray(train,dtype=float);q=np.asarray(query,dtype=float)
    if sparse:
        t,q=csr_matrix(t),csr_matrix(q)
        cross=(q@t.T).toarray()
        tnorm=np.asarray(t.multiply(t).sum(1)).ravel()
        qnorm=np.asarray(q.multiply(q).sum(1)).ravel()
    else:
        cross=q@t.T
        tnorm=np.einsum("ij,ij->i",t,t)
        qnorm=np.einsum("ij,ij->i",q,q)
    d2=np.maximum(0,qnorm[:,None]+tnorm[None,:]-2*cross)
    return np.argsort(d2,axis=1,kind="stable")[:,:max(K_VALUES)],d2


def choice(gram,truth_dot,truth_energy,neighbors,grid=K_VALUES):
    """Select the generic-error minimum using only centered training/calibration Gram products."""
    best=None
    for k in grid:
        idx=neighbors[:,:k]
        correlation=np.mean(np.take_along_axis(truth_dot,idx,axis=1))
        variance=np.mean(gram[idx[:,:,None],idx[:,None,:]])
        mix=0. if variance<=1e-14*truth_energy else float(np.clip(correlation/variance,0,1))
        loss=float(truth_energy-2*mix*correlation+mix*mix*variance)
        item=(loss,k,mix)
        if best is None or item[0]<best[0]:best=item
    return dict(k=int(best[1]),mix=best[2],relative_generic_cal_loss=best[0]/truth_energy)


def response(memory,neighbors,settings):
    """Return the selected neighbor average for arbitrary RHS, with analytic global shrinkage."""
    avg=memory.mean(0)
    result=np.broadcast_to(avg,(len(neighbors),*avg.shape)).copy()
    if settings["mix"]==0:return result
    k=settings["k"]
    for start in range(0,len(neighbors),32):
        idx=neighbors[start:start+32,:k]
        result[start:start+len(idx)]+=settings["mix"]*(memory[idx].mean(1)-avg)
    return result


def gram_statistics(model,train,cal,weight):
    # Full-width calibration Frobenius loss uses all hidden channels, independent of donor atom IDs.
    path=CACHE/(model+"_generic_gram.npz")
    signature=sha256()
    for array in (train,cal):
        signature.update(np.ascontiguousarray(array).view(np.uint8))
    digest=signature.hexdigest()
    if path.exists():
        with np.load(path) as f:
            if "jacobian_digest" in f and str(f["jacobian_digest"])==digest and float(f["weight"])==weight:
                return f["gram"],f["truth_dot"],float(f["truth_energy"])
    y=boost_shape(train.astype(float),weight).reshape(len(train),-1)
    mu=y.mean(0); y-=mu
    true=boost_shape(cal.astype(float),weight).reshape(len(cal),-1)-mu
    gram=y@y.T
    truth_dot=true@y.T
    truth_energy=np.mean(np.einsum("ij,ij->i",true,true))
    np.savez(path,gram=gram,truth_dot=truth_dot,truth_energy=truth_energy,weight=weight,jacobian_digest=digest)
    return gram,truth_dot,float(truth_energy)


def key_material(model,seed):
    ids,x,labels=histories()
    h,pos,old_ids=read_materials(OLD,model)
    new,other,new_ids=read_materials(FRESH,model)
    np.testing.assert_array_equal(pos,other);np.testing.assert_array_equal(ids,np.r_[old_ids,new_ids])
    dense=np.concatenate((h.mean(1),new.mean(1)))
    ckpt=dictionary_path(model) if seed==0 else Path("/tmp/tsfm_boundary_20261120")/f"seed{seed}_{model}.pt"
    with ckpt.open("rb") as f: digest=file_digest(f,"sha256").hexdigest()
    with np.load(REP/"results"/f"{model}_seed{seed}.npz") as f:
        assert digest==str(f["source_checkpoint_hash"])
        atoms=f["atoms"]
        ref_codes=f["source_codes"]
        roi=f["ids"]
    sae=load_sae_checkpoint(str(ckpt))
    with torch.inference_mode():
        codes=np.concatenate([sae.encode(torch.from_numpy(z)).numpy() for z in np.array_split(dense,16)])
    prev=np.isin(labels,["cal","daily","machines"])
    np.testing.assert_array_equal(roi,ids[prev])
    np.testing.assert_allclose(ref_codes,codes[prev][:,atoms],atol=3e-4,rtol=1e-4)
    decoder=sae.W_dec.detach().numpy()[atoms].astype(float).T
    return ids,labels,dict(raw=x,dense=dense,amplitude=codes,support=codes>0),atoms,decoder,digest


def fit(model,seed):
    ids,labels,keys,atoms,D,digest=key_material(model,seed)
    train=labels=="train";cal=labels=="cal";ev=~(train|cal)
    assert (train.sum(),cal.sum(),ev.sum())==(1024,96,576)
    with np.load(CACHE/(model+"_train.npz")) as f:
        np.testing.assert_array_equal(ids[train],f["ids"])
        train_j=f["jac"];fd=float(f["fd_relative"])
    # Explicitly slice calibration indices. This fit never receives an evaluation derivative.
    with np.load(HISTORICAL/(model+"_jac.npz")) as f:
        sel=f["split"]=="cal"
        np.testing.assert_array_equal(ids[cal],f["ids"][sel]);cal_j=f["jac"][sel].astype(float)
    common=cal_j.mean(1,keepdims=True)
    weight=float(np.sqrt(np.mean(common**2)/np.mean((cal_j-common)**2)))
    gram,truth_dot,truth_energy=gram_statistics(model,train_j,cal_j,weight)
    train_projection=train_j.astype(float)@D
    record=dict(ids=ids[ev],labels=labels[ev],atoms=atoms,codes=keys["amplitude"][ev][:,atoms],
                decoder=D,train_ids=ids[train],cal_ids=ids[cal],weight=weight)
    record["mean"]=np.broadcast_to(train_projection.mean(0),(ev.sum(),8,len(atoms)))
    report=dict(source=model,seed=seed,checkpoint=digest,weight=weight,finite_difference_error=fd,
                memory_count=int(train.sum()),calibration_count=int(cal.sum()),fits={})
    perm=np.random.default_rng(20261121081).permutation(len(train_projection))
    for key in KEYS:
        val=keys[key]
        order,dist=rank_neighbours(val[train],val[~train],sparse=key in ("amplitude","support"))
        val_rows=cal[~train]
        audit=dict(median_train_nearest_squared_distance_by_domain={str(k):float(np.median(dist[labels[~train]==k].min(1))) for k in np.unique(labels[~train])})
        for sham in (False,True):
            arm=key+("_shuffled" if sham else "")
            idx=perm[order] if sham else order
            settings=choice(gram,truth_dot,truth_energy,idx[val_rows])
            record[arm]=response(train_projection,idx[~val_rows],settings)
            record[arm+"_neighbors"]=idx[~val_rows,:settings["k"]].astype("i2")
            report["fits"][arm]=dict(**settings,**audit)
    path=CACHE/f"{model}_seed{seed}_fit.npz"
    np.savez_compressed(path,**record)
    (HERE/"results"/f"{model}_seed{seed}_fit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(model,seed,[(k,v["k"],round(v["mix"],3)) for k,v in report["fits"].items()],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("model",choices=MODELS);p.add_argument("seed",type=int,choices=[0,1])
    args=p.parse_args();torch.set_num_threads(1);fit(args.model,args.seed)
