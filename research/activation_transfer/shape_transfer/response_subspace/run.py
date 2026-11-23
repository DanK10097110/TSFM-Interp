"""Physically compare a real SAE removal with projected and equal-norm surrogate removals."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL,SCRATCH as CAL_JAC,apply,boost_shape
from research.activation_transfer.shape_transfer.quantiles.geometry import LEVELS,material
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
FIT=Path("/tmp/tsfm_resp_subspace_20261121")
CHECKPOINTS=Path("/tmp/tsfm_boundary_20261120")


def response_basis(jac,weight):
    mean=jac.mean(axis=0)
    singular=np.linalg.svd(mean,compute_uv=False)
    if singular[-1]<singular[0]*1e-8:
        raise ValueError("Calibration mean does not resolve eight directions")
    common=np.linalg.svd(mean,full_matrices=False)[2]
    stack=boost_shape(jac,weight).reshape(-1,jac.shape[-1])
    residual=stack-(stack@common.T)@common
    extra=np.linalg.svd(residual,full_matrices=False)[2][:88]
    basis=np.concatenate([common,extra])
    np.testing.assert_allclose(basis@basis.T,np.eye(96),atol=1e-10)
    return basis


def rotated_remainder(directions,basis,rng):
    """Apply the same orthogonal transformation to all directions in the complement."""
    parallel=(directions@basis.T)@basis
    complement=null_space(basis)
    coordinates=directions @ complement
    perm=rng.permutation(complement.shape[1])
    signs=rng.choice([-1.,1.],size=len(perm))
    return parallel+(coordinates[:,perm]*signs)@complement.T


def fit(name,out):
    ids,_,labels,tokens,_=material(name)
    train=labels=="train"
    states=tokens[train].mean(axis=1).astype(float)
    with np.load(CAL_JAC/(name+"_jac.npz")) as f:
        cal=f["split"]=="cal"
        np.testing.assert_array_equal(f["ids"][cal],ids[labels=="cal"])
        jac=f["jac"][cal].astype(float)
    with np.load(FULL/"results"/(name+".npz")) as f:
        weight=float(f["shape_weight"])
    response=response_basis(jac,weight)
    pca=np.linalg.svd(states-states.mean(axis=0),full_matrices=False)[2][:32]
    random=np.linalg.qr(np.random.default_rng(119451).normal(size=(states.shape[1],32)))[0].T
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(name+".npz"),response=response,pca=pca,random=random,mean_jac=jac.mean(axis=0),
                        fit_ids=ids[labels=="cal"],train_ids=ids[train])


def predict_all(enc,layer,x,tokens,delta,select,batch):
    result=[]
    for i in range(0,len(x),batch):
        obs=x[i:i+batch]
        if delta is None:
            y=enc.predict(obs,16,LEVELS)["quantiles"]
        else:
            t=torch.from_numpy(tokens[i:i+batch]+delta[i:i+batch,None,:])
            with token_patch(enc.module,layer,select,t):
                y=enc.predict(obs,16,LEVELS)["quantiles"]
        result.append(y.reshape(len(obs),48))
    return np.concatenate(result)


def run(name,seed,calibration,out,batch):
    all_ids,x,labels,tokens,pos=material(name)
    ev=~np.isin(labels,["cal","train"])
    ids,x,tokens=all_ids[ev],x[ev],tokens[ev]
    with np.load(REPLICATION/"results"/f"{name}_seed{seed}.npz") as f:
        keep=f["split"]!="cal"
        old_ids=f["ids"][keep]
        np.testing.assert_array_equal(old_ids,ids[:len(old_ids)])
        atoms=f["atoms"]
        old_values=f["source_codes"][keep]
        old_forecast=f["native"][keep]
    path=dictionary_path(name) if seed==0 else CHECKPOINTS/f"seed{seed}_{name}.pt"
    sae=load_sae_checkpoint(str(path))
    dirs=sae.W_dec.detach().numpy()[atoms].astype(float)
    with torch.inference_mode():
        codes=sae.encode(torch.from_numpy(tokens.mean(axis=1))).numpy()[:,atoms]
    np.testing.assert_allclose(codes[:len(old_ids)],old_values,rtol=2e-4,atol=2e-4)
    with np.load(calibration/(name+".npz")) as f:
        basis,pca,random,mean=f["response"],f["pca"],f["random"],f["mean_jac"]
    directions={"native":dirs,"pca32":(dirs@pca.T)@pca}
    if seed==0:
        directions["random32"]=(dirs@random.T)@random
    for rank in ((8,32,96) if seed==0 else (8,32)):
        sub=basis[:rank]
        projected=(dirs@sub.T)@sub
        rng=np.random.default_rng(37158+rank)
        foil=rotated_remainder(dirs,sub,rng)
        np.testing.assert_allclose(foil @ foil.T,dirs @ dirs.T,atol=2e-10)
        for variant in (projected,foil):
            np.testing.assert_allclose(variant @ mean.T,dirs @ mean.T,atol=1e-10)
        directions["project"+str(rank)]=projected
        directions["foil"+str(rank)]=foil
    adapter=adapter_for(name)
    layer=MODELS[name][-1]
    select=last_window_slice(adapter,pos)
    base=predict_all(adapter,layer,x,tokens,None,select,batch)
    record=dict(ids=ids,split=labels[ev],atoms=atoms,active=codes>0,codes=codes)
    for k,d in directions.items():
        y,_=apply(adapter,layer,select,x,tokens,base,codes,d.T,np.inf,batch,predict_all)
        record[k]=y.reshape(*codes.shape,16,3)
        record[k+"_norm_fraction"]=np.linalg.norm(d,axis=1)/np.linalg.norm(dirs,axis=1)
        record[k+"_hidden_cosine"]=np.sum(d*dirs,axis=1)/(np.linalg.norm(d,axis=1)*np.linalg.norm(dirs,axis=1))
        print(name,seed,k,"complete",flush=True)
    np.testing.assert_allclose(record["native"][:len(old_ids),:,:8,1],old_forecast,rtol=.004,atol=4e-4)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"{name}_seed{seed}.npz",**record)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("--seed",type=int,choices=[0,1,2])
    p.add_argument("--calibration",type=Path,default=FIT)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.seed is None:
        fit(a.model,a.calibration)
    else:
        run(a.model,a.seed,a.calibration,a.out,a.batch)
