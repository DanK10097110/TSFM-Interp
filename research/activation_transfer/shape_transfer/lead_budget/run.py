"""Fix how eight measurements span a forecast and evaluate four common, undisclosed coordinates."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as REP,HERE as REFERENCE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.continuation.probe import SCRATCH as OLD16
from research.activation_transfer.shape_transfer.full_space.transfer import apply,boost_shape,solve_adaptive
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
TMP=Path("/tmp/tsfm_lead_budget_20261121")
OUT=HERE/"results"
AUDIT=np.array([8,10,12,14])
FIRST=np.arange(8)
SPREAD=np.array([0,2,4,6,9,11,13,15])


def forecast32(enc,layer,x,h,delta,sel,batch):
    values=[]
    for k in range(0,len(x),batch):
        if delta is None:
            y=enc.predict(x[k:k+batch],32,[.5])["quantiles"][...,0]
        else:
            edited=torch.from_numpy(h[k:k+batch]+delta[k:k+batch,None])
            with token_patch(enc.module,layer,sel,edited):
                y=enc.predict(x[k:k+batch],32,[.5])["quantiles"][...,0]
        values.append(y)
    return np.concatenate(values)


def collect(name,scratch,batch):
    all_j={}
    enc=adapter_for(name)
    for p in enc.module.parameters():
        p.requires_grad_(False)
    site=MODELS[name][-1]
    for cohort,root,horizon in (("cal",REP,32),("test",BLIND,16)):
        h,pos,ids=read_materials(root,name)
        with np.load(root/"data.npz") as f:
            use=f["split"]=="cal" if cohort=="cal" else np.ones(len(ids),dtype=bool)
            x=f["x"][use];ids=ids[use];h=h[use];split=f["split"][use]
        sel=last_window_slice(enc,pos)
        js=[]
        for i in range(0,len(x),batch):
            state=torch.from_numpy(h[i:i+batch])
            d=torch.zeros_like(state[:,0],requires_grad=True)
            with torch.enable_grad(),token_patch(enc.module,site,sel,state+d[:,None]):
                pred=enc.predict_differentiable(x[i:i+batch],horizon)
                g=[torch.autograd.grad(pred[:,lead].sum(),d,retain_graph=lead<horizon-1)[0] for lead in range(horizon)]
            js.append(torch.stack(g,dim=1).numpy())
        jac=np.concatenate(js)
        if cohort=="cal":
            with np.load(OLD16/(name+"_jac16.npz")) as f:
                m=f["split"]=="cal"
                np.testing.assert_array_equal(ids,f["ids"][m])
                np.testing.assert_allclose(jac[:,:16],f["jac"][m],rtol=.003,atol=5e-5)
        else:
            with np.load(BLIND/(name+"_early.npz")) as f:
                np.testing.assert_array_equal(ids,f["ids"])
                np.testing.assert_allclose(jac[:,:8],f["jac"],rtol=.003,atol=5e-5)
        all_j[cohort]=jac;all_j[cohort+"_ids"]=ids
    scratch.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(scratch/(name+".npz"),**all_j)
    print(name,"geometry done",flush=True)


def predictor(gram,cols):
    core=gram[np.ix_(cols,cols)]
    ridge=.001*np.linalg.eigvalsh(core)[-1]
    return np.linalg.solve(core+ridge*np.eye(len(cols)),gram[cols]).T


def select(cal):
    permitted=np.setdiff1d(np.arange(16),AUDIT)
    by_row=cal@cal.swapaxes(1,2)
    folds=np.array_split(np.random.default_rng(88132).permutation(len(cal)),4)
    scores=[]
    subsets=list(combinations(permitted,8))
    eye=np.eye(32)
    for cols in subsets:
        cols=np.array(cols)
        error=0.
        for val in folds:
            fit=np.ones(len(cal),dtype=bool);fit[val]=False
            op=eye[AUDIT]-predictor(by_row[fit].sum(axis=0),cols)[AUDIT]@eye[cols]
            op-=op.mean(axis=0,keepdims=True)
            error+=np.trace(op@by_row[val].sum(axis=0)@op.T)
        scores.append(error)
    return np.array(subsets[np.argmin(scores)]),np.asarray(subsets),np.asarray(scores)


def run(source,scratch,out,batch):
    target=next(k for k in MODELS if k!=source)
    with np.load(BLIND/"data.npz") as f:
        x,ids,split=f["x"],f["ids"],f["split"]
    with np.load(scratch/(source+".npz")) as s,np.load(scratch/(target+".npz")) as t:
        np.testing.assert_array_equal(ids,s["test_ids"])
        np.testing.assert_array_equal(ids,t["test_ids"])
        cal=s["cal"].astype(float);sj=s["test"].astype(float);tj=t["test"].astype(float)
    with np.load(REFERENCE/"results"/(source+"_seed0.npz")) as f:
        atoms=f["atoms"]
    with np.load(OLD16/(target+"_jac16.npz")) as f:
        rad=float(f["radius"])
    sae=load_sae_checkpoint(str(dictionary_path(source)))
    dec=sae.W_dec.detach().numpy()[atoms].astype(float).T
    hs,ps,ii=read_materials(BLIND,source)
    np.testing.assert_array_equal(ids,ii)
    ht,pt,ii=read_materials(BLIND,target)
    np.testing.assert_array_equal(ids,ii)
    with torch.inference_mode():
        codes=sae.encode(torch.from_numpy(hs.mean(axis=1))).numpy()[:,atoms]
    chosen,subsets,cv=select(cal)
    with np.errstate(over="raise"):
        sig=sj@dec
        gram=np.einsum("nhd,nkd->hk",cal,cal,optimize=True)
    record=dict(ids=ids,split=split,atoms=atoms,active=codes>0,subsets=subsets,cv_error=cv)
    src=adapter_for(source)
    sel=last_window_slice(src,ps)
    base=forecast32(src,MODELS[source][-1],x,hs,None,sel,batch)
    record["source"],_=apply(src,MODELS[source][-1],sel,x,hs,base,codes,dec,np.inf,batch,forecast32)
    del src
    enc=adapter_for(target); sel=last_window_slice(enc,pt)
    base=forecast32(enc,MODELS[target][-1],x,ht,None,sel,batch)
    for label,cols in (("first",FIRST),("spread",SPREAD),("selected",chosen)):
        center=cal[:,cols].mean(axis=1,keepdims=True)
        weight=np.sqrt(np.mean(center**2)/np.mean((cal[:,cols]-center)**2))
        w=solve_adaptive(boost_shape(tj[:,cols],weight),boost_shape(sig[:,cols],weight))
        record[label],record[label+"_clips"]=apply(enc,MODELS[target][-1],sel,x,ht,base,codes,w,rad,batch,forecast32)
        record[label+"_columns"]=cols
        raw=(-codes[:,:,None]*sig[:,cols].transpose(0,2,1))
        record[label+"_regression"]=(raw@predictor(gram,cols).T).astype("f4")
        record[label+"_interp"]=np.stack([np.interp(np.arange(32),cols,row) for row in raw.reshape(-1,8)]).reshape(*codes.shape,32).astype("f4")
        print(source,label,(cols+1).tolist(),flush=True)
    from research.activation_transfer.shape_transfer.blind_frequency.data import HERE as PRIOR
    with np.load(PRIOR/"results"/f"{source}_fresh.npz") as prev:
        np.testing.assert_array_equal(prev["ids"],ids)
        np.testing.assert_allclose(record["source"][:,:,:16],prev["source"],atol=4e-4,rtol=.004)
        np.testing.assert_allclose(record["first"][:,:,:16],prev["minimum"],atol=4e-4,rtol=.004)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(source+".npz"),**record)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("geometry","run"))
    p.add_argument("name",choices=MODELS)
    p.add_argument("--scratch",type=Path,default=TMP)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=8)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.stage=="geometry": collect(a.name,a.scratch,a.batch)
    else: run(a.name,a.scratch,a.out,a.batch)
