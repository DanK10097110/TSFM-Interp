"""Physically compare a transported dense write and independent SAE matches in the same output units."""
import argparse
from pathlib import Path
from hashlib import sha256
import os
import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.match import OUT,SIZES,dictionary,encode
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,layers,load
from research.activation_transfer.moment.token_paths.intervene import receiver_write
from research.activation_transfer.moment.head_test.capture import scaling
from tsfm_lens.extraction.hooks import ActivationCatcher,token_patch

ARMS=("dense","corr_own","corr_forced","geom_forced","random","response")
SCRATCH=CACHE/"dictionary_transfer"


def physical(a,x,middle,delta,batch):
    ys=[];final=[];scales=[]
    for lo in range(0,len(x),batch):
        ix=slice(lo,lo+batch)
        write=torch.from_numpy((middle[ix]+delta[ix]).astype("f4")[:,None])
        with token_patch(a.module,layers(a)["middle"],lambda _:slice(61,62),write),ActivationCatcher(a.module,["encoder.final_layer_norm"]) as cap:
            ys.append(a.predict(x[ix],16,[.5])["point"])
            final.append(cap.collect()["encoder.final_layer_norm"][:,-2:].numpy())
            scales.extend(scaling(a.module.normalizer)[1])
    return np.concatenate(ys),np.concatenate(final),np.asarray(scales)


def restartable_physical(adapter, x, middle, delta, batch, archive):
    """Resume finite writes after a job interruption, validating the exact inputs in each block."""
    parts=archive.with_suffix(".blocks")
    parts.mkdir(exist_ok=True)
    ys=[];states=[];scales=[]
    for start in range(0,len(x),96):
        block=slice(start,start+96)
        path=parts/f"{start}.npz"
        digest=sha256()
        for array in (x[block],middle[block],delta[block]):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        signature=digest.hexdigest()
        if not path.exists():
            y,h,sd=physical(adapter,x[block],middle[block],delta[block],batch)
            tmp=parts/f"{start}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp,y=y,h=h,sd=sd,signature=signature)
            tmp.replace(path)
        with np.load(path) as f:
            assert str(f["signature"])==signature,(archive,start)
            ys.append(f["y"]);states.append(f["h"]);scales.append(f["sd"])
    return np.concatenate(ys),np.concatenate(states),np.concatenate(scales)


def run(source,target,seed,batch,requested):
    stem=f"{source}_{seed}_to_{target}"
    folder=SCRATCH/stem
    folder.mkdir(exist_ok=True,parents=True)
    rec=np.load(OUT/(stem+"_matches.npz"))
    atoms,dec=rec["atoms"],rec["source_decoder"]
    a=load(target)
    x,ids,split=histories(False)
    with np.load(CACHE/source/"eval.npz") as f:
        np.testing.assert_array_equal(f["ids"],ids)
        s=f["middle"][:,0]
        js=f["middle_jac"][:,:8,0].astype(float)
    with np.load(CACHE/target/"eval.npz") as f:
        np.testing.assert_array_equal(f["ids"],ids)
        t=f["middle"][:,0]
        jt=f["middle_jac"][:,:8,0].astype(float)
        base=f["base"]
        final0=f["readout"][:,1:]
    # Pure native RevIN factors from the ordinary past, never from a steered future.
    with torch.inference_mode():
        vals,mask=a.prepare(x)
        a.module.normalizer(x=vals,mask=mask,mode="norm")
        _,sd=scaling(a.module.normalizer)
    sa,ta=dictionary(source,seed),dictionary(target,int(rec["recipient_seed"]))
    zs=encode(sa,s)[:,atoms]
    zt=encode(ta,t)
    direction=ta.W_dec.detach().numpy()
    row,col=np.nonzero(zs>0)
    amplitude=zs[row,col,None]
    dense=-amplitude*rec["mapped"][col]
    norm=np.linalg.norm(dense,axis=1)[:,None]
    request=np.einsum("nhd,nd->nh",js[row],-amplitude*dec[col])
    with np.load(CACHE/target/"train.npz") as f:
        radius=.2*np.median(np.linalg.norm(f["middle"][:,0],axis=1))
    # A fixed balanced objective chosen a priori gives the 8 coordinates equal weight.
    response,unbounded,scale=receiver_write(jt[row],request,radius,1.)
    which=rec["corr_id"][col]
    vectors=dict(dense=dense,corr_own=-zt[row,which,None]*direction[which],
                 corr_forced=-norm*direction[which],geom_forced=-norm*direction[rec["geom_id"][col]],
                 random=-norm*direction[rec["random_id"][col]],response=response)
    np.savez_compressed(folder/"events.npz",ids=ids,split=split,atoms=atoms,z=zs,row=row,col=col,
                        target_active=zt[:,rec["corr_id"]]>0,target_codes=zt[:,rec["corr_id"]],
                        sd=sd,radius=radius,response_scale=scale)
    rng=np.random.default_rng(22112267)
    sample=rng.choice(len(ids),16,replace=False)
    clean,hclean,sc=physical(a,x[sample],t[sample],np.zeros_like(t[sample]),batch)
    np.testing.assert_allclose(clean,base[sample],atol=3e-5,rtol=1e-4)
    np.testing.assert_allclose(hclean,final0[sample],atol=2e-4,rtol=1e-4)
    np.testing.assert_allclose(sc,sd[sample],atol=3e-6)
    visit=ARMS if requested=="all" else (requested,)
    for key in visit:
        dest=folder/(key+".npz")
        if dest.exists():continue
        delta=vectors[key]
        y,post,_=restartable_physical(a,x[row],t[row],delta,batch,dest)
        # Cached ordinary passes may have slightly different floating-point summation across batches.
        dstate=(post.astype(float)-final0[row]).astype("f4")
        effects=y-base[row]
        direct=(dstate.astype(float)@a.module.head.linear.weight.detach().numpy().T*sd[row,None,None]).reshape(-1,16)
        np.testing.assert_allclose(direct,effects,atol=1e-5,rtol=.0002)
        temp=dest.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temp,native=effects,dstate=dstate,norm=np.linalg.norm(delta,axis=1),
                            linear=np.einsum("nhd,nd->nh",jt[row],delta))
        temp.replace(dest)
        print(stem,key,len(row),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=SIZES);p.add_argument("target",choices=SIZES)
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    p.add_argument("--arm",choices=(*ARMS,"all"),default="all");p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);run(a.source,a.target,a.seed,a.batch,a.arm)
