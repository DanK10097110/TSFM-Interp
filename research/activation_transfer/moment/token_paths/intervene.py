"""Physically replay donor feature removals through specified MOMENT patch positions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND, HERE as OBSERVED
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.moment.token_paths.geometry import CACHE,HERE,SLOTS,histories,layers,load,physical
from tsfm_lens.sae.train import load_sae_checkpoint


def readout(adapter,x,states,slot,delta):
    """Apply the actual pointwise reconstruction head and RevIN to cached final states."""
    with torch.no_grad():
        values,mask=adapter.prepare(x)
        adapter.module.normalizer(x=values,mask=mask,mode="norm")
        edited=torch.as_tensor(states.copy())
        edited[:,SLOTS[slot]]+=torch.as_tensor(delta[:,None])
        y=adapter.module.head(edited[:,None])
        return adapter.module.normalizer(x=y,mode="denorm")[:,0,-16:].numpy()


def receiver_write(jac,request,limit,weight):
    """Return zero for the structurally disconnected slot; use ridge at reachable slots."""
    j=boost_shape(jac,weight)
    y=boost_shape(request[:,:,None],weight)
    gram=j@j.swapaxes(-1,-2)
    spectral=np.linalg.eigvalsh(gram)[:,-1]
    valid=spectral>0
    delta=np.zeros((len(j),j.shape[-1]),dtype=float)
    if np.any(valid):
        a=gram[valid]+(.001*spectral[valid])[:,None,None]*np.eye(j.shape[1])
        coeff=np.linalg.solve(a,y[valid])
        delta[valid]=(j[valid].swapaxes(-1,-2)@coeff)[:,:,0]
    length=np.linalg.norm(delta,axis=1)
    scale=np.minimum(1.,limit/np.maximum(length,1e-20))
    return (delta*scale[:,None]).astype("f4"),length,scale


def source_requests(source):
    with np.load(OBSERVED/"results"/(source+"_fresh.npz")) as f:
        ids,z,atoms,effects,w=f["ids"],f["codes"],f["atoms"],f["source"],float(f["weight"])
    with np.load(BLIND/(source+"_early.npz")) as f:
        np.testing.assert_array_equal(ids,f["ids"])
        j=f["jac"].astype(float)
    directions=load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float)
    row,col=np.nonzero(z>0)
    request=np.einsum("nhd,nd->nh",j[row],-z[row,col,None]*directions[col])
    return dict(ids=ids,active=z>0,atoms=atoms,row=row,col=col,donor=effects[row,col],request=request,weight=w)


def run(size,source,out,batch,site_name):
    adapter=load(size)
    sites=layers(adapter)
    x,ids,split=histories(False)
    cache=CACHE/size
    rec=source_requests(source)
    np.testing.assert_array_equal(ids,rec["ids"])
    r=rec["row"]
    out.mkdir(parents=True,exist_ok=True)
    with np.load(cache/"train.npz") as f:train=f[site_name]
    with np.load(cache/"eval.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"])
        states,jac,base=f[site_name],f[site_name+"_jac"].astype(float),f["base"]
    rec["split"]=split
    rec["base"]=base
    rec["layer"]=sites[site_name]
    baseline=np.concatenate([adapter.predict(x[r[lo:lo+batch]],16,[.5])["point"] for lo in range(0,len(r),batch)])
    rng=np.random.default_rng(22112233)
    replay=np.concatenate([rng.choice(np.flatnonzero(split[r]==k),6,replace=False) for k in np.unique(split)])
    for slot,selector in SLOTS.items():
        index=np.arange(3)[selector]
        # Hold total Euclidean write-energy cap fixed when sending the same vector to two future tokens.
        radius_slot=1 if slot=="both" else index[0]
        limit=.2*np.median(np.linalg.norm(train[:,radius_slot].astype(float),axis=-1))/np.sqrt(len(index))
        j=jac[:,0:8,index,:].sum(2)
        future=jac[:,8:,index,:].sum(2)
        delta,raw,scale=receiver_write(j[r],rec["request"],limit,rec["weight"])
        subset=r[replay]
        zero=physical(adapter,sites[site_name],x[subset],states[subset],slot,np.zeros_like(delta[replay]),batch)
        np.testing.assert_allclose(zero,base[subset],atol=2e-5,rtol=.0002)
        if site_name=="readout":
            result=readout(adapter,x[r],states[r],slot,delta)-readout(adapter,x[r],states[r],slot,np.zeros_like(delta))
            live=physical(adapter,sites[site_name],x[subset],states[subset],slot,delta[replay],batch)-zero
            rec[slot+"_live_error"]=np.max(np.abs(result[replay]-live))
            np.testing.assert_allclose(result[replay],live,atol=2e-5,rtol=.0002)
        else:
            result=physical(adapter,sites[site_name],x[r],states[r],slot,delta,batch)-baseline
        rec[slot]=result
        rec[slot+"_linear"]=np.concatenate([np.einsum("nhd,nd->nh",a[r],delta) for a in (j,future)],axis=1).astype("f4")
        rec[slot+"_length"]=np.linalg.norm(delta.astype(float),axis=1)*np.sqrt(len(index))
        rec[slot+"_unbounded"]=raw*np.sqrt(len(index))
        rec[slot+"_scale"]=scale
        rec[slot+"_cap"]=limit*np.sqrt(len(index))
        print(size,source,site_name,slot,len(r),int(np.sum(scale<1)),flush=True)
    if site_name=="readout":
        np.testing.assert_array_equal(rec["first"][:,8:],0.)
        np.testing.assert_array_equal(rec["history"],0.)
        np.testing.assert_array_equal(rec["second"],0.)
        # Shared readout and common RevIN make the same hidden write repeat in the second patch.
        np.testing.assert_allclose(rec["both"][:,:8],rec["both"][:,8:],atol=5e-5,rtol=.001)
    np.savez_compressed(out/f"{size}_{source}_{site_name}.npz",**rec)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("size",choices=("small","large"))
    p.add_argument("source",choices=("bolt","timesfm"))
    p.add_argument("site",choices=("middle","readout"))
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--out",type=Path,default=HERE/"results")
    args=p.parse_args();torch.set_num_threads(args.threads)
    run(args.size,args.source,args.out,args.batch,args.site)
