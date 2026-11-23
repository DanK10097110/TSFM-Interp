"""Four physical forwards for a joint pair, using a shared target norm cap across factorial conditions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.geometry import HERE as DEPTH, SCRATCH as FIT, histories
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape, solve_adaptive

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def common_scale(first,second,limit):
    length=np.maximum.reduce([np.linalg.norm(v,axis=1) for v in (first,second,first+second)])
    return np.minimum(1.,limit/np.maximum(length,1e-15))


def run(source,target,mode,out,batch):
    ids,z,atoms,dec,sig,weight=source_material(source)
    model=source if mode=="native" else target
    depth=3 if model=="bolt" else 10
    all_ids,xx,lab=histories()
    only=~np.isin(lab,["train","cal"])
    x=xx[only]
    with np.load(FIT/f"{model}_{depth}.npz") as f:
        ev=f["split"]!="cal"
        ids=ids[ev]
        np.testing.assert_array_equal(ids,all_ids[only])
        np.testing.assert_array_equal(f["ids"][ev],ids)
        h,j,rad,site,pos=f["states"][ev],f["jac"][ev].astype(float),float(f["radius"]),str(f["layer"]),f["positions"]
        base=f["base"][ev]
    z,sig=z[ev],sig[ev]
    w=dec if mode=="native" else solve_adaptive(boost_shape(j,weight),boost_shape(sig,weight))
    aa,bb=np.triu_indices(len(atoms),1)
    active=(z[:,aa]>0)&(z[:,bb]>0)
    rows,pairs=np.nonzero(active)
    record=dict(ids=ids,split=lab[only],atoms=atoms,indices=np.stack((aa,bb),axis=1),
                rows=rows,pairs=pairs,scale=np.ones(len(rows)),a=np.zeros((len(rows),16),dtype="f4"),
                b=np.zeros((len(rows),16),dtype="f4"),joint=np.zeros((len(rows),16),dtype="f4"),
                noise=np.zeros((len(rows),16),dtype="f4"))
    enc=adapter_for(model)
    select=last_window_slice(enc,pos)
    for start in range(0,len(rows),batch):
        obs=rows[start:start+batch]; k=pairs[start:start+batch]
        def vector(features):
            direction=w[:,features].T if mode=="native" else w[obs,:,features]
            return -z[obs,features,None]*direction
        a,b=vector(aa[k]),vector(bb[k])
        scale=common_scale(a,b,np.inf if mode=="native" else .2*rad)
        record["scale"][start:start+len(obs)]=scale
        a,b=a*scale[:,None],b*scale[:,None]
        ordinary=forecast16(enc,site,x[obs],h[obs],None,select,batch).astype(float)
        np.testing.assert_allclose(ordinary,base[obs],atol=8e-5,rtol=3e-5)
        for key,v in (("a",a),("b",b),("joint",a+b),("noise",np.zeros_like(a))):
            y=forecast16(enc,site,x[obs],h[obs],v.astype("f4"),select,batch)
            record[key][start:start+len(obs)]=y.astype(float)-ordinary
    if mode=="native":
        with np.load(DEPTH/"results"/(source+"_reference.npz")) as f:
            flat=f["effects"]
            np.testing.assert_array_equal(ids,f["ids"])
            for key,index in (("a",aa),("b",bb)):
                np.testing.assert_allclose(record[key],flat[rows,index[pairs]],atol=4e-4,rtol=.004)
    out.mkdir(exist_ok=True,parents=True)
    label="native" if mode=="native" else target
    np.savez_compressed(out/f"{source}_{label}.npz",**record)
    print(source,label,"completed",len(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=MODELS)
    p.add_argument("target",choices=("native",*MODELS))
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.source,a.target,"native" if a.target=="native" else "transfer",a.out,a.batch)
