"""Forecast a true full-dose interaction from four partial donor perturbations and realize it in the recipient."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH_FIT,histories
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape,solve_adaptive
from research.activation_transfer.shape_transfer.interactions.run import OUT, common_scale
from tsfm_lens.sae.train import load_sae_checkpoint

PERM=[3,6,1,7,0,2,5,4]


def mixed(low_low,low_high,high_low,high_high):
    return 4.*(high_high-high_low-low_high+low_low)


def geometry(cohort):
    names=("bolt","timesfm")
    dec=None
    ids0,z,atoms,dec,_,weight=source_material("bolt")
    states,jacs,positions,radii={},{},{},{}
    if cohort=="blind":
        with np.load(BLIND/"data.npz") as f:
            keep=f["split"]=="daily"
            x,ids=f["x"][keep],f["ids"][keep]
        for n in names:
            h,pos,all_ids=read_materials(BLIND,n)
            np.testing.assert_array_equal(ids,all_ids[keep])
            states[n],positions[n]=h[keep],pos
            with np.load(BLIND/(n+"_early.npz")) as f:
                jacs[n]=f["jac"][keep].astype(float)
                np.testing.assert_array_equal(ids,f["ids"][keep])
        sae=load_sae_checkpoint(str(dictionary_path("bolt")))
        with torch.inference_mode():
            codes=sae.encode(torch.from_numpy(states["bolt"].mean(axis=1))).numpy()[:,atoms]
    else:
        all_ids,all_x,split=histories()
        label="daily" if cohort=="original" else "extra_daily"
        x,ids=all_x[split==label],all_ids[split==label]
        k=np.isin(ids0,ids)
        np.testing.assert_array_equal(ids0[k],ids)
        codes=z[k]
    for n,depth in (("bolt",3),("timesfm",10)):
        with np.load(DEPTH_FIT/f"{n}_{depth}.npz") as f:
            radii[n]=float(f["radius"])
            if cohort!="blind":
                use=f["split"]==label
                np.testing.assert_array_equal(ids,f["ids"][use])
                jacs[n]=f["jac"][use].astype(float)
                states[n],positions[n]=f["states"][use],f["positions"]
    return x,ids,atoms,codes,dec,weight,states,jacs,positions,radii


def run(cohort,out,batch,part=0,parts=1):
    x,ids,atoms,z,dec,weight,h,j,pos,rad=geometry(cohort)
    at,bt=np.triu_indices(len(atoms),1)
    rows,pairs=np.nonzero((z[:,at]>0)&(z[:,bt]>0))
    if parts<1 or not 0<=part<parts:
        raise ValueError("Invalid partition")
    selection=np.array_split(np.arange(len(rows)),parts)[part]
    rows,pairs=rows[selection],pairs[selection]
    shape=(len(rows),16)
    rec={k:np.zeros(shape,dtype="f4") for k in ("source_a","source_b","source_joint","inner","corner",
                                                 "target_a","target_b","target_joint","corrected","wrong","zero",
                                                 "source_noise","target_noise")}
    rec.update(ids=ids,atoms=atoms,indices=np.stack((at,bt),axis=1),rows=rows,pairs=pairs)
    rec["scale"]=np.ones(len(rows))
    for arm in ("corrected","wrong","zero"):
        rec[arm+"_capped"]=np.zeros(len(rows),dtype=bool)
    vectors=solve_adaptive(boost_shape(j["timesfm"],weight),boost_shape(j["bolt"]@dec,weight))
    for model in ("bolt","timesfm"):
        enc=adapter_for(model)
        site=MODELS[model][-1]
        sel=last_window_slice(enc,pos[model])
        for start in range(0,len(rows),batch):
            ix=slice(start,start+batch)
            i,k=rows[ix],pairs[ix]
            def predict(v):
                return forecast16(enc,site,x[i],h[model][i],None if v is None else v.astype("f4"),sel,batch).astype(float)
            def get(features):
                direction=dec[:,features].T if model=="bolt" else vectors[i,:,features]
                return -z[i,features,None]*direction
            a,b=get(at[k]),get(bt[k])
            base=predict(np.zeros_like(a))
            key="source" if model=="bolt" else "target"
            rec[key+"_noise"][ix]=base-predict(None)
            if model=="timesfm":
                factor=common_scale(a,b,.2*rad[model])
                rec["scale"][ix]=factor
                a,b=a*factor[:,None],b*factor[:,None]
            ya,yb,yjoint=(predict(v)-base for v in (a,b,a+b))
            rec[key+"_a"][ix],rec[key+"_b"][ix],rec[key+"_joint"][ix]=ya,yb,yjoint
            if model=="bolt":
                rec["inner"][ix]=mixed(predict(.25*a+.25*b),predict(.25*a+.75*b),
                                       predict(.75*a+.25*b),predict(.75*a+.75*b))
                rec["corner"][ix]=mixed(base,predict(.5*b),predict(.5*a),predict(.5*(a+b)))
            else:
                actual=yjoint-ya-yb
                request=rec["inner"][ix,:8].astype(float)
                for arm,g in (("corrected",request),("wrong",request[:,PERM]),("zero",np.zeros_like(request))):
                    residual=(g-actual[:,:8])[:,:,None]
                    correction=solve_adaptive(boost_shape(j[model][i],weight),boost_shape(residual,weight))[:,:,0]
                    total=a+b+correction
                    clip=np.minimum(1.,(.2*rad[model])/np.maximum(np.linalg.norm(total,axis=1),1e-15))
                    rec[arm+"_capped"][ix]=clip<1.
                    rec[arm][ix]=predict(total*clip[:,None])-base
        print(cohort,model,"done",flush=True)
    out.mkdir(exist_ok=True,parents=True)
    path=out/(cohort+"_conditional.npz") if parts==1 else out/f"{cohort}_{part}of{parts}.npz"
    np.savez_compressed(path,**rec)


def merge(cohort,out,parts,shards):
    collected={}
    shared=None
    for k in range(parts):
        with np.load(shards/f"{cohort}_{k}of{parts}.npz") as f:
            if shared is None:
                shared={key:f[key] for key in ("ids","atoms","indices")}
            else:
                for key,value in shared.items():
                    np.testing.assert_array_equal(value,f[key])
            for key in f:
                if key not in shared:
                    collected.setdefault(key,[]).append(f[key])
    combined={key:np.concatenate(values) for key,values in collected.items()}
    tags=combined["rows"]*len(shared["indices"])+combined["pairs"]
    if len(tags)==0 or np.any(np.diff(tags)<=0):
        raise ValueError("Missing ordering or duplicate paired cases")
    _,ids,atoms,codes,*_=geometry(cohort)
    np.testing.assert_array_equal(ids,shared["ids"])
    np.testing.assert_array_equal(atoms,shared["atoms"])
    a,b=shared["indices"].T
    rows,pairs=np.nonzero((codes[:,a]>0)&(codes[:,b]>0))
    np.testing.assert_array_equal(rows,combined["rows"])
    np.testing.assert_array_equal(pairs,combined["pairs"])
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(cohort+"_conditional.npz"),**shared,**combined)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort",choices=("original","extra","blind"))
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=24)
    p.add_argument("--part",type=int,default=0)
    p.add_argument("--parts",type=int,default=1)
    p.add_argument("--merge-from",type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.merge_from is None:
        run(a.cohort,a.out,a.batch,a.part,a.parts)
    else:
        merge(a.cohort,a.out,a.parts,a.merge_from)
