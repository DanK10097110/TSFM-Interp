"""Physically test fixed versus current-history transfer at controlled hidden write norms."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.functional.gradient import OUT
from research.activation_transfer.moment.dictionary_transfer.functional.run import fixed
from research.activation_transfer.moment.dictionary_transfer.intervene import SCRATCH
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH
from research.activation_transfer.moment.token_paths.geometry import CACHE, histories, load
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.steering_budget.run import coefficients
from research.activation_transfer.shape_transfer.steering_budget.score import Panel, component, entry, difference


def historical(source,target,stem,decoder,limit,weight):
    values={}
    for side in (source,target):
        with np.load(OUT/f"{side}_gradient.npz") as f:values[side]=(f["ids"],f["jac"].astype(float),f["split"])
    ids,jac,split=values[target]
    np.testing.assert_array_equal(ids,values[source][0])
    donor=values[source][1]@decoder.T
    a=boost_shape(jac,weight);b=boost_shape(donor,weight)
    tr=split=="fit";va=split=="cal"
    def stack(x,rows):return x[rows].reshape(-1,x.shape[-1])
    fixed4,meta=fixed(stack(a,tr),stack(b,tr),stack(a,va),stack(b,va),4*limit)
    return fixed4.T,meta


def control_rows(split):
    rng=np.random.default_rng(22112360)
    alternate=np.empty(len(split),int)
    for domain in sorted(set(split)):
        ix=rng.permutation(np.flatnonzero(split==domain))
        alternate[ix]=np.roll(ix,1)
    assert np.all(alternate!=np.arange(len(split)))
    return alternate


def run(source,target,batch):
    stem=f"{source}_22112201_to_{target}"
    with np.load(OUT/f"{stem}_fit.npz") as f:cap,weight=f["cap"],float(f["weight"])
    with np.load(MATCH/f"{stem}_matches.npz") as f:decoder=f["source_decoder"].astype(float)
    v4,meta=historical(source,target,stem,decoder,cap,weight)
    x,ids,split=histories(False)
    alternate=control_rows(split)
    with np.load(SCRATCH/stem/"events.npz") as f:
        row,col,z=f["row"],f["col"],f["z"]
        np.testing.assert_array_equal(ids,f["ids"])
    with np.load(CACHE/source/"eval.npz") as f:source_gradient=f["middle_jac"][:,:8,0].astype(float)
    with np.load(CACHE/target/"eval.npz") as f:target_gradient=f["middle_jac"][:,:8,0].astype(float)
    amp=z[row,col]
    rawgoal=np.einsum("nhd,nd->nh",source_gradient[row],-amp[:,None]*decoder[col])
    goal=boost_shape(rawgoal[:,:,None],weight)[:,:,0]
    limits=amp*cap[col]
    mat=boost_shape(target_gradient,weight)
    _,current=coefficients(mat[row],goal,limits)
    _,other=coefficients(mat[alternate[row]],goal,limits)
    length=np.linalg.norm(current,axis=1)
    other*=np.divide(length,np.linalg.norm(other,axis=1))[:,None]
    np.testing.assert_allclose(np.linalg.norm(other,axis=1),length,atol=1e-10)
    assert np.all(length<=limits+1e-8)
    vecs=dict(current_matched=current,wrong_matched=other,dense4=-amp[:,None]*v4[col])
    folder=CACHE/"functional"/(stem+"_budget");folder.mkdir(exist_ok=True,parents=True)
    a=load(target)
    payload=dict(row=row,col=col,ids=ids,alternate=alternate,cap=limits,early_goal=rawgoal)
    for arm,v in vecs.items():
        blocks=[]
        for lo in range(0,len(row),80):
            part=folder/f"{arm}_{lo}.npy"
            if not part.exists():
                effects=[]
                for i in range(lo,min(len(row),lo+80),batch):
                    subset=slice(i,min(i+batch,lo+80,len(row)))
                    h=x[row[subset]]
                    base=a.predict(h,16,[.5])["point"]
                    with addition(a,v[subset]):
                        moved=a.predict(h,16,[.5])["point"]
                    effects.append(moved-base)
                tmp=part.with_suffix(".tmp.npy");np.save(tmp,np.concatenate(effects));tmp.replace(part)
            blocks.append(np.load(part))
        payload[arm]=np.concatenate(blocks)
        payload[arm+"_length"]=np.linalg.norm(v,axis=1)
        print(stem,arm,len(row),flush=True)
    np.savez_compressed(OUT/f"{stem}_budget.npz",**payload)
    (OUT/f"{stem}_budget_fit.json").write_text(json.dumps(meta,indent=2)+"\n")


def report(source,target):
    stem=f"{source}_22112201_to_{target}"
    with np.load(MATCH/f"{stem}_forecast.npz") as f:e=dict(f)
    with np.load(OUT/f"{stem}_budget.npz") as f:
        np.testing.assert_array_equal(e["row"],f["row"])
        np.testing.assert_array_equal(e["col"],f["col"])
        vecs={k:f[k] for k in ("current_matched","wrong_matched","dense4")}
        norms={k:f[k+"_length"] for k in vecs};limits=f["cap"]
    with np.load(OUT/f"{stem}_dense.npz") as f:vecs["historical"]=f["effect"]
    vecs["state_map"]=e["native_dense"]
    out={}
    for domain in sorted(set(e["split"])):
        panel=Panel(e["z"]>0,e["row"],e["col"],e["split"]==domain)
        cell=dict(events=int(panel.events.sum()),n=int(panel.keep.sum()),
                  fraction_of_budget={k:float(np.median(v[panel.events]/limits[panel.events])) for k,v in norms.items()})
        for name,sl in (("early",slice(8)),("late",slice(8,16))):
            for kind in ("shape","full"):
                ref=component(e["truth_native"][:,sl],kind)
                estimates={k:panel.estimate(component(v[:,sl],kind),ref) for k,v in vecs.items()}
                block={k:entry(v) for k,v in estimates.items()}
                for a,b in (("current_matched","wrong_matched"),("current_matched","historical"),("dense4","historical")):
                    block[a+"_minus_"+b]=difference(estimates[a],estimates[b])
                cell[name+"_"+kind]=block
        out[str(domain)]=cell
    (OUT/f"{stem}_budget.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("run","score"))
    p.add_argument("source",choices=("small","large"));p.add_argument("target",choices=("small","large"))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="score":report(a.source,a.target)
    else:run(a.source,a.target,a.batch)
