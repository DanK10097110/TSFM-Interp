"""Finite-dose cross-model writes that explicitly protect the recipient's second forecast patch."""
import argparse
import json
import numpy as np
import torch

from research.activation_transfer.moment.token_paths.geometry import CACHE,HERE,histories,load,layers,physical
from research.activation_transfer.moment.token_paths.intervene import source_requests
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.steering_budget.run import coefficients
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,entry,component

ARMS=("early","protected","wrong","equal_early","equal_protected")


def protect(early,late):
    _,s,v=np.linalg.svd(late,full_matrices=False)
    v=v*(s>1e-7*s[:,:1])[:,:,None]
    return early-(early@v.swapaxes(-1,-2))@v


def solve(early,goal,cap,weight):
    j=boost_shape(early,weight);y=boost_shape(goal[:,:,None],weight)[:,:,0]
    return list(coefficients(j,y,np.full(len(y),cap)))[1]


def sample(rec,split):
    row,col=rec["row"],rec["col"]
    chosen=np.zeros(len(row),bool)
    rng=np.random.default_rng(22112273)
    for dom in np.unique(split):
        for atom in np.unique(col):
            subset=np.flatnonzero((split[row]==dom)&(col==atom))
            chosen[rng.choice(subset,min(16,len(subset)),replace=False)]=True
    indices=np.flatnonzero(chosen)
    return indices,row[indices],col[indices]


def run(size,source,batch):
    a=load(size);layer=layers(a)["middle"]
    x,ids,split=histories(False)
    rec=source_requests(source)
    np.testing.assert_array_equal(ids,rec["ids"])
    ix,r,c=sample(rec,split)
    active=np.zeros_like(rec["active"]);active[r,c]=True
    with np.load(CACHE/size/"eval.npz") as f:states,jac,base=f["middle"],f["middle_jac"].astype(float),f["base"]
    with np.load(CACHE/size/"train.npz") as f:train=f["middle"]
    rng=np.random.default_rng(22112274)
    other=np.arange(len(ids))
    for dom in np.unique(split):
        perm=rng.permutation(np.flatnonzero(split==dom))
        other[perm]=np.roll(perm,1)
    assert (other!=np.arange(len(ids))).all()
    folder=CACHE/size/"protected"/source;folder.mkdir(exist_ok=True,parents=True)
    out=dict(ids=ids,split=split,row=r,col=c,atoms=rec["atoms"],active=active,other=other,
             donor=rec["donor"][ix],request=rec["request"][ix])
    for slot,idx in (("first",1),("second",2)):
        early,late=jac[r,:8,idx],jac[r,8:,idx]
        cap=.2*np.median(np.linalg.norm(train[:,idx],axis=1))
        goal=rec["request"][ix]
        writes={}
        for name,op in (("early",early),("protected",protect(early,late)),("wrong",protect(early,jac[other[r],8:,idx]))):
            writes[name]=solve(op,goal,cap,rec["weight"])
        norm={k:np.linalg.norm(v,axis=1) for k,v in writes.items()}
        small=np.minimum(norm["early"],norm["protected"])
        for key in ("early","protected"):
            writes["equal_"+key]=writes[key]*(small/np.maximum(norm[key],1e-30))[:,None]
        for name,v in writes.items():
            length=np.linalg.norm(v,axis=1)
            assert (length<cap*(1+1e-6)).all()
            start=slot+"_"+name
            out[start+"_norm"]=length
            linear=np.einsum("nhd,nd->nh",jac[r,:,idx],v)
            if name=="protected":
                normalizer=np.linalg.norm(late,axis=(1,2))*length
                np.testing.assert_allclose(np.linalg.norm(linear[:,8:],axis=1)/np.maximum(normalizer,1e-20),0.,atol=1e-6)
            out[start+"_linear"]=linear
            for dose in (1.,.25):
                tag=start+("_1" if dose==1 else "_quarter")
                dest=folder/(tag+".npy")
                if not dest.exists():
                    wave=physical(a,layer,x[r],states[r],slot,(v*dose).astype("f4"),batch)-base[r]
                    np.save(dest,wave)
                out[tag]=np.load(dest)
            print(size,source,start,len(r),flush=True)
    np.savez_compressed(HERE/"results"/f"{size}_{source}_protected.npz",**out)


def score(size,source):
    path=HERE/"results"/f"{size}_{source}_protected.npz"
    with np.load(path) as f:d=dict(f)
    report={}
    for dom in np.unique(d["split"]):
        p=Panel(d["active"],d["row"],d["col"],d["split"]==dom)
        cell={"eligible":int(p.keep.sum()),"events":int(p.events.sum()),"atoms":d["atoms"][p.keep].tolist()}
        for slot in ("first","second"):
            for dose,suffix in ((1.,"_1"),(.25,"_quarter")):
                target=d["donor"][:,:8]*dose
                block={}
                for name in ARMS:
                    k=slot+"_"+name;y=d[k+suffix]
                    amp=np.sqrt(np.mean(y[p.events]**2,axis=0))
                    early_rms=float(np.sqrt(np.mean(amp[:8]**2)))
                    late_rms=float(np.sqrt(np.mean(amp[8:]**2)))
                    late=y[:,8:].astype(float)
                    # Event-weighted energy bootstrap clusters all of a history's firings.
                    energy=np.sum(p.cube(late)**2,axis=(1,2))
                    cnt=p.mask.sum(1)*8
                    late_ci=np.sqrt((p.hdraw@energy)/np.maximum(1,p.hdraw@cnt))
                    block[name]=dict(shape=entry(p.estimate(component(y[:,:8],"shape"),component(target,"shape"))),
                                     full=entry(p.estimate(y[:,:8],target)),early_rms=early_rms,late_rms=late_rms,
                                     late_rms_ci=np.quantile(late_ci,[.025,.975]).tolist(),
                                     median_norm=float(np.median(d[k+"_norm"][p.events]))*dose)
                cell[slot+suffix]=block
        report[str(dom)]=cell
    path.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size",choices=("small","large"));p.add_argument("source",choices=("bolt","timesfm"))
    p.add_argument("--batch",type=int,default=16);p.add_argument("--score-only",action="store_true")
    a=p.parse_args();torch.set_num_threads(2)
    if not a.score_only:run(a.size,a.source,a.batch)
    score(a.size,a.source)
