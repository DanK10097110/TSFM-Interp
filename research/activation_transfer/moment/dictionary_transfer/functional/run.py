"""Fit historical output-based fixed vectors and causal finite predictions at natural source doses."""
import argparse
import json
import numpy as np
import torch
from research.activation_transfer.moment.dictionary_transfer.functional.gradient import OUT
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH,dictionary
from research.activation_transfer.moment.dictionary_transfer.intervene import physical,SCRATCH
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,load
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,entry,difference
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape

ARMS=("dense","group","sham")
PENALTIES=(1e-6,1e-4,.001,.01,.1)


def local_svd(a):
    return np.linalg.svd(a,full_matrices=False)


def fit_ball(svd,goals,limits,penalty):
    u,s,vt=svd
    num=s[:,None]*(u.T@goals)
    denom=s[:,None]**2+penalty*s[0]**2
    def coeff(eta):return num/(denom+eta[None,:])
    raw=coeff(np.zeros(len(limits)))
    exceeds=np.linalg.norm(raw,axis=0)>limits
    lo=np.zeros(len(limits));hi=np.linalg.norm(num,axis=0)/np.maximum(limits,1e-12)
    for _ in range(48):
        mid=(lo+hi)/2
        too_long=np.linalg.norm(coeff(mid),axis=0)>limits
        hi=np.where(too_long,hi,mid);lo=np.where(too_long,mid,lo)
    return vt.T@coeff(np.where(exceeds,hi,0.))


def fixed(a,y,va,vy,limits):
    svd=local_svd(a)
    energy=np.sum(vy**2,axis=0).clip(1e-20)
    weights=[];errors=[]
    for p in PENALTIES:
        w=fit_ball(svd,y,limits,p);weights.append(w)
        errors.append(float(np.mean(np.sum((va@w-vy)**2,axis=0)/energy)))
    good=int(np.argmin(errors))
    return weights[good],dict(penalty=PENALTIES[good],validation_errors=errors)


def pursuit(a,y,axes,cap,penalty):
    responses=a@axes.T
    allowed=np.arange(len(axes))
    residual=y.copy();ids=[]
    length=np.linalg.norm(responses,axis=0).clip(1e-20)
    for _ in range(16):
        scores=np.abs(responses.T@residual)/length;scores[ids]=-np.inf
        ids.append(int(np.argmax(scores)))
        q=np.linalg.qr(axes[ids].T)[0]
        coeff=fit_ball(local_svd(a@q),y[:,None],np.array([cap]),penalty)[:,0]
        vector=q@coeff;residual=y-a@vector
    return vector,np.asarray(ids),np.linalg.lstsq(axes[ids].T,vector,rcond=1e-9)[0]


def train(source,target):
    seed=SEEDS[0];stem=f"{source}_{seed}_to_{target}"
    with np.load(MATCH/(stem+"_matches.npz")) as f:
        dec,mapped,eligible,oth=f["source_decoder"],f["mapped"],np.flatnonzero(f["eligible_target"]),int(f["recipient_seed"])
    vals={}
    for side in (source,target):
        with np.load(OUT/f"{side}_gradient.npz") as f:vals[side]=(f["ids"],f["jac"].astype(float),f["split"])
    np.testing.assert_array_equal(vals[source][0],vals[target][0])
    ids,js,splits=vals[source];jt=vals[target][1];tr=splits=="fit";cal=~tr
    goal=js@dec.T
    mu=goal[tr].mean(1,keepdims=True)
    weight=float(np.sqrt(np.mean(mu**2)/np.mean((goal[tr]-mu)**2)))
    A=boost_shape(jt,weight);Y=boost_shape(goal,weight)
    x=A[tr].reshape(-1,A.shape[-1]);y=Y[tr].reshape(-1,len(dec))
    vx=A[cal].reshape(-1,A.shape[-1]);vy=Y[cal].reshape(-1,len(dec))
    cap=np.linalg.norm(mapped,axis=1)
    true,meta=fixed(x,y,vx,vy,cap)
    rng=np.random.default_rng(22112307)
    perm=rng.permutation(tr.sum());shift=np.empty_like(perm);shift[perm]=np.roll(perm,1)
    fake,badmeta=fixed(x,Y[tr][shift].reshape(y.shape),vx,vy,cap)
    cat=dictionary(target,oth).W_dec.detach().numpy()[eligible].astype(float)
    curves=[pursuit(x,y[:,i],cat,cap[i],meta["penalty"]) for i in range(len(dec))]
    sparse=np.stack([v[0] for v in curves]).T
    selected=eligible[np.stack([v[1] for v in curves])]
    coeff=np.stack([v[2] for v in curves])
    np.testing.assert_allclose(np.einsum("nk,nkd->nd",coeff,dictionary(target,oth).W_dec.detach().numpy()[selected]),sparse.T,atol=1e-6)
    for v in (true,fake,sparse):assert (np.linalg.norm(v,axis=0)<=cap*(1+1e-8)).all()
    np.savez_compressed(OUT/(stem+"_fit.npz"),dense=true.T,sham=fake.T,group=sparse.T,ids=ids,cap=cap,selected=selected,
                        coefficients=coeff,weight=weight,permutation=shift)
    meta.update(sham=badmeta,source=source,target=target,weight=weight,fit=int(tr.sum()),val=int(cal.sum()),
                n_candidate_atoms=len(eligible),used_unique_ids=len(np.unique(selected)))
    (OUT/(stem+"_fit.json")).write_text(json.dumps(meta,indent=2)+"\n")


def run(source,target,batch):
    seed=SEEDS[0];stem=f"{source}_{seed}_to_{target}"
    saved=OUT/(stem+"_fit.npz")
    if not saved.exists():train(source,target)
    with np.load(saved) as f:vecs={k:f[k] for k in ARMS};trainids=f["ids"]
    with np.load(SCRATCH/stem/"events.npz") as f:event=dict(f)
    r,c=event["row"],event["col"]
    x,ids,split=histories(False);assert set(ids).isdisjoint(trainids)
    with np.load(CACHE/target/"eval.npz") as f:mid,base=f["middle"][:,0],f["base"]
    np.testing.assert_array_equal(ids,event["ids"])
    a=load(target)
    folder=CACHE/"functional"/stem;folder.mkdir(exist_ok=True,parents=True)
    for arm in ARMS:
        chunks=[]
        for lo in range(0,len(r),112):
            ix=slice(lo,lo+112);rs=r[ix];cs=c[ix]
            part=folder/f"{arm}_{lo}.npy"
            if not part.exists():
                delta=(-event["z"][rs,cs,None]*vecs[arm][cs]).astype("f4")
                y,_,_=physical(a,x[rs],mid[rs],delta,batch)
                temp=part.with_suffix(".tmp.npy");np.save(temp,y-base[rs]);temp.replace(part)
            chunks.append(np.load(part))
        np.savez_compressed(OUT/f"{stem}_{arm}.npz",effect=np.concatenate(chunks),ids=ids,row=r,col=c)
        print(stem,arm,len(r),flush=True)


def score(source,target):
    seed=SEEDS[0];stem=f"{source}_{seed}_to_{target}"
    with np.load(MATCH/(stem+"_forecast.npz")) as f:e=dict(f)
    with np.load(MATCH/(stem+"_group_waves.npz")) as f:geometry_group=f["sixteen_native"]
    effects=dict(geometry=e["native_dense"],geometry_group=geometry_group,current=e["native_response"])
    for k in ARMS:
        with np.load(OUT/f"{stem}_{k}.npz") as f:
            np.testing.assert_array_equal(e["row"],f["row"]);np.testing.assert_array_equal(e["col"],f["col"])
            effects[k]=f["effect"]
    out={}
    for dom in np.unique(e["split"]):
        p=Panel(e["z"]>0,e["row"],e["col"],e["split"]==dom);cell=dict(n=int(p.keep.sum()),events=int(p.events.sum()))
        for label,sl in (("early",slice(8)),("late",slice(8,16))):
            for kind in ("full","shape"):
                ref=component(e["truth_native"][:,sl],kind)
                est={k:p.estimate(component(v[:,sl],kind),ref) for k,v in effects.items()}
                block={k:entry(v) for k,v in est.items()}
                for a,b in (("dense","geometry"),("group","geometry_group"),("dense","sham"),("dense","group"),("current","dense")):
                    block[a+"_minus_"+b]=difference(est[a],est[b])
                cell[label+"_"+kind]=block
        out[str(dom)]=cell
    (OUT/(stem+"_results.json")).write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("train","run","score"))
    p.add_argument("source",choices=("small","large"));p.add_argument("target",choices=("small","large"))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="run":run(a.source,a.target,a.batch)
    else:{"score":score,"train":train}[a.stage](a.source,a.target)
