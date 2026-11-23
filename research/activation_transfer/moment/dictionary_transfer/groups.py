"""Learn reusable geometric groups of SAE atoms and replay their finite interventions."""
import argparse
import os
import json
import numpy as np
import torch
from research.activation_transfer.moment.dictionary_transfer.match import OUT,dictionary,encode,SIZES
from research.activation_transfer.moment.dictionary_transfer.intervene import restartable_physical,SCRATCH
from research.activation_transfer.moment.dictionary_transfer.score import alternate
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,load
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,entry,difference

ARMS=("one","four","sixteen","gated","dose")
RANK={"one":1,"four":4,"sixteen":16}


def pursuit(y,axes,ids):
    """Unit-norm rows, unique choices, least-squares refit after every addition."""
    selected=[];residual=y.copy();out={}
    for k in range(1,17):
        scores=np.abs(axes@residual)
        scores[selected]=-np.inf
        selected.append(int(np.argmax(scores)))
        a=axes[selected]
        coeff=np.linalg.lstsq(a.T,y,rcond=1e-9)[0]
        residual=y-coeff@a
        if k in RANK.values():out[k]=(ids[selected],coeff.copy(),coeff@a)
    return out


def build(source,target,seed):
    stem=f"{source}_{seed}_to_{target}"
    with np.load(OUT/(stem+"_matches.npz")) as f:
        y,eligible,other=f["mapped"],np.flatnonzero(f["eligible_target"]),int(f["recipient_seed"])
    d=dictionary(target,other).W_dec.detach().numpy().astype(float)
    fitted=[pursuit(row,d[eligible],eligible) for row in y]
    res={}
    for arm,r in RANK.items():
        for i,key in enumerate(("ids","coeff","vector")):res[arm+"_"+key]=np.stack([g[r][i] for g in fitted])
        res[arm+"_r2"]=1-np.sum((res[arm+"_vector"]-y)**2,axis=1)/np.sum(y**2,axis=1)
    assert np.all(res["sixteen_r2"]>=res["four_r2"]-1e-10) and np.all(res["four_r2"]>=res["one_r2"]-1e-10)
    np.savez_compressed(OUT/(stem+"_groups.npz"),**res)
    return res


def run(source,target,seed,batch):
    stem=f"{source}_{seed}_to_{target}"
    d=build(source,target,seed)
    with np.load(OUT/(stem+"_matches.npz")) as f:other=int(f["recipient_seed"])
    with np.load(SCRATCH/stem/"events.npz") as f:ev=dict(f)
    r,c,z=ev["row"],ev["col"],ev["z"]
    x,ids,dom=histories(False)
    np.testing.assert_array_equal(ids,ev["ids"])
    with np.load(CACHE/target/"eval.npz") as f:
        base,mid,end=f["base"],f["middle"][:,0],f["readout"][:,1:]
    ta=dictionary(target,other)
    codes=encode(ta,mid)
    selected=d["sixteen_ids"][c]
    allowed=codes[r[:,None],selected]>0
    coeff=d["sixteen_coeff"][c]*allowed
    rows=ta.W_dec.detach().numpy()[selected]
    groups={arm:d[arm+"_vector"][c] for arm in RANK}
    groups["gated"]=np.einsum("nk,nkd->nd",coeff,rows)
    full=groups["sixteen"]
    groups["dose"]=full*(np.linalg.norm(groups["gated"],axis=1)/np.linalg.norm(full,axis=1))[:,None]
    a=load(target)
    for key,vec in groups.items():
        path=SCRATCH/stem/("group_"+key+".npz")
        if path.exists():continue
        delta=(-z[r,c,None]*vec).astype("f4")
        y,h,_=restartable_physical(a,x[r],mid[r],delta,batch,path)
        tmp=path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp,native=y-base[r],dstate=(h-end[r]),norm=np.linalg.norm(delta,axis=1),group_active_count=allowed.sum(1))
        tmp.replace(path)
        print(stem,key,len(r),flush=True)


def score(source,target,seed):
    stem=f"{source}_{seed}_to_{target}"
    with np.load(OUT/(stem+"_forecast.npz")) as f:ev=dict(f)
    with np.load(OUT/(stem+"_groups.npz")) as f:groups=dict(f)
    rec={}
    for key in ARMS:
        with np.load(SCRATCH/stem/("group_"+key+".npz")) as f:
            native,state,length=f["native"],f["dstate"],f["norm"]
            zero=length==0
            if zero.any():np.testing.assert_allclose(native[zero],0.,atol=2e-5)
            native[zero]=0;state[zero]=0
            rec[key]={"native":native,"interior":alternate(target,state,ev["sd"][ev["row"]]),"norm":length}
            active=f["group_active_count"]
    out={};arr={}
    for key,dic in rec.items():
        for label,values in dic.items():arr[key+"_"+label]=values
    np.savez_compressed(OUT/(stem+"_group_waves.npz"),ids=ev["ids"],row=ev["row"],col=ev["col"],group_active_count=active,**arr)
    for dom in np.unique(ev["split"]):
        p=Panel(ev["z"]>0,ev["row"],ev["col"],ev["split"]==dom)
        cell=dict(n=int(p.keep.sum()),events=int(p.events.sum()),
                  geometric_r2_median={key:float(np.median(groups[key+"_r2"][p.keep])) for key in RANK},
                  mean_active_in_group=float(active[p.events].mean()))
        for head in ("native","interior"):
            for part,sl in (("early",slice(8)),("late",slice(8,16))):
                for kind in ("shape","full"):
                    ref=component(ev["truth_"+head][:,sl],kind)
                    est={k:p.estimate(component(v[head][:,sl],kind),ref) for k,v in rec.items()}
                    est["corr_own"]=p.estimate(component(ev[head+"_corr_own"][:,sl],kind),ref)
                    est["dense"]=p.estimate(component(ev[head+"_dense"][:,sl],kind),ref)
                    block={k:entry(v) for k,v in est.items()}
                    for a,b in (("sixteen","one"),("sixteen","corr_own"),("gated","dose")):block[a+"_minus_"+b]=difference(est[a],est[b])
                    cell[head+"_"+part+"_"+kind]=block
        out[str(dom)]=cell
    (OUT/(stem+"_group_scores.json")).write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("run","score"));p.add_argument("source",choices=SIZES);p.add_argument("target",choices=SIZES)
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0]);p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="run":run(a.source,a.target,a.seed,a.batch)
    else:score(a.source,a.target,a.seed)
