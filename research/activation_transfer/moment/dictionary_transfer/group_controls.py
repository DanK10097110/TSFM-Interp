"""Rotate the entire comparison dictionary and replay matched group writes of equal hidden length."""
import argparse
import json
import numpy as np
import torch
from research.activation_transfer.moment.dictionary_transfer.match import CACHE,OUT,dictionary,hidden
from research.activation_transfer.moment.dictionary_transfer.groups import pursuit
from research.activation_transfer.moment.dictionary_transfer.intervene import SCRATCH,physical
from research.activation_transfer.moment.token_paths.geometry import histories,load
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,entry,difference

ARMS=("rotated","rotated_equal","pca16")


def vectors(size):
    seed=SEEDS[0];stem=f"{size}_{seed}_to_{size}"
    with np.load(OUT/(stem+"_matches.npz")) as f:
        y,ids,other=f["source_decoder"].astype(float),np.flatnonzero(f["eligible_target"]),int(f["recipient_seed"])
    with np.load(OUT/(stem+"_groups.npz")) as f:real=f["sixteen_vector"]
    d=dictionary(size,other).W_dec.detach().numpy()[ids].astype(float)
    # Correct Householder sign convention for a Haar-distributed rotation.
    raw=np.random.default_rng(22112309).standard_normal((d.shape[1],d.shape[1]))
    q,r=np.linalg.qr(raw);q*=np.sign(np.diag(r))[None,:]
    rot=d@q
    np.testing.assert_allclose(rot@rot.T,d@d.T,atol=1e-12)
    sur=np.stack([pursuit(a,rot,ids)[16][2] for a in y])
    state=hidden(size)[:,-1].astype(float);state-=state.mean(0)
    _,_,v=np.linalg.svd(state,full_matrices=False)
    pca=(y@v[:16].T)@v[:16]
    bank={"rotated":sur,"rotated_equal":sur*(np.linalg.norm(real,axis=1)/np.linalg.norm(sur,axis=1))[:,None],"pca16":pca}
    geom={name:1-np.sum((arr-y)**2,axis=1)/np.sum(y**2,axis=1) for name,arr in bank.items()}
    return stem,bank,geom


def run(size,batch):
    stem,vec,geom=vectors(size)
    with np.load(SCRATCH/stem/"events.npz") as f:ev=dict(f)
    x,ids,split=histories(False);np.testing.assert_array_equal(ev["ids"],ids)
    with np.load(CACHE/size/"eval.npz") as f:state,base=f["middle"][:,0],f["base"]
    a=load(size);r,c=ev["row"],ev["col"]
    rec={};folder=CACHE/"group_controls"/size;folder.mkdir(exist_ok=True,parents=True)
    for arm in ARMS:
        vals=[]
        for lo in range(0,len(r),112):
            dest=folder/f"{arm}_{lo}.npy"
            if not dest.exists():
                ix=slice(lo,lo+112);rr,cc=r[ix],c[ix]
                delta=(-ev["z"][rr,cc,None]*vec[arm][cc]).astype("f4")
                y,_,_=physical(a,x[rr],state[rr],delta,batch)
                temp=dest.with_suffix(".tmp.npy");np.save(temp,y-base[rr]);temp.replace(dest)
            vals.append(np.load(dest))
        rec[arm]=np.concatenate(vals);rec[arm+"_geom"]=geom[arm]
        print(stem,arm,flush=True)
    np.savez_compressed(OUT/(stem+"_rotation.npz"),ids=ids,row=r,col=c,**rec)


def score(size):
    stem=f"{size}_{SEEDS[0]}_to_{size}"
    with np.load(OUT/(stem+"_rotation.npz")) as f:d=dict(f)
    with np.load(OUT/(stem+"_forecast.npz")) as f:e=dict(f)
    with np.load(OUT/(stem+"_group_waves.npz")) as f:real=f["sixteen_native"]
    np.testing.assert_array_equal(d["ids"],e["ids"])
    result={}
    for dom in np.unique(e["split"]):
        p=Panel(e["z"]>0,e["row"],e["col"],e["split"]==dom)
        cell=dict(n=int(p.keep.sum()),events=int(p.events.sum()),median_geometric_r2={arm:float(np.median(d[arm+"_geom"][p.keep])) for arm in ARMS})
        for part,sl in (("early",slice(8)),("late",slice(8,16))):
            for kind in ("full","shape"):
                ref=component(e["truth_native"][:,sl],kind)
                est={k:p.estimate(component(v[:,sl],kind),ref) for k,v in {"trained":real,**{a:d[a] for a in ARMS}}.items()}
                block={k:entry(v) for k,v in est.items()}
                for arm in ARMS:block["trained_minus_"+arm]=difference(est["trained"],est[arm])
                cell[part+"_"+kind]=block
        result[str(dom)]=cell
    (OUT/(stem+"_rotation.json")).write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("run","score"));p.add_argument("size",choices=("small","large"))
    p.add_argument("--batch",type=int,default=16);a=p.parse_args();torch.set_num_threads(2)
    run(a.size,a.batch) if a.stage=="run" else score(a.size)
