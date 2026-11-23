"""Readout repeatability on real ablations and norm-matched differences of ordinary histories."""
import argparse
import json
import numpy as np
from research.activation_transfer.moment.head_test.data import SCRATCH as READOUT
from research.activation_transfer.moment.head_test.fit import ridge
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.head_interventions.headbank import HERE

SPLITS=tuple(range(22112280,22112288))
TYPES=("removal","neighbor","deranged","isotropic")


def fit(size):
    with np.load(READOUT/f"{size}_0/interior.npz") as f:h,y,ids,split=f["h"],f["y"],f["ids"],f["split"]
    tr=np.flatnonzero(split=="train");cal=split=="cal";d=h.shape[-1]
    hs=h[tr].reshape(-1,d).astype(float);hs-=hs.mean(0)
    eig,q=np.linalg.eigh(hs.T@hs)
    out=dict(ids=ids,pcs=q[:,-128:][:,::-1].astype("f4"),eig=eig[::-1])
    for seed in SPLITS:
        rng=np.random.default_rng(seed);order=rng.permutation(tr)
        for half,row in enumerate(np.array_split(order,2)):
            w,b,meta=ridge(h[row].reshape(-1,d),y[row].reshape(-1,8),h[cal].reshape(-1,d),y[cal].reshape(-1,8))
            k=f"s{seed}_{half}"
            out[k]=w;out[k+"_bias"]=b;out[k+"_train"]=row;out[k+"_penalty"]=meta["relative"]
    dest=HERE/"results"/f"{size}_repeated_heads.npz";np.savez_compressed(dest,**out)


def cov_fraction(v,pc):
    proj=v@pc
    den=np.sum(v*v,axis=(1,2))
    return {str(k):float(np.sum(proj[:,:,:k]**2)/den.sum()) for k in (8,32,128)}


def run(size,seed):
    with np.load(HERE/"results"/f"{size}_repeated_heads.npz") as f:heads=dict(f)
    x,ids,domains=histories(False)
    with np.load(CACHE/size/"eval.npz") as f:base=f["readout"][:,1:].astype(float)
    with np.load(CACHE/size/f"ablation_{seed}.npz") as f:
        row,col,delta,sd,codes,atoms=f["row"],f["col"],f["dstate"].astype(float),f["sd"],f["codes"],f["atoms"]
        np.testing.assert_array_equal(f["ids"],ids)
    rng=np.random.default_rng(22112292)
    neighbor=np.arange(len(ids));deranged=neighbor.copy()
    z=(x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
    for domain in np.unique(domains):
        ix=np.flatnonzero(domains==domain)
        dist=np.mean((z[ix,None]-z[ix][None])**2,axis=2);np.fill_diagonal(dist,np.inf)
        neighbor[ix]=ix[np.argmin(dist,axis=1)]
        p=rng.permutation(ix);deranged[p]=np.roll(p,1)
    assert (neighbor!=np.arange(len(ids))).all() and (deranged!=np.arange(len(ids))).all()
    scale=np.linalg.norm(delta,axis=(1,2))
    def equal_norm(v):return v*(scale/np.linalg.norm(v,axis=(1,2)))[:,None,None]
    shifts=dict(removal=delta,neighbor=equal_norm(base[row]-base[neighbor[row]]),
                deranged=equal_norm(base[row]-base[deranged[row]]),isotropic=equal_norm(rng.normal(size=delta.shape)))
    summary={};raw={}
    for domain in np.unique(domains):
        eligible=np.flatnonzero((codes[domains==domain]>0).sum(0)>=8)
        live=(domains[row]==domain)&np.isin(col,eligible)
        result={}
        for typ,d in shifts.items():
            np.testing.assert_allclose(np.linalg.norm(d,axis=(1,2)),scale,atol=1e-9)
            rec=dict(top_train_pc_energy=cov_fraction(d[live],heads["pcs"]),
                     median_hidden_norm=float(np.median(scale[live])),events=int(live.sum()),atoms=atoms[eligible].tolist())
            for patch,idx in (("early",0),("late",1)):
                split_metrics=[];atom_metrics=[]
                for split_seed in SPLITS:
                    pair=[(d[:,idx]@heads[f"s{split_seed}_{h}"])*sd[:,None] for h in (0,1)]
                    w1,w2=[v-v.mean(1,keepdims=True) for v in pair]
                    err=(w1-w2)**2;energy=(w1**2+w2**2)/2
                    summaries=[]
                    for j in eligible:
                        active=live&(col==j)
                        summaries.append([1-float(err[active].sum()/max(1e-30,energy[active].sum())),
                                          np.sqrt(err[active].mean()),np.sqrt(energy[active].mean())])
                    values=np.asarray(summaries);atom_metrics.append(values)
                    macro=values.mean(0)
                    split_metrics.append([*macro,1-float(err[live].sum()/energy[live].sum())])
                vals=np.asarray(split_metrics)
                rec[patch]={key:dict(median=float(np.median(vals[:,i])),minimum=float(vals[:,i].min()),maximum=float(vals[:,i].max()))
                            for i,key in enumerate(("symmetric_agreement","difference_rms","wave_rms","pooled_agreement"))}
                raw[str(domain)+"_"+typ+"_"+patch]=np.stack(atom_metrics)
            result[typ]=rec
        summary[str(domain)]=result
    loc=HERE/"results"/f"{size}_{seed}_repeated"
    np.savez_compressed(loc.with_suffix(".npz"),neighbor=neighbor,deranged=deranged,seeds=SPLITS,**raw)
    loc.with_suffix(".json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("fit","score"));p.add_argument("size",choices=("small","large"))
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0]);a=p.parse_args()
    if a.stage=="fit":fit(a.size)
    else:run(a.size,a.seed)
