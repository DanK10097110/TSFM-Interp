"""Test decoder pairing against a measurable input quantity using two independently fit linear probes."""
import json
import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.mechanism import SPLIT,OUT,map_inputs,sources,large_parts
from research.activation_transfer.moment.crosscoder.directions import ordinary_map
from research.activation_transfer.moment.dictionary_transfer.match import hidden
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import WORK
from research.activation_transfer.moment.token_paths.geometry import CACHE

DEST=OUT/"patch_probe"
BOOT=2500


def small_endpoints():
    with np.load(CACHE/"small"/"train.npz") as tr, np.load(CACHE/"small"/"eval.npz") as ev, np.load(
        WORK/"physical_ramp_confirmation"/"small_base.npz") as we:
        return np.concatenate((tr["middle"][:,0],ev["middle"][:,0],we["arr"]))


def decomposition(a):
    level=a.mean(-1,keepdims=True)
    return {"full":a,"shape":a-level,"level":level}


def fit(size,fit_on,xt,yt,indices):
    endpoint=small_endpoints() if size=="small" else large_parts()
    if fit_on=="all":
        states=hidden(size)[:SPLIT].reshape(-1,endpoint.shape[-1])
        labels=xt[:SPLIT].reshape(-1,8)
    else:
        states=endpoint[:SPLIT]
        labels=yt[:SPLIT]
    q=np.vstack([endpoint, np.eye(endpoint.shape[1]),np.zeros(endpoint.shape[1])])
    best=np.inf
    for lam,p in map_inputs(states,labels,q):
        loss=np.mean((p[SPLIT:1024]-yt[SPLIT:1024])**2)
        if loss<best:
            best,pen,pred=loss,lam,p
    w=pred[len(endpoint):-1]-pred[-1]
    np.testing.assert_allclose(endpoint@w+pred[-1],pred[:len(endpoint)],atol=1e-9)
    summ=dict(relative=pen,validation_mse=best,panels={})
    for domain,ix in indices.items():
        ys=decomposition(yt[ix])
        estimate=decomposition(pred[ix])
        center=decomposition(labels.mean(axis=0,keepdims=True))
        stat={}
        for measure,y in ys.items():
            err=np.mean((estimate[measure]-y)**2)
            stat[measure]=dict(rmse=float(np.sqrt(err)),r2=float(1-err/np.mean((y-center[measure])**2)))
        summ["panels"][domain]=stat
    return w,summ


def compare(s,t,ids):
    sa=s[ids];tb=t[ids]
    def unit(a):
        a=a-a.mean(-1,keepdims=True)
        length=np.linalg.norm(a,axis=-1,keepdims=True)
        assert np.min(length)>1e-12
        return a/length
    a,b=unit(sa),unit(tb)
    same=(a*b).sum(-1)
    off=((a*b.sum(1,keepdims=True)).sum(-1)-same)/(a.shape[1]-1)
    sl=np.sign(sa.mean(-1));tl=np.sign(tb.mean(-1))
    assert np.all(sl!=0) and np.all(tl!=0)
    sign=sl==tl
    offsign=((ids.shape[1]+sl*tl.sum(1,keepdims=True))/2-sign)/(ids.shape[1]-1)
    return dict(shape=same,other_shape=off,level_sign=sign.astype(float),other_level_sign=offsign)


def crossed(arm,atoms,items,weather):
    """Paired-weight bootstrap of history blocks and atom IDs, conditional on fitted probes."""
    g=np.random.default_rng(22112475)
    n,k=atoms.shape
    unique,idx=np.unique(atoms,return_inverse=True)
    if not weather:
        rows=g.multinomial(n,np.full(n,1/n),BOOT)
    else:
        stations,dates=items
        a=g.multinomial(stations,np.full(stations,1/stations),BOOT)
        b=g.multinomial(dates,np.full(dates,1/dates),BOOT)
        rows=(a[:,:,None]*b[:,None,:]).reshape(BOOT,n)
    counts=g.multinomial(len(unique),np.full(len(unique),1/len(unique)),BOOT)
    flat=idx.reshape(-1);cell=np.repeat(np.arange(n),k)*len(unique)+flat
    countmat=np.bincount(cell,minlength=n*len(unique)).reshape(n,-1)
    denom=np.einsum("bi,bi->b",rows@countmat,counts)
    out={}
    for name,z in arm.items():
        mat=np.bincount(cell,weights=z.ravel(),minlength=n*len(unique)).reshape(n,-1)
        draws=np.einsum("bi,bi->b",rows@mat,counts)/denom
        per=mat.sum(0)/countmat.sum(0)
        out[name]=dict(mean=float(z.mean()),history_atom_95=np.quantile(draws,[.025,.975]).tolist(),
            equal_atom=float(np.mean(per)),n_atoms=len(unique),atoms_at_least_8=int((countmat.sum(0)>=8).sum()),
            equal_atom_min8=float(per[countmat.sum(0)>=8].mean()))
    return out


def run():
    xs,ids,domains,stations,dates=sources()
    x=np.concatenate(xs).astype(float)
    x=(x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
    indices={"validation":np.arange(SPLIT,1024),"daily":1024+np.flatnonzero(domains=="daily"),
             "traffic":1024+np.flatnonzero(domains=="monash_traffic"),"weather":np.arange(1280,len(x))}
    y=x[:,-8:]
    small=small_endpoints()
    mapping=ordinary_map()
    report={};DEST.mkdir(parents=True,exist_ok=True)
    for fit_on in ("all","last"):
        stats={};probe={}
        for size in cd.SIZES:
            probe[size],stats[size]=fit(size,fit_on,x,y,indices)
        report[fit_on]=stats
        np.savez_compressed(DEST/f"probe_{fit_on}.npz",small_weight=probe["small"],large_weight=probe["large"],
            train_ids=ids[:SPLIT],validation_ids=ids[SPLIT:1024])
        for method in cd.METHODS:
            for seed in cd.SEEDS:
                model,mu=cd.load_artifact(method,seed)
                with torch.inference_mode():
                    z=cd.encode_views(model,mu,small).numpy()
                assert np.all((z>0).sum(1)==cd.TOPK)
                atoms=np.where(z>0)[1].reshape(len(small),cd.TOPK)
                dose=np.take_along_axis(z,atoms,axis=1)
                with np.load(cd.CACHE/f"{method}_{seed}.npz") as f: ds,dt=f["decoder_small"],f["decoder_large"]
                raw=ds@mapping
                raw*=np.linalg.norm(dt,axis=1,keepdims=True)/np.linalg.norm(raw,axis=1,keepdims=True)
                source=ds@probe["small"]
                target=dt@probe["large"]
                dense=raw@probe["large"]
                aligned=compare(source,target,atoms)
                mapped=compare(source,dense,atoms)
                obs=dict(**aligned,dense_shape=mapped["shape"],dense_level_sign=mapped["level_sign"])
                obs["shape_advantage"]=obs["shape"]-obs["other_shape"]
                obs["shape_over_dense"]=obs["shape"]-obs["dense_shape"]
                obs["sign_advantage"]=obs["level_sign"]-obs["other_level_sign"]
                outputs={}
                for domain,ix in indices.items():
                    if domain=="validation":continue
                    cell=crossed({k:v[ix] for k,v in obs.items()},atoms[ix],(len(stations),len(dates)),domain=="weather")
                    delta_s=dose[ix,:,None]*source[atoms[ix]];delta_t=dose[ix,:,None]*target[atoms[ix]]
                    sa,tb=decomposition(delta_s),decomposition(delta_t)
                    cell["magnitude"]={}
                    for key,a in sa.items():
                        b=tb[key]
                        cell["magnitude"][key]=dict(small_rms=float(np.sqrt(np.mean(a*a))),large_rms=float(np.sqrt(np.mean(b*b))),
                            wave_skill=float(1-np.sum((a-b)**2)/np.sum(a*a)))
                    outputs[domain]=cell
                (DEST/f"{fit_on}_{method}_{seed}.json").write_text(json.dumps(outputs,indent=2,allow_nan=False)+"\n")
                np.savez_compressed(DEST/f"{fit_on}_{method}_{seed}.npz",source=source,target=target,dense=dense,
                    ids=ids[1024:],atoms=atoms[1024:],dose=dose[1024:])
    (DEST/"clean_probes.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
