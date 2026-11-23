"""Follow the same finite middle-token edit through three depths and two final readouts."""
import argparse
from contextlib import nullcontext
import json

import numpy as np
import torch

from tsfm_lens.extraction.hooks import ActivationCatcher
from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import describe,sample
from research.activation_transfer.moment.crosscoder.mechanism import OUT,SPLIT,map_inputs,SCRATCH
from research.activation_transfer.moment.crosscoder.patch_probe import decomposition
from research.activation_transfer.moment.crosscoder.readout_slots import native_slots,patch,writes,OUT as FINITE
from research.activation_transfer.moment.token_paths.geometry import load,histories,layers

DEST=OUT/"depth_trace"
DEPTHS=("middle","upper","final")


def sites(a):
    n=len(a.module.encoder.block)
    return dict(middle=layers(a)["middle"], upper=f"encoder.block.{3*n//4-1}", final=layers(a)["readout"])


def prepare(size):
    a=load(size)
    names=sites(a)
    x,i,_=histories(True); xe,e,dom=histories(False)
    rows=np.r_[x,xe]
    scaled=rows.astype(float)
    scaled=(scaled-scaled.mean(1,keepdims=True))/scaled.std(1,keepdims=True)
    dim=a.module.config.d_model
    root=SCRATCH/"depth_probe";root.mkdir(exist_ok=True,parents=True)
    states={key:np.lib.format.open_memmap(root/f"{size}_{key}.npy",mode="w+",dtype="f4",shape=(SPLIT,60,dim)) for key in ("upper","final")}
    end={key:np.empty((len(rows),dim),"f4") for key in DEPTHS}
    for lo in range(0,len(rows),16):
        ix=slice(lo,lo+16)
        with ActivationCatcher(a.module,list(names.values())) as cap:
            a.predict(rows[ix],16,[.5])
            found=cap.collect()
        for key,site in names.items():
            act=found[site].numpy()
            end[key][ix]=act[:,61]
            if lo<SPLIT and key in states:
                states[key][ix]=act[:,2:62]
    weights={}; meta={}
    for key,xlast in end.items():
        labels=scaled[:SPLIT].reshape(-1,8)
        if key=="middle":
            with np.load(OUT/"patch_probe"/"probe_all.npz") as f:
                w=f[size+"_weight"]
            from research.activation_transfer.moment.dictionary_transfer.match import hidden
            train=hidden(size)[:SPLIT].reshape(-1,dim)
            b=labels.mean(0)-train.mean(0,dtype=float)@w
            pen=None
        else:
            train=states[key].reshape(-1,dim)
            query=np.r_[xlast[SPLIT:1024],np.eye(dim),np.zeros((1,dim))]
            ymin=np.inf
            for lam,p in map_inputs(train,labels,query):
                loss=np.mean((p[:256]-scaled[SPLIT:1024,-8:])**2)
                if loss<ymin:
                    ymin=loss;pen=lam;b=p[-1].copy();w=p[256:-1]-b
        weights[key]=w;weights[key+"_bias"]=b
        preds=xlast@w+b
        vals={}
        for d in np.unique(dom):
            ix=1024+np.flatnonzero(dom==d)
            y=scaled[ix,-8:]
            aa,bb,cc=decomposition(y),decomposition(preds[ix]),decomposition(labels.mean(0))
            vals[str(d)]={k:float(1-np.sum((aa[k]-bb[k])**2)/np.sum((aa[k]-cc[k])**2)) for k in aa}
        meta[key]=dict(native_site=names[key],relative=pen,clean_R2=vals)
    DEST.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(DEST/f"{size}_probes.npz",**weights,train_ids=i[:SPLIT],val_ids=i[SPLIT:])
    (DEST/f"{size}_clean.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")


def states_and_output(a,names,x,move=None):
    context=nullcontext() if move is None else patch(a,61,move)
    # Register the capture after the edit so the middle measurement includes the physical change.
    with context,ActivationCatcher(a.module,list(names.values())) as cap:
        read,_=native_slots(a,x)
        f=cap.collect()
    states={k:f[site][:,61].numpy() for k,site in names.items()}
    final=torch.tensor(states["final"],dtype=torch.float32)
    with torch.no_grad():
        h=a.module.head.linear(final)[:,None,:]
        native=a.module.normalizer(x=h,mode="denorm")[:,0].numpy()
    np.testing.assert_allclose(read[:,2],native,rtol=1e-5,atol=1.8e-5)
    return states,read[:,2],float(abs(read[:,2]-native).max())


def trace_one(a,names,x,probes,moves):
    store={k:{depth:[] for depth in (*DEPTHS,"native")} for k in moves}
    err=0.
    for lo in range(0,len(x),16):
        sl=slice(lo,lo+16)
        bs,bo,e=states_and_output(a,names,x[sl]);err=max(err,e)
        for arm,v in moves.items():
            es,eo,e=states_and_output(a,names,x[sl],v[sl]);err=max(err,e)
            for depth in DEPTHS:
                drift=(es[depth].astype(float)-bs[depth])
                if depth=="middle":np.testing.assert_allclose(drift,v[sl],rtol=1e-5,atol=7e-6)
                store[arm][depth].append(drift@probes[depth])
            store[arm]["native"].append((eo-bo)/x[sl].std(1,keepdims=True))
    return {a+"_"+b:np.concatenate(v) for a,h in store.items() for b,v in h.items()},err


def run(method):
    x,ids,dom=histories(False)
    rows,_,_=sample(dom)
    x=x[rows];ids=ids[rows];dom=dom[rows]
    models={k:load(k) for k in cd.SIZES}
    probes={}
    for size in cd.SIZES:
        with np.load(DEST/f"{size}_probes.npz") as f:probes[size]={k:f[k] for k in DEPTHS}
    for seed in cd.SEEDS:
        with np.load(FINITE/f"{method}_{seed}_edit61.npz") as f:
            old=dict(f)
            np.testing.assert_array_equal(old["ids"],ids)
        with np.load(cd.CACHE/f"{method}_{seed}.npz") as f:
            moves=writes(f["decoder_small"],f["decoder_large"],old["atoms"],old["decoy_atom"],old["dose"])
        waves={};checks={}
        for size,a in models.items():
            names=("small",) if size=="small" else ("paired","decoy")
            v,checks[size]=trace_one(a,sites(a),x,probes[size],{k:moves[k] for k in names})
            waves.update(v)
        for arm in ("small","paired","decoy"):
            np.testing.assert_allclose(waves[arm+"_native"]*x.std(1,keepdims=True),old[arm][:,2],rtol=.0001,atol=2e-5)
        out={"native_reconstruction_max_error":checks}
        for domain in np.unique(dom):
            ix=dom==domain
            local={};raw_differences={}
            for d in (*DEPTHS,"native"):
                measure={k:decomposition(waves[k+"_"+d][ix].astype(float))["shape"] for k in ("small","paired","decoy")}
                src=measure["small"];vals={}
                for k,v in measure.items():
                    length=np.linalg.norm(src,axis=1)*np.linalg.norm(v,axis=1)
                    assert np.min(length)>1e-20
                    vals[k]=(src*v).sum(1)/length
                raw_differences[d]=vals["paired"]-vals["decoy"]
                local[d]=dict(paired=describe(vals["paired"]),decoy=describe(vals["decoy"]),
                    advantage=describe(raw_differences[d]),rms={k:float(np.sqrt(np.mean(v*v))) for k,v in measure.items()})
            local["final_probe_minus_native_advantage"]=describe(raw_differences["final"]-raw_differences["native"])
            out[str(domain)]=local
        np.savez_compressed(DEST/f"{method}_{seed}.npz",ids=ids,domains=dom,atoms=old["atoms"],**waves)
        (DEST/f"{method}_{seed}.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","run"));p.add_argument("name")
    args=p.parse_args();torch.set_num_threads(2)
    (prepare if args.stage=="prepare" else run)(args.name)
