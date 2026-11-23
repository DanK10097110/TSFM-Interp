"""Run finite native slot-specific writes with checksum-bound ordinary prefix tapes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import ordinary_map
from research.activation_transfer.moment.crosscoder.evaluate import write_npz
from research.activation_transfer.moment.crosscoder.fresh_readout.panel import OUT
from research.activation_transfer.moment.crosscoder.readout_slots import native_slots, patch, source_events, writes
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle
from research.activation_transfer.moment.token_paths.geometry import layers, load

CACHE=Path("/tmp/tsfm_moment_crosscoder/fresh_readout")
BATCH=16
SLOT=61
SLOTS=np.array([60,61,62,63])


def sha(*arrays):
    h=sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).view("u1"))
    return h.hexdigest()


def sample(n):
    g=np.random.default_rng(22112433);pick=g.integers(0,cd.TOPK,n)
    alt=(pick+g.integers(1,cd.TOPK,n))%cd.TOPK
    return pick,alt


def ordinary(a,x,size):
    path=CACHE/(size+"_prefix.npz")
    signature=sha(x);site=layers(a)["middle"]
    if path.exists():
        with np.load(path) as f:
            assert str(f["signature"])==signature
            return f["hidden"],f["base"]
    states=[];base=[]
    def record(_,args,out):
        states.append(out[0].detach().cpu().numpy().copy())
    hook=a.module.get_submodule(site).register_forward_hook(record)
    try:
        for lo in range(0,len(x),BATCH):
            v,_=native_slots(a,x[lo:lo+BATCH]);base.append(v[:,1:])
    finally:hook.remove()
    hs,bs=np.concatenate(states),np.concatenate(base)
    with torch.inference_mode():
        point=a.predict(x[:BATCH],16,[.5])["point"]
    np.testing.assert_array_equal(bs[:BATCH,-2:].reshape(BATCH,16),point)
    assert hs.shape[:2]==(len(x),64)
    write_npz(path,hidden=hs,base=bs,signature=signature)
    print(size,"ordinary",len(bs),flush=True)
    return hs,bs


def edit(a,x,h,base,move,arm,signature):
    path=CACHE/(arm+".npz");signature=sha(np.frombuffer(signature.encode(),dtype="u1"),move)
    if path.exists():
        with np.load(path) as f:
            assert str(f["signature"])==signature
            return f["wave"]
    waves=[];site=layers(a)["middle"]
    for lo in range(0,len(x),BATCH):
        ix=slice(lo,lo+BATCH)
        with cached_middle(a,h[ix],site),patch(a,SLOT,move[ix]):
            v,_=native_slots(a,x[ix])
        waves.append(v[:,1:]-base[ix])
    wave=np.concatenate(waves);assert np.isfinite(wave).all()
    write_npz(path,wave=wave,signature=signature)
    print(arm,"complete",len(wave),flush=True)
    return wave


def direct_check(a,x,hs,base,waves,moves,indices):
    site=layers(a)["middle"]
    ordinary_full,state=native_slots(a,x[indices],capture_slot=SLOT)
    np.testing.assert_allclose(state,hs[indices,SLOT],rtol=4e-5,atol=4e-5)
    clean=ordinary_full[:,1:]
    disagreement={"ordinary":float(np.max(abs(clean-base[indices])))}
    for arm,d in moves.items():
        # Complete graph only; no cached states or overridden block evaluation in this replay.
        with patch(a,SLOT,d[indices]):
            changed,_=native_slots(a,x[indices])
        actual=changed[:,1:]-clean
        disagreement[arm]=float(np.max(abs(actual-waves[arm][indices])))
    with cached_middle(a,hs[indices],site),patch(a,SLOT,np.zeros_like(hs[indices,SLOT])):
        zero,_=native_slots(a,x[indices])
    disagreement["zero_cached_vs_direct"]=float(np.max(abs(zero[:,1:]-clean)))
    assert max(disagreement.values())<.0002,disagreement
    return disagreement


def main(methods):
    torch.set_num_threads(2);CACHE.mkdir(parents=True,exist_ok=True)
    with np.load(OUT/"panel.npz") as f:
        x=f["x"].reshape(-1,480);ids,dates=f["ids"],f["dates"]
    assert str(sha(x))==json.loads((OUT/"manifest.json").read_text())["raw_input_sha256"]
    adapters={s:load(s) for s in cd.SIZES}
    h,b={},{}
    for s in cd.SIZES:h[s],b[s]=ordinary(adapters[s],x,s)
    write_npz(OUT/"ordinary.npz",ids=ids,dates=dates,slots=SLOTS,small=b["small"],large=b["large"])
    pick,alt=sample(len(x));w=ordinary_map()
    for method in methods:
        for seed in cd.SEEDS:
            stem=f"{method}_{seed}"
            with np.load(cd.CACHE/(stem+".npz")) as f:
                ds,dl=f["decoder_small"],f["decoder_large"]
            model,means=cd.load_artifact(method,seed)
            atoms,other,dose=source_events(model,means,h["small"][:,SLOT],pick,alt)
            moves=writes(ds,dl,atoms,other,dose)
            vm=ds[atoms].astype(float)@w
            assert np.all(np.linalg.norm(vm,axis=1)>0)
            vm*=np.linalg.norm(dl[atoms],axis=1,keepdims=True)/np.linalg.norm(vm,axis=1,keepdims=True)
            moves["dense"]=(-dose[:,None]*vm).astype("f4")
            sizes=np.stack([np.linalg.norm(moves[k],axis=1) for k in ("small","paired","decoy","dense")],axis=1)
            np.testing.assert_allclose(sizes[:,1],sizes[:,3],rtol=2e-6)
            waves={};signature=sha(x,atoms,other,dose)
            for arm,move in moves.items():
                size="small" if arm=="small" else "large"
                waves[arm]=edit(adapters[size],x,h[size],b[size],move,stem+"_"+arm,signature)
            indices=np.array([1,57,121,184,240,287])
            errors={}
            for s,names in (("small",["small"]),("large",["paired","decoy","dense"])):
                errors[s]=direct_check(adapters[s],x,h[s],b[s],waves,{k:moves[k] for k in names},indices)
            write_npz(OUT/(stem+".npz"),ids=ids,dates=dates,slots=SLOTS,source_atom=atoms,other_atom=other,
                      dose=dose,write_norms=sizes,write_order=np.array(["small","paired","decoy","dense"]),
                      input_sha256=sha(x),replay_rows=indices,replay_error_json=json.dumps(errors),**waves)
            print(stem,"published",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("method",nargs="+",choices=cd.METHODS)
    args=p.parse_args();main(args.method)
