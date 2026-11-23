"""Read the already computed eight-value native reconstruction at each selected patch."""
import argparse
from contextlib import contextmanager
import json

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import SEED, OUT as DIRECTIONS, describe, read_views, sample
from research.activation_transfer.moment.token_paths.geometry import load, layers

OUT=DIRECTIONS.parent/"readout_slots"
SLOTS=(59,60,61,62,63)


@contextmanager
def patch(adapter,slot,changes):
    """Apply one raw-unit displacement per item to a chosen native middle-block slot."""
    delta=torch.as_tensor(changes,dtype=torch.float32)
    def add(_,args,output):
        h=output[0].clone()
        h[:,slot]+=delta.to(h.device)
        return (h,*output[1:])
    hook=adapter.module.get_submodule(layers(adapter)["middle"]).register_forward_hook(add)
    try: yield
    finally: hook.remove()


@torch.inference_mode()
def native_slots(adapter,x,*,capture_slot=None):
    """Return raw-unit [N,5,8] slots 59..63 and optional [N,D] clean middle state.

    Slots 62 and 63 concatenate to the native H16 forecast. No head is fitted or replaced.
    """
    past,mask=adapter.prepare(x)
    site=layers(adapter)["middle"]
    state=[]
    def capture(_,args,output):
        state.append(output[0][:,capture_slot].detach().cpu().numpy().copy())
    hook=None if capture_slot is None else adapter.module.get_submodule(site).register_forward_hook(capture)
    try: out=adapter.module.short_forecast(x_enc=past,input_mask=mask,forecast_horizon=16)
    finally:
        if hook is not None: hook.remove()
    recon=out.reconstruction[:,0].reshape(len(x),64,8)
    torch.testing.assert_close(recon[:,62:].reshape(len(x),16),out.forecast[:,0],rtol=0,atol=0)
    return recon[:,SLOTS].cpu().numpy(),state[0] if state else None


def deltas(adapter,x,slot,moves,batch=16):
    """Return baseline slots, requested middle state, and each genuine finite response."""
    outputs={key:[] for key in moves}
    base=[]; states=[]
    for i in range(0,len(x),batch):
        ix=slice(i,i+batch)
        b,state=native_slots(adapter,x[ix],capture_slot=slot)
        base.append(b); states.append(state)
        for name,d in moves.items():
            with patch(adapter,slot,d[ix]): new,_=native_slots(adapter,x[ix])
            outputs[name].append(new-b)
    return np.concatenate(base), np.concatenate(states), {k:np.concatenate(v) for k,v in outputs.items()}


def source_events(model,means,hs,pick,other):
    with torch.inference_mode():
        z=cd.encode_views(model,means,hs).numpy()
    assert np.all((z>0).sum(1)==cd.TOPK)
    support=np.where(z>0)[1].reshape(len(z),cd.TOPK)
    row=np.arange(len(z));atoms=support[row,pick];alters=support[row,other]
    assert np.all(atoms!=alters)
    return atoms,alters,z[row,atoms]


def writes(ds,dl,atoms,alters,strength):
    a = -strength[:,None]
    other=dl[alters]*(np.linalg.norm(dl[atoms],axis=1)/np.linalg.norm(dl[alters],axis=1))[:,None]
    edits={"small":a*ds[atoms],"paired":a*dl[atoms],"decoy":a*other}
    np.testing.assert_allclose(np.linalg.norm(edits["paired"],axis=1),np.linalg.norm(edits["decoy"],axis=1),rtol=2e-6)
    return edits


def per_history(waves):
    """Return cosine for eight-value shapes; preserve undefined rows for explicit coverage."""
    ref=waves["small"].astype(float)
    ref-=ref.mean(-1,keepdims=True)
    out={}
    for name in ("paired","decoy"):
        v=waves[name].astype(float)
        v-=v.mean(-1,keepdims=True)
        den=np.linalg.norm(ref,axis=-1)*np.linalg.norm(v,axis=-1)
        out[name]=np.divide((ref*v).sum(-1),den,out=np.full_like(den,np.nan),where=den>1e-20)
    return out


def collect(method,seed,slot,adapters,x,ids,domains,pick,other,hs):
    model,means=cd.load_artifact(method,seed)
    with np.load(cd.CACHE/f"{method}_{seed}.npz") as f:
        ds,dl=f["decoder_small"],f["decoder_large"]
    atoms,alters,strength=source_events(model,means,hs,pick,other)
    moves=writes(ds,dl,atoms,alters,strength)
    bs,actual,src=deltas(adapters["small"],x,slot,{"small":moves["small"]},32)
    np.testing.assert_allclose(actual,hs,rtol=2e-5,atol=2e-5)
    bl,_,tgt=deltas(adapters["large"],x,slot,{k:moves[k] for k in ("paired","decoy")},16)
    waves={**src,**tgt}
    if slot==61 and method!="shuffled":
        ref=DIRECTIONS/f"finite_{method}_{seed}.npz"
        if ref.exists():
            with np.load(ref) as f:
                np.testing.assert_array_equal(f["ids"],ids)
                np.testing.assert_array_equal(f["atom"],atoms);np.testing.assert_array_equal(f["decoy_atom"],alters)
                for a,b in (("small","small"),("paired","large"),("decoy","decoy")):
                    np.testing.assert_allclose(waves[a][:,-2:].reshape(len(ids),16),f[b],rtol=.0001,atol=2e-5)
    cos=per_history(waves)
    summary={}
    sd=x.std(1)
    for domain in np.unique(domains):
        ix=np.flatnonzero(domains==domain)
        def c(y):
            z=y[ix].astype(float)/sd[ix,None,None]
            return z-z.mean(-1,keepdims=True)
        centered={k:c(v) for k,v in waves.items()}
        ref=centered["small"]
        entry={}
        for j,j_slot in enumerate(SLOTS):
            valid=np.isfinite(cos["paired"][ix,j]) & np.isfinite(cos["decoy"][ix,j])
            sel=ix[valid]
            d=cos["paired"][sel,j]-cos["decoy"][sel,j]
            loc=dict(n=len(sel), paired=describe(cos["paired"][sel,j]),decoy=describe(cos["decoy"][sel,j]),
                paired_minus_decoy=describe(d),rms_history_sd={k:float(np.sqrt(np.mean(v[:,j]**2))) for k,v in centered.items()},
                waveform_skill={k:float(1-np.sum((v[:,j]-ref[:,j])**2)/np.sum(ref[:,j]**2)) for k,v in centered.items() if k!="small"})
            if j_slot<62:
                target=x[ix, -(62-j_slot)*8:(j_slot-61)*8 or None]
                assert target.shape==(len(ix),8)
                loc["clean_reconstruction_rmse_history_sd"]={k:float(np.sqrt(np.mean(((z[ix,j]-target)/sd[ix,None])**2))) for k,z in (("small",bs),("large",bl))}
            entry[str(j_slot)]=loc
        diff=cos["paired"][ix]-cos["decoy"][ix]
        home=SLOTS.index(slot)
        entry["edited_minus_immediate_future_advantage"]=describe(diff[:,home]-diff[:,SLOTS.index(62)])
        entry["edited_minus_next_slot_advantage"]=describe(diff[:,home]-diff[:,home+1])
        summary[str(domain)]=entry
    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/f"{method}_{seed}_edit{slot}"
    np.savez_compressed(stem.with_suffix(".npz"),ids=ids,domains=domains,atoms=atoms,decoy_atom=alters,dose=strength,
        slots=SLOTS,edit=slot,sd=sd,small_base=bs,large_base=bl,**waves)
    stem.with_suffix(".json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("method",choices=cd.METHODS);p.add_argument("--slot",type=int,choices=(60,61),default=61)
    args=p.parse_args(); torch.set_num_threads(2)
    x,ids,domains,states,_=read_views()
    rows,pick,other=sample(domains)
    adapters={s:load(s) for s in cd.SIZES}
    selected=np.concatenate([native_slots(adapters["small"],x[rows[i:i+16]],capture_slot=args.slot)[1] for i in range(0,len(rows),16)])
    if args.slot==61: np.testing.assert_allclose(selected,states["small"][rows],rtol=2e-5,atol=2e-5)
    for seed in cd.SEEDS:
        collect(args.method,seed,args.slot,adapters,x[rows],ids[rows],domains[rows],pick,other,selected)
