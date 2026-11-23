"""Reconstruct the registered finite ramp edits from raw inputs with the official MOMENT API alone."""
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from momentfm import MOMENTPipeline

HERE=Path(__file__).resolve().parent
MOMENT=HERE.parents[2]
CHECKPOINTS=Path(os.environ.get("TSFM_MOMENT_CHECKPOINTS","/tmp/tsfm_moment_native"))
SCRATCH=Path(os.environ.get("TSFM_MOMENT_GEOMETRY","/tmp/tsfm_moment_nov22_geometry"))/"official_input_ramp_replay"
ARMS=("ramp_reference","ramp_correct","ramp_wrong_step","ramp_wrong_order","permuted_reference","permuted_correct","permuted_swapped")


def call(model, raw, *, replace=None, add=None, observe=False):
    width, patch=model.config.seq_len,model.config.patch_len
    last=(width-16)//patch-1
    assert raw.shape[1]==480 and last==61
    padded=np.zeros((len(raw),width),dtype=np.float32)
    mask=np.zeros((len(raw),width),dtype=np.int64)
    padded[:,-480:]=raw; mask[:,-480:]=1
    seen=[]
    blocks=model.encoder.block
    def intercept(_, inputs, out):
        h=out[0]
        if observe:
            seen.append(h[:,last].detach().numpy().copy())
        if replace is None and add is None:
            return None
        assert (replace is None) != (add is None)
        z=h.clone()
        if replace is not None:
            z[:,last]=torch.from_numpy(replace).to(h.dtype)
        else:
            z[:,last]+=torch.from_numpy(add).to(h.dtype)
        return (z, *out[1:])
    hook=blocks[len(blocks)//2-1].register_forward_hook(intercept)
    try:
        result=model.short_forecast(x_enc=torch.from_numpy(padded)[:,None,:],input_mask=torch.from_numpy(mask),
                                    forecast_horizon=16).forecast[:,0].detach().numpy()
    finally:
        hook.remove()
    return result,seen[0] if observe else None


def native(size):
    return MOMENTPipeline.from_pretrained(str(CHECKPOINTS/size),strict=True,model_kwargs={"task_name":"reconstruction"}).eval()


def scale(v, other):
    return v*(np.sqrt((other*other).sum(1))/np.sqrt((v*v).sum(1)))[:,None]


def one_station(ids, x, delta, w, small, large):
    ramp=x.copy();ramp[:,-8:]+=delta
    order=x.copy();order[:,-8:]+=delta[[0,6,5,4,3,2,1,7]]
    step=x.copy();step[:,-8:]+=2
    _,s0=call(small,x,observe=True)
    maps={}
    for k,z in (("ramp",ramp),("order",order),("step",step)):
        _,s=call(small,z,observe=True)
        maps[k]=(s-s0).astype(float)@w
    orig,_=call(large,x)
    _,hr=call(large,ramp,observe=True)
    _,hp=call(large,order,observe=True)
    writes=(("ramp_reference","replace",hr),("ramp_correct","add",maps["ramp"]),
            ("ramp_wrong_step","add",scale(maps["step"],maps["ramp"])),
            ("ramp_wrong_order","add",scale(maps["order"],maps["ramp"])),
            ("permuted_reference","replace",hp),("permuted_correct","add",maps["order"]),
            ("permuted_swapped","add",scale(maps["ramp"],maps["order"])))
    out={}
    for name,how,z in writes:
        y,_=call(large,x,**{how:z})
        out[name]=y-orig
    return out


def main():
    torch.set_num_threads(2)
    panel=MOMENT/"horizon_prefix/new_dates/results/panel.npz"
    with np.load(panel) as f:
        keep=f["cohort"]=="secondary";x=f["x"][keep];ids=f["ids"][keep]
    target=HERE/"results";order=HERE.parent/"order/results"
    mapfile=HERE.parents[1]/"results/small_to_large_map.npz"
    with np.load(mapfile) as f:w=f["w"].astype(float)
    delta=np.linspace(-2,2,8)
    signature=sha256(x.tobytes()+w.tobytes()+Path(__file__).read_bytes()).hexdigest()
    cache=SCRATCH/signature
    cache.mkdir(parents=True,exist_ok=True)
    small,large=native("small"),native("large")
    assert len(small.encoder.block)==8 and len(large.encoder.block)==24
    collected={k:[] for k in ARMS}
    with torch.inference_mode():
        for i,station in enumerate(ids):
            save=cache/(str(station)+".npz")
            if not save.exists():
                part=one_station(station,x[i],delta,w,small,large)
                tmp=save.with_suffix(".tmp.npz")
                np.savez_compressed(tmp,**part)
                tmp.replace(save)
            with np.load(save) as f:
                for k in ARMS:collected[k].append(f[k])
            if (i+1)%5==0: print(i+1,flush=True)
    files=(target/"reference.npz",target/"mapped.npz",target/"wrong_edit.npz",order/"ramp_swapped.npz",
           order/"permuted_reference.npz",order/"permuted_correct.npz",order/"permuted_swapped.npz")
    checks={}
    for arm,archive in zip(ARMS,files):
        fresh=np.concatenate(collected[arm])
        with np.load(archive) as f:
            np.testing.assert_array_equal(f["ids"],ids)
            diff=fresh.astype(float)-f["wave"]
        assert fresh.shape==(385,16) and np.isfinite(fresh).all()
        err=float(np.max(abs(diff)))
        assert err<.00003,(arm,err)
        checks[arm]=dict(scalars=int(fresh.size),maximum_absolute_C=err,rms_difference_C=float(np.sqrt((diff**2).mean())))
    out=dict(models=["small","large"],stations=len(ids),origins=x.shape[1],model_api="MOMENTPipeline.short_forecast",
             clean_recipients_per_arm=385,whole_graph=True,reference_write="absolute independently measured state",
             source_and_map_fingerprint=signature,checks=checks)
    (target/"official_replay.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
