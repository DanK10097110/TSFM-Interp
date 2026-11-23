"""Replace selected final TimesFM block functions by their ordinary-state local tangent."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.activation_transfer.atoms.materials import adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.feedback.data import OUT, SCRATCH
from research.activation_transfer.shape_transfer.depth.feedback.run import limit
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL, boost_shape, solve_adaptive
from tsfm_lens.extraction.hooks import token_patch

ATTN = ("pre_attn_ln", "attn.query_ln", "attn.key_ln", "post_attn_ln")
FF = ("pre_ff_ln", "activation", "post_ff_ln")
ALL = ATTN + FF
ARMS = {k: (k,) for k in ALL}
ARMS.update(ordinary=(), attention_norms=ATTN, feedforward=FF,
            all_norms=tuple(k for k in ALL if k != "activation"), all=ALL)


def tangent(module, clean, baseline, live):
    dx = live - clean
    if isinstance(module, nn.Identity):
        return live
    if isinstance(module, nn.ReLU):
        return baseline + dx * (clean > 0)
    if isinstance(module, nn.SiLU):
        sig = clean.sigmoid()
        return baseline + dx * (sig + clean * sig * (1 - sig))
    if module.__class__.__name__ != "RMSNorm":
        raise TypeError(f"Unsupported last-block module {type(module)}")
    inv = torch.rsqrt(clean.square().mean(dim=-1, keepdim=True) + module.epsilon)
    radial = (clean * dx).mean(dim=-1, keepdim=True) * inv.square()
    return baseline + module.scale * inv * (dx - clean * radial)


def collect(out, scratch, batch):
    with np.load(scratch/"data.npz") as f:
        ids, split, x = f["ids"], f["split"], f["x"]
    with np.load(scratch/"bolt_3.npz") as s, np.load(scratch/"timesfm_10.npz") as t:
        k = int(np.flatnonzero(s["atoms"] == 5876)[0])
        z = s["codes"][:, k]
        used = np.flatnonzero(z > 0)
        sig = s["jac"][used].astype(float) @ s["dec"][:, k:k+1].astype(float)
        j = t["jac"][used].astype(float)
        h, layer, pos, rad = t["states"][used], str(t["layer"]), t["pos"], float(t["radius"])
    with np.load(FULL/"results/bolt.npz") as f:
        weight = float(f["shape_weight"])
    w = solve_adaptive(boost_shape(j,weight), boost_shape(sig,weight))[:,:,0]
    v = limit(-z[used,None]*w,.2*rad)
    x, ids = x[used], ids[used]
    enc = adapter_for("timesfm")
    sel = last_window_slice(enc,pos)
    block = enc.module.get_submodule(enc.all_layer_names()[-1])
    modules = {k:block.get_submodule(k) for k in ALL}
    rec = {k:[] for k in ARMS}
    max_clean = 0.
    for start in range(0,len(x),batch):
        ix = slice(start,start+batch)
        cache = {}
        def run(dose, capture=False, replacements=()):
            def hook(name):
                def patch(mod,args,y):
                    if capture:
                        if name in cache:
                            raise ValueError(f"Module called twice: {name}")
                        cache[name] = args[0].detach().clone(), y.detach().clone()
                    elif name in replacements:
                        return tangent(mod,*cache[name],args[0])
                return patch
            with ExitStack() as stack:
                state = torch.from_numpy(h[ix]+(dose*v[ix,None]).astype("f4"))
                stack.enter_context(token_patch(enc.module,layer,sel,state))
                for name in ALL if capture else replacements:
                    stack.enter_context(modules[name].register_forward_hook(hook(name)))
                return enc.predict(x[ix],8,[.5])["quantiles"][...,0].astype(float)
        clean = run(0.,capture=True)
        if set(cache) != set(ALL):
            raise ValueError("Unobserved module")
        max_clean = max(max_clean,float(np.abs(run(0.,replacements=ALL)-clean).max()))
        for label, names in ARMS.items():
            rec[label].append(run(1.,replacements=names)-clean)
    result = {k:np.concatenate(rows).astype("f4") for k,rows in rec.items()}
    with np.load(OUT/"norm_matched.npz") as old:
        np.testing.assert_array_equal(old["ids"],ids)
        np.testing.assert_allclose(result["ordinary"],old["full"][:,old["named_index"]],rtol=.004,atol=3e-4)
    np.testing.assert_array_less(max_clean,3e-6)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/"last_block.npz",ids=ids,split=split[used],norm_fraction=np.linalg.norm(v,axis=1)/rad,
                        linear=np.einsum("nhd,nd->nh",j,v),zero_dose_max=max_clean,
                        activation_type=type(modules["activation"]).__name__,**result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--batch",type=int,default=6)
    a = p.parse_args()
    torch.set_num_threads(2)
    collect(a.out,a.scratch,a.batch)
