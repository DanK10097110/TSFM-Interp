"""Independently install a physical output decoder and remove natural hidden feature contributions."""
import argparse
import json
import numpy as np
import torch
from research.activation_transfer.moment.head_interventions.headbank import HERE,bank
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,load,layers
from tsfm_lens.sae.train import load_sae_checkpoint


def replay(size,seed):
    with np.load(HERE/"results"/f"{size}_{seed}_wave.npz") as f:arch=dict(f)
    x,ids,dom=histories(False)
    np.testing.assert_array_equal(ids,arch["ids"])
    sa=load_sae_checkpoint(str(CACHE/size/f"sae_{seed}.pt"))
    a=load(size);heads=bank(size)
    rng=np.random.default_rng(22112301)
    idx=np.concatenate([rng.choice(np.flatnonzero(dom[arch["row"]]==key),12,replace=False) for key in np.unique(dom)])
    rows=arch["row"][idx];atoms=arch["atoms"][arch["col"][idx]]
    layer=a.module.get_submodule(layers(a)["middle"])
    def remove(_,inputs,output):
        t=output[0] if isinstance(output,tuple) else output
        h=t[:,61].detach().cpu()
        z=sa.encode(h)[np.arange(len(h)),atoms]
        np.testing.assert_allclose(z.numpy(),arch["codes"][rows,arch["col"][idx]],rtol=2e-5,atol=2e-5)
        r=t.clone()
        r[:,61]-=(z[:,None]*sa.W_dec[atoms]).to(t)
        return (r,*output[1:]) if isinstance(output,tuple) else r
    original=a.module.head.linear;result={}
    try:
        for name,(w,b) in heads.items():
            alt=torch.nn.Linear(w.shape[0],w.shape[1])
            with torch.no_grad():
                alt.weight.copy_(torch.as_tensor(w.T.copy(),dtype=alt.weight.dtype))
                alt.bias.copy_(torch.as_tensor(b.copy(),dtype=alt.bias.dtype))
            a.module.head.linear=alt.eval()
            clean=a.predict(x[rows],16,[.5])["point"]
            h=layer.register_forward_hook(remove)
            try:edited=a.predict(x[rows],16,[.5])["point"]
            finally:h.remove()
            obs=edited-clean;expected=arch[name+"_effect"][idx]
            gap=np.abs(obs-expected)
            clean_gap=np.abs(clean-arch[name+"_base"][idx])
            np.testing.assert_allclose(clean_gap,0.,atol=1e-5)
            np.testing.assert_allclose(gap,0.,atol=1.5e-5)
            result[name]=dict(max_forecast_gap=float(clean_gap.max()),max_effect_gap=float(gap.max()),
                              rms_relative_effect_gap=float(np.linalg.norm(gap)/np.linalg.norm(expected)))
    finally:a.module.head.linear=original
    meta=dict(n=len(idx),ids=ids[rows].tolist(),atoms=atoms.tolist(),heads=result)
    (HERE/"results"/f"{size}_{seed}_replay.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size");p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0]);a=p.parse_args()
    torch.set_num_threads(2);replay(a.size,a.seed)
