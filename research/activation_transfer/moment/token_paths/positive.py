"""Native full-graph patch readout routing under independently constructed nonzero additions."""
import argparse
import json
import numpy as np
import torch
from research.activation_transfer.moment.token_paths.geometry import load,layers,histories,HERE


def run(size):
    a=load(size)
    x,ids,split=histories(False)
    rng=np.random.default_rng(22112271)
    rows=np.concatenate([rng.choice(np.flatnonzero(split==d),16,replace=False) for d in np.unique(split)])
    x=x[rows]
    y0=a.predict(x,16,[.5])["point"]
    layer=a.module.get_submodule(layers(a)["readout"])
    w=a.module.head.linear.weight.detach().numpy()
    vec=rng.normal(size=(len(x),w.shape[1]))
    vec=(vec/np.linalg.norm(vec,axis=1)[:,None]).astype("f4")
    out={}
    for i in (61,62,63):
        def patch(_,args,output):
            t=output.clone()
            t[:,i]+=torch.from_numpy(vec).to(t)
            return t
        hook=layer.register_forward_hook(patch)
        try:out[i]=a.predict(x,16,[.5])["point"]-y0
        finally:hook.remove()
    np.testing.assert_array_equal(out[61],0)
    np.testing.assert_array_equal(out[62][:,8:],0)
    np.testing.assert_array_equal(out[63][:,:8],0)
    np.testing.assert_allclose(out[62][:,:8],out[63][:,8:],atol=1e-6,rtol=.0001)
    sd=a.module.normalizer.stdev.detach().numpy().reshape(-1)
    if a.module.normalizer.affine:sd=sd/(a.module.normalizer.affine_weight.detach().numpy().item()+a.module.normalizer.eps**2)
    expected=vec@w.T*sd[:,None]
    np.testing.assert_allclose(out[62][:,:8],expected,atol=1e-6,rtol=.0001)
    assert np.min(np.linalg.norm(out[62][:,:8],axis=1))>1e-4
    rec=dict(histories=ids[rows].tolist(),input_sd_min=float(sd.min()),n=len(rows),write_norm=1,
             observed_effect_max=float(np.abs(out[61]).max()),first_to_second_max=float(np.abs(out[62][:,8:]).max()),
             second_to_first_max=float(np.abs(out[63][:,:8]).max()),repeat_pattern_max=float(np.abs(out[62][:,:8]-out[63][:,8:]).max()),
             analytic_error_max=float(np.max(np.abs(out[62][:,:8]-expected))),positive_rms=float(np.sqrt(np.mean(out[62][:,:8]**2))))
    (HERE/"results"/f"{size}_positive.json").write_text(json.dumps(rec,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size");a=p.parse_args();torch.set_num_threads(2);run(a.size)
