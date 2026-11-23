"""Collect recipient and donor local derivatives out to 16, with independent production checks."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as ORIGINAL
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from tsfm_lens.extraction.hooks import token_patch

SCRATCH = Path("/tmp/tsfm_continuation_20261120")
HERE = Path(__file__).resolve().parent


def collect(name, batch=8):
    adapter = adapter_for(name)
    layer = MODELS[name][-1]
    states, positions, ids = read_materials(CACHE, name)
    selector = last_window_slice(adapter, positions)
    with np.load(CACHE / "data.npz") as data:
        np.testing.assert_array_equal(data["ids"], ids)
        use = data["split"] != "train"
        split, x = data["split"][use], data["x"][use]
    h, ids = states[use], ids[use]
    jacobians, forecasts = [], []
    for start in range(0, len(x), batch):
        tokens = torch.from_numpy(h[start:start+batch])
        delta = torch.zeros_like(tokens[:, 0], requires_grad=True)
        with torch.enable_grad(), token_patch(adapter.module, layer, selector, tokens + delta[:, None]):
            point = adapter.predict_differentiable(x[start:start+batch], 16)
            gradients = [torch.autograd.grad(point[:, i].sum(), delta, retain_graph=i<15)[0] for i in range(16)]
        jacobians.append(torch.stack(gradients, dim=1).detach().numpy())
        forecasts.append(point.detach().numpy())
    jac = np.concatenate(jacobians)
    ordinary = forecast16(adapter, layer, x, h, None, selector, batch)
    clean_err = float(np.max(np.abs(ordinary-np.concatenate(forecasts))))
    np.testing.assert_allclose(ordinary, np.concatenate(forecasts), atol=8e-5, rtol=2e-5)
    with np.load(ORIGINAL / (name+"_jac.npz")) as old:
        np.testing.assert_array_equal(ids,old["ids"])
        before = old["jac"]
        early_err = float(np.linalg.norm(jac[:,:8]-before)/np.linalg.norm(before))
        np.testing.assert_allclose(jac[:,:8], before, atol=2e-4, rtol=2e-4)
        radius = float(old["radius"])
    check = np.r_[0:8,96:104,288:296]
    u = np.random.default_rng(33191).normal(size=(len(check), h.shape[-1])).astype("f4")
    u *= (.002*radius/np.linalg.norm(u, axis=1))[:,None]
    plus, minus = [forecast16(adapter,layer,x[check],h[check],sign*u,selector,batch) for sign in (1,-1)]
    measured = (plus-minus)/2
    predicted = np.einsum("nhd,nd->nh",jac[check],u)
    error = np.asarray([np.linalg.norm(p-m)/np.linalg.norm(m) for p,m in
                        zip(np.split(predicted,2,axis=1),np.split(measured,2,axis=1))])
    if np.max(error)>.08 or not np.isfinite(jac).all():
        raise ValueError(f"Production forecast disagrees with local derivative: {error}")
    SCRATCH.mkdir(exist_ok=True)
    np.savez_compressed(SCRATCH/(name+"_jac16.npz"),ids=ids,split=split,jac=jac,forecast=ordinary,
                        radius=radius,early_relative_error=early_err,fd_indices=check,fd_relative=error,
                        clean_max_error=clean_err,fd_measured=measured,fd_predicted=predicted)
    print(name, jac.shape, early_err, error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",choices=MODELS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    collect(args.model)
