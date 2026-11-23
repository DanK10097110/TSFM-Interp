"""Measure the live median derivative at the sites where the two SAEs were trained."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.replication.experiment import read_materials
from tsfm_lens.extraction.hooks import token_patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/tsfm_full_space_sae_20261120")


def collect(name, cache, out, batch):
    adapter = adapter_for(name)
    layer = MODELS[name][-1]
    states, positions, ids = read_materials(cache, name)
    selector = last_window_slice(adapter, positions)
    with np.load(cache / "data.npz") as f:
        np.testing.assert_array_equal(f["ids"], ids)
        split, inputs = f["split"], f["x"]
    radius = np.median(np.linalg.norm(states[split == "train"].mean(axis=1), axis=1))
    use = split != "train"
    x, h = inputs[use], states[use]
    jacobians, forecasts = [], []
    for start in range(0, len(x), batch):
        obs = x[start:start+batch]
        tokens = torch.from_numpy(h[start:start+batch])
        delta = torch.zeros_like(tokens[:, 0], requires_grad=True)
        with torch.enable_grad(), token_patch(adapter.module, layer, selector, tokens + delta[:, None]):
            point = adapter.predict_differentiable(obs, 8)
            gradients = [torch.autograd.grad(point[:, i].sum(), delta, retain_graph=i < 7)[0]
                         for i in range(8)]
        jacobians.append(torch.stack(gradients, dim=1).detach().numpy())
        forecasts.append(point.detach().numpy())
    jac, base = np.concatenate(jacobians), np.concatenate(forecasts)
    ordinary = change_forecasts(adapter, layer, x, h, None, selector, batch)
    np.testing.assert_allclose(base, ordinary, atol=8e-5, rtol=2e-5)
    # An independent production forecast checks the derivative sign, lead order, units and token mask.
    r = np.random.default_rng(8903)
    checks = np.r_[0:8, 96:104, 288:296]
    u = r.normal(size=(len(checks), h.shape[-1])).astype("f4")
    u *= (.002 * radius / np.linalg.norm(u, axis=1))[:, None]
    high, low = [change_forecasts(adapter, layer, x[checks], h[checks], sign*u, selector, batch)
                 for sign in (1, -1)]
    predicted = np.einsum("nhd,nd->nh", jac[checks], u)
    measured = (high - low) / 2
    relative_error = np.linalg.norm(predicted - measured) / np.linalg.norm(measured)
    if not np.isfinite(jac).all() or relative_error > .08:
        raise ValueError(f"Production finite difference disagreed with gradient by {relative_error}")
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out / (name + "_jac.npz"), ids=ids[use], split=split[use], jac=jac,
                        forecast=ordinary, radius=radius, measured=measured, predicted=predicted,
                        fd_indices=checks, fd_relative_error=relative_error,
                        max_clean_error=np.max(np.abs(ordinary-base)))
    print(name, jac.shape, "finite difference relative error", relative_error, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=MODELS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=SCRATCH)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(2)
    collect(a.model, a.cache, a.out, a.batch)
