"""Symmetric forward-input sensitivities on the original normalized samples."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND


HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/tsfm_input_bridge_20261121")
WINDOW = 32
RANK = 12
STEP = .02 * np.sqrt(WINDOW)
CAP = .25 * np.sqrt(WINDOW)
BATCH = 64


def probe_path(model):
    durable = HERE/"results"/(model+"_probe.npz")
    return durable if durable.is_file() else CACHE/(model+"_probe.npz")


def basis():
    sample = np.arange(WINDOW) + .5
    dct = np.sqrt(2 / WINDOW) * np.cos(np.pi * np.arange(RANK)[:, None] * sample / WINDOW)
    dct[0] /= np.sqrt(2)
    return dct


def forecast(adapter, x, batch=BATCH):
    return np.concatenate([adapter.predict(x[i:i+batch], 16, [.5])["quantiles"][..., 0]
                           for i in range(0, len(x), batch)])


def edit(x, coeff, axes):
    out = x.copy()
    out[..., -WINDOW:] += (coeff @ axes).astype(x.dtype)
    return out


def run(model):
    CACHE.mkdir(exist_ok=True)
    with np.load(BLIND / "data.npz") as f:
        x, ids, split = f["x"], f["ids"], f["split"]
    assert x.shape == (256, 480)
    axes = basis()
    adapter = adapter_for(model)
    base = forecast(adapter, x)
    with np.load(BLIND / (model + "_early.npz")) as f:
        np.testing.assert_array_equal(ids, f["ids"])
        cl_error = float(np.max(np.abs(base-f["forecast"])))
        np.testing.assert_allclose(base, f["forecast"], atol=1.5e-5, rtol=2e-6)
    jac = []
    for axis in axes:
        v = np.broadcast_to(axis, (len(x), WINDOW))
        cat = np.tile(x, (2, 1))
        cat[:len(x), -WINDOW:] += (STEP*v).astype(x.dtype)
        cat[len(x):, -WINDOW:] -= (STEP*v).astype(x.dtype)
        y = forecast(adapter, cat).astype(float)
        jac.append((y[:len(x), :8]-y[len(x):, :8])/(2*STEP))
    jac = np.stack(jac, axis=2)
    rng = np.random.default_rng(20261121091)
    checks = []
    for domain in np.unique(split):
        idx = np.flatnonzero(split == domain)[:8]
        v = rng.normal(size=(len(idx), RANK))
        v *= (.01*np.sqrt(WINDOW)/np.linalg.norm(v, axis=1))[:, None]
        yp, ym = [forecast(adapter, edit(x[idx], sign*v, axes)) for sign in (1, -1)]
        fd = (yp[:, :8].astype(float)-ym[:, :8])/2
        pred = np.einsum("nhk,nk->nh", jac[idx], v)
        error = float(np.linalg.norm(fd-pred)/max(np.linalg.norm(fd), 1e-15))
        checks.append(dict(domain=domain, ids=ids[idx].tolist(), rel_error=error,
                           rms_delta=float(np.sqrt(np.mean(fd**2)))))
    if not np.isfinite(jac).all():
        raise ValueError("Nonfinite public-probe sensitivities")
    np.savez_compressed(CACHE / (model + "_probe.npz"), ids=ids, split=split, jac=jac, base=base, axes=axes)
    shutil.copyfile(CACHE / (model + "_probe.npz"), HERE / "results" / (model + "_probe.npz"))
    audit = dict(checkpoint=model, max_clean_replay=cl_error, heldout_directions=checks,
                 orthonormal_error=float(np.max(np.abs(axes@axes.T-np.eye(RANK)))))
    (HERE/"results"/(model+"_probe.json")).write_text(json.dumps(audit, indent=2)+"\n")
    print(model, audit, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=MODELS)
    torch.set_num_threads(1)
    run(p.parse_args().model)
