"""Fit a target write to the measured source-atom effect, and replay it on new items."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.replication.data import CACHE, HERE, ORIGINAL
from research.activation_transfer.replication.experiment import clean_forecast, read_materials
from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from tsfm_lens.steering import solve_direction


def balance(values, level_rms, shape_rms):
    mean = values.mean(axis=1, keepdims=True)
    return mean / level_rms + (values - mean) / shape_rms


def fit_feature(jac, z, truth, radius):
    """Fit one typical write in PC coordinates using only this atom's active cal items."""
    active = z > 0
    if not active.any():
        return np.zeros(jac.shape[-1]), np.zeros(truth.shape[-1])
    z, y, j = z[active].astype(float), truth[active].astype(float), jac[active].astype(float)
    average_amplitude = np.mean(z)
    design = -(z / average_amplitude)[:, None, None] * j
    mean = y.mean(axis=1, keepdims=True)
    floor = max(.1 * np.sqrt(np.mean(y**2)), 1e-10)
    level_rms = max(np.sqrt(np.mean(mean**2)), floor)
    shape_rms = max(np.sqrt(np.mean((y - mean)**2)), floor)
    x = balance(design, level_rms, shape_rms).reshape(-1, j.shape[-1])
    labels = balance(y, level_rms, shape_rms).ravel()
    penalty = .01*np.linalg.svd(x, compute_uv=False)[0]**2
    delta = solve_direction(x, labels, ridge=max(penalty, 1e-12), max_norm=.2*radius).delta
    template = np.sum(z[:, None]*y, axis=0)/np.sum(z*z)
    return delta / average_amplitude, template


def run(source, cache, out, batch):
    target = next(m for m in MODELS if m != source)
    with np.load(cache / "data.npz") as f:
        x, ids, split = f["x"], f["ids"], f["split"]
    with np.load(ORIGINAL / (source+"_vectors.npz")) as v:
        atom_ids = v["source_atoms"]
    tok, pos, token_ids = read_materials(cache, target)
    np.testing.assert_array_equal(ids, token_ids)
    target_train = tok[split=="train"].mean(axis=1).astype(float)
    _, _, basis = np.linalg.svd(target_train - target_train.mean(axis=0), full_matrices=False)
    basis = basis[:24]
    radius = np.median(np.linalg.norm(target_train, axis=1))
    step = .01 * radius
    adapter = adapter_for(target)
    layer = MODELS[target][-1]
    select = last_window_slice(adapter, pos)
    cal = split=="cal"
    jac = []
    for axis in basis:
        d = np.broadcast_to(axis*step, (cal.sum(), len(axis))).astype("f4")
        hi = change_forecasts(adapter, layer, x[cal], tok[cal], d, select, batch)
        lo = change_forecasts(adapter, layer, x[cal], tok[cal], -d, select, batch)
        jac.append((hi-lo)/(2*step))
    jac = np.stack(jac, axis=-1)
    primary = HERE.parent / "atoms/results" / (source+"_effects.npz")
    with np.load(primary) as f:
        mask = f["split"]=="cal"
        np.testing.assert_array_equal(ids[cal], f["ids"][mask])
        codes = f["source_codes"][mask]
        reference = f["source_native"][mask]
    fitted = [fit_feature(jac, codes[:, i], reference[:, i], radius) for i in range(len(atom_ids))]
    directions = np.stack([f[0] for f in fitted]) @ basis
    template = np.stack([f[1] for f in fitted])
    with np.load(cache / (source+".npz")) as f:
        states = f["tokens"].mean(axis=1)
    from research.activation_transfer.atoms.materials import dictionary_path
    from tsfm_lens.sae.train import load_sae_checkpoint
    sae = load_sae_checkpoint(str(dictionary_path(source)))
    keep = split!="train"
    with torch.inference_mode():
        z = sae.encode(torch.from_numpy(states[keep])).numpy()[:, atom_ids]
    np.testing.assert_allclose(z[split[keep]=="cal"], codes, atol=2e-4, rtol=1e-4)
    ordinary = clean_forecast(adapter, layer, x[keep], tok[keep], select, batch)
    patch = run_arm(adapter, layer, select, x[keep], tok[keep], ordinary, z>0, z, directions, batch)
    np.savez_compressed(out/(source+"_functional.npz"), ids=ids[keep], split=split[keep],
                        atoms=atom_ids, active=z>0, derivative_write=patch, template=z[:,:,None]*template,
                        directions=directions, ncal=np.sum(codes>0, axis=0), typical_radius=radius)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=MODELS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=HERE/"results")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(3)
    run(a.source, a.cache, a.out, a.batch)
