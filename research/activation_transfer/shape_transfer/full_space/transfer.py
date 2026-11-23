"""Apply one shared derivative-based translation rule to previously selected SAE atoms."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE, SCRATCH
from tsfm_lens.sae.train import load_sae_checkpoint


def boost_shape(array, factor):
    mean = array.mean(axis=-2, keepdims=True)
    return mean + factor * (array - mean)


def solve_fixed(target, responses):
    """Multiple independent right-hand sides, sharing a single training operator."""
    design = target.reshape(-1, target.shape[-1])
    rhs = responses.reshape(-1, responses.shape[-1])
    gram = design @ design.T
    ridge = .001 * np.linalg.eigvalsh(gram)[-1]
    if ridge <= 0:
        raise ValueError("Degenerate target response")
    return design.T @ cho_solve(cho_factor(gram + ridge * np.eye(len(gram))), rhs)


def solve_adaptive(target, responses):
    """Damped minimum-norm inverse for the eight actual leads of each history."""
    transpose = target.swapaxes(-2, -1)
    gram = target @ transpose
    ridge = .001 * np.linalg.eigvalsh(gram)[:, -1]
    if np.min(ridge) <= 0:
        raise ValueError("Degenerate target response")
    return transpose @ np.linalg.solve(gram + ridge[:, None, None]*np.eye(gram.shape[-1]), responses)


def apply(adapter, layer, selector, x, tokens, ordinary, codes, directions, radius, batch,
          forecast_change=change_forecasts, max_fraction=.2):
    active = codes > 0
    rows, feats = np.nonzero(active)
    effects = np.zeros((*active.shape, ordinary.shape[-1]), dtype="f4")
    clipped = np.zeros_like(active)
    for start in range(0, len(rows), batch):
        i, j = rows[start:start+batch], feats[start:start+batch]
        vectors = directions[:, j].T if directions.ndim == 2 else directions[i, :, j]
        delta = -codes[i, j, None] * vectors
        norms = np.linalg.norm(delta, axis=1)
        scales = np.minimum(1., (max_fraction*radius) / np.maximum(norms, 1e-15))
        clipped[i, j] = scales < 1.
        patched = forecast_change(adapter, layer, x[i], tokens[i],
                                  (delta*scales[:, None]).astype("f4"), selector, batch)
        effects[i, j] = patched - ordinary[i]
    return effects, clipped


def run(source, cache, scratch, out, batch):
    target = next(m for m in MODELS if m != source)
    with np.load(REPLICATION / "results" / (source + "_seed0.npz")) as f:
        ids, labels, codes, atoms = f["ids"], f["split"], f["source_codes"], f["atoms"]
    with np.load(cache / "data.npz") as f:
        use = f["split"] != "train"
        np.testing.assert_array_equal(ids, f["ids"][use])
        np.testing.assert_array_equal(labels, f["split"][use])
        x = f["x"][use]
    jac = {}
    for n in [source, target]:
        with np.load(scratch / (n + "_jac.npz")) as f:
            np.testing.assert_array_equal(ids, f["ids"])
            np.testing.assert_array_equal(labels, f["split"])
            jac[n] = f["jac"].astype(float)
            if n == target:
                target_radius, ordinary = float(f["radius"]), f["forecast"]
    sae = load_sae_checkpoint(str(dictionary_path(source)))
    directions = sae.W_dec.detach().numpy()[atoms].astype(float).T
    source_tokens, _, item_ids = read_materials(cache, source)
    np.testing.assert_array_equal(item_ids[use], ids)
    with torch.inference_mode():
        actual_codes = sae.encode(torch.from_numpy(source_tokens[use].mean(axis=1))).numpy()[:, atoms]
    np.testing.assert_allclose(actual_codes, codes, atol=2e-4, rtol=1e-4)
    cal = labels == "cal"
    ev = ~cal
    source_cal = jac[source][cal]
    common = source_cal.mean(axis=1, keepdims=True)
    shape_weight = np.sqrt(np.mean(common**2) / np.mean((source_cal-common)**2))
    responses = jac[source] @ directions
    target_cal = boost_shape(jac[target][cal], shape_weight)
    fixed = solve_fixed(target_cal, boost_shape(responses[cal], shape_weight))
    adapter = adapter_for(target)
    layer = MODELS[target][-1]
    tokens, positions, item_ids = read_materials(cache, target)
    np.testing.assert_array_equal(ids, item_ids[use])
    selector = last_window_slice(adapter, positions)
    tokens = tokens[use][ev]
    jsu = responses[ev]
    jt = jac[target][ev]
    order = np.array([3, 6, 1, 7, 0, 2, 5, 4])
    answers = dict(source_linear=(-codes[ev, :, None]*jsu.swapaxes(1, 2)).astype("f4"))
    for method in ("fixed", "adaptive", "unweighted", "mean_donor", "wrong_leads"):
        if method == "fixed":
            writes = fixed
        else:
            obs = jt[:, order] if method == "wrong_leads" else jt
            requested = np.broadcast_to(responses[cal].mean(axis=0), jsu.shape) if method == "mean_donor" else jsu
            weight = 1. if method == "unweighted" else shape_weight
            writes = solve_adaptive(boost_shape(obs, weight), boost_shape(requested, weight))
        answers[method], answers[method + "_clipped"] = apply(
            adapter, layer, selector, x[ev], tokens, ordinary[ev], codes[ev], writes, target_radius, batch)
        print(source, method, "clips", answers[method + "_clipped"].sum(), flush=True)
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out / (source + ".npz"), ids=ids[ev], split=labels[ev], atoms=atoms,
                        active=codes[ev] > 0, shape_weight=shape_weight, **answers)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=MODELS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a.source, a.cache, a.scratch, a.out, a.batch)
