"""Replay native source-atom removals and translated writes across SAE initializations."""
import argparse
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.replication.data import CACHE, HERE
from research.activation_transfer.atoms.materials import adapter_for, dictionary_path, MODELS
from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.atoms.vectors import bridge, choose_atoms
from tsfm_lens.sae.train import load_sae_checkpoint
from tsfm_lens.steering import solve_direction


def read_materials(cache, name):
    with np.load(cache / (name + ".npz")) as file:
        return file["tokens"], file["positions"], file["ids"]


def clean_forecast(model, layer, x, tokens, select, batch):
    return change_forecasts(model, layer, x, tokens, None, select, batch)


def common_level(model, layer, xcal, tcal, select, training, batch):
    """Fit a single level-lifting write, regardless of the identity of a source atom."""
    x = training.astype(float)
    basis = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)[2][:24]
    radius = np.median(np.linalg.norm(x, axis=1))
    step = .01 * radius
    output = []
    for axis in basis:
        d = np.broadcast_to(step * axis, (len(xcal), len(axis))).astype("f4")
        hi = change_forecasts(model, layer, xcal, tcal, d, select, batch)
        lo = change_forecasts(model, layer, xcal, tcal, -d, select, batch)
        output.append((hi - lo) / (2*step))
    jac = np.stack(output, axis=-1).reshape(-1, len(basis)).astype(float)
    penalty = .01 * np.linalg.svd(jac, compute_uv=False)[0]**2
    coefficients = solve_direction(jac, np.full(len(jac), .02), ridge=penalty, max_norm=.2*radius)
    direction = coefficients.delta @ basis
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("No measurable calibration level response")
    return direction / norm


def dose_scale(pred, source):
    """Signed scalar per atom. Inactive examples have both effects exactly zero."""
    dot = np.einsum("nkh,nkh->k", pred.astype(float), source.astype(float))
    energy = np.einsum("nkh,nkh->k", pred.astype(float), pred.astype(float))
    return np.clip(np.divide(dot, energy, out=np.zeros_like(dot), where=energy>1e-15), -8, 8)


def derangements(n):
    rng = np.random.default_rng(2026112019)
    for _ in range(4):
        order = rng.permutation(n)
        perm = np.empty_like(order)
        perm[order] = np.roll(order, 1)
        yield perm


def run(source, seed, cache, out, checkpoints, batch):
    target = next(n for n in MODELS if n != source)
    with np.load(cache / "data.npz") as f:
        xs, split, ids = f["x"], f["split"], f["ids"]
    train, keep = split == "train", split != "train"
    x, names, labels = xs[keep], ids[keep], split[keep]
    cal = labels == "cal"
    source_tokens, source_pos, si = read_materials(cache, source)
    target_tokens, target_pos, ti = read_materials(cache, target)
    np.testing.assert_array_equal(ids, si)
    np.testing.assert_array_equal(ids, ti)
    checkpoint = dictionary_path(source) if seed==0 else checkpoints / f"seed{seed}_{source}.pt"
    with checkpoint.open("rb") as reader:
        file_hash = file_digest(reader, "sha256").hexdigest()
    sae = load_sae_checkpoint(str(checkpoint))
    state = source_tokens.mean(axis=1)
    with torch.inference_mode():
        codes = sae.encode(torch.from_numpy(state)).numpy()
        selected = choose_atoms(codes[train], 16, 2026112017 if seed==0 else 2026112070+seed)
        source_dirs = sae.W_dec.numpy()[selected].copy()
    target_dirs, _, _ = bridge(state[train], target_tokens[train].mean(axis=1), source_dirs, 2026112018)
    z = codes[keep][:, selected]
    active = z > 0
    arrays = dict(ids=names, split=labels, active=active, atoms=selected, source_codes=z,
                  source_checkpoint_hash=file_hash)
    basename = f"{source}_seed{seed}"
    arm_cache = cache / basename
    arm_cache.mkdir(exist_ok=True)

    def replay(model, layer, select, tok, ordinary, arm, strength, directions):
        path = arm_cache / (arm + ".npz")
        if path.exists():
            with np.load(path) as f:
                np.testing.assert_array_equal(f["ids"], names)
                np.testing.assert_array_equal(f["atoms"], selected)
                if str(f["checkpoint"]) != file_hash:
                    raise ValueError("Checkpoint changed")
                return f["effects"]
        effects = run_arm(model, layer, select, x, tok, ordinary, active, strength, directions, batch)
        np.savez_compressed(path, effects=effects, ids=names, atoms=selected, checkpoint=file_hash)
        print(basename, arm, flush=True)
        return effects

    src = adapter_for(source)
    layer = MODELS[source][-1]
    ss = last_window_slice(src, source_pos)
    base = clean_forecast(src, layer, x, source_tokens[keep], ss, batch)
    arrays["native"] = replay(src, layer, ss, source_tokens[keep], base, "native", z, source_dirs)
    del src
    tgt = adapter_for(target)
    layer = MODELS[target][-1]
    select = last_window_slice(tgt, target_pos)
    target_train = target_tokens[train].mean(axis=1)
    tok = target_tokens[keep]
    base = clean_forecast(tgt, layer, x, tok, select, batch)

    def candidate(arm, strength, directions):
        arrays[arm] = replay(tgt, layer, select, tok, base, arm, strength, directions)

    candidate("dense", z, target_dirs)
    size = np.linalg.norm(target_dirs, axis=1, keepdims=True)
    for i, perm in enumerate(derangements(len(selected))):
        wrong = target_dirs[perm] * size / np.maximum(size[perm], 1e-15)
        candidate("wrong" + str(i), z, wrong)
    if seed == 0:
        unit = common_level(tgt, layer, x[cal], tok[cal], select, target_train, batch)
        level = unit * size
        level_cal = run_arm(tgt, layer, select, x[cal], tok[cal], base[cal],
                            active[cal], z[cal], level, batch)
        for arm, predictor, directions in [("dense_dosed", arrays["dense"][cal], target_dirs),
                                            ("level_dosed", level_cal, level)]:
            scales = dose_scale(predictor, arrays["native"][cal])
            arrays[arm + "_scale"] = scales
            candidate(arm, z*scales, directions)
        arrays["common_level_direction"] = unit
    np.savez_compressed(out / (basename + ".npz"), **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=MODELS)
    parser.add_argument("seed", choices=[0, 1, 2], type=int)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--checkpoints", type=Path, default=Path("/tmp/tsfm_boundary_20261120"))
    parser.add_argument("--batch", type=int, default=32)
    a = parser.parse_args()
    torch.set_num_threads(3)
    a.out.mkdir(exist_ok=True, parents=True)
    run(a.source, a.seed, a.cache, a.out, a.checkpoints, a.batch)
