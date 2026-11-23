"""Compare active decoder spans to four fixed and four globally rotated, state-varying spans."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND, HERE as OBSERVED
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import HERE, material
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.span_inverse import constrained
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from tsfm_lens.sae.train import load_sae_checkpoint

SEEDS = range(2301, 2305)
OUT = HERE / "results"


def run(seed, out):
    target, x, ids, split, z, atoms, donor, _, h, pos, _, radius, row, col, v, _ = material("bolt")
    keep = (z[split == "daily"] > 0).sum(0) >= 8
    use = (split[row] == "daily") & keep[col]
    with np.load(OUT / "bolt_span_inverse.npz") as parent:
        np.testing.assert_array_equal(row, parent["rows"])
        np.testing.assert_array_equal(col, parent["cols"])
        dense = parent["dense"][use]
        original = {k: parent[k][use] for k in ("active_same", "active_cap")}
    row, col, v = row[use], col[use], v[use].astype(float)
    with np.load(BLIND / (target + "_early.npz")) as f:
        jac = f["jac"][row].astype(float)
    with np.load(OBSERVED / "results/bolt_fresh.npz") as f:
        weight = float(f["weight"])
    sae = load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        active = sae.encode(torch.from_numpy(h.mean(1))).topk(sae.k).indices.numpy()
    q = np.linalg.svd(sae.W_dec.detach().numpy()[active].astype(float), full_matrices=False)[2]
    d = q.shape[-1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d)
    signs = rng.choice([-1., 1.], size=d)
    rotated = q[:, :, perm] * signs
    fixed = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, 32)))[0].T
    # A single hidden-coordinate isometry preserves relations within and across histories.
    np.testing.assert_allclose(rotated[0] @ rotated[-1].T, q[0] @ q[-1].T, atol=1e-12)
    adapter = adapter_for(target)
    select = last_window_slice(adapter, pos)
    data = dict(row=row, atom=atoms[col], id=ids[row], dense=dense, donor=donor[row,col], **original)
    def effects(delta):
        return forecast16(adapter, MODELS[target][-1], x[row], h[row], delta.astype("f4"), select, 24)
    baseline = effects(np.zeros_like(v))
    length = np.linalg.norm(v, axis=1)
    for name, axes in (("rotated", rotated[row]), ("fixed", np.broadcast_to(fixed,(len(v),*fixed.shape)))):
        proposed = constrained(jac, axes, v, weight)
        norm = np.linalg.norm(proposed, axis=1)
        for budget, limit in (("same", length), ("cap", .2*radius)):
            arm = name + "_" + budget
            scale = np.minimum(1., limit / norm)
            data[arm] = effects(proposed * scale[:,None]) - baseline
            data[arm+"_ratio"] = norm * scale / length
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"span_control_{seed}.npz", **data)


def skill(p, y, atoms):
    p = p - p.mean(1,keepdims=True)
    y = y - y.mean(1,keepdims=True)
    scores = []
    for a in np.unique(atoms):
        use = atoms == a
        energy = np.sum(y[use]**2)
        if energy > 1e-10 * use.sum() * y.shape[1]:
            scores.append(1 - np.sum((p[use]-y[use])**2)/energy)
    return float(np.mean(scores))


def score(out):
    rows = {}
    differences = {"fixed": [], "rotated": []}
    with np.load(OUT / "bolt_span_inverse.npz") as parent:
        daily = np.flatnonzero(parent["split"] == "daily")
        keep = parent["active"][daily].sum(0) >= 8
        atoms = parent["atoms"][keep]
        mask = parent["active"][daily][:,keep]
    hdraw = resampling_counts(len(daily), draws=3000)
    fdraw = np.random.default_rng(221106).multinomial(len(atoms),np.full(len(atoms),1/len(atoms)),size=3000)
    for seed in SEEDS:
        with np.load(out/f"span_control_{seed}.npz") as data:
            row = {}
            row_ids = np.searchsorted(daily, data["row"])
            col_ids = np.array([np.flatnonzero(atoms == a).item() for a in data["atom"]])
            def centered(key):
                wave = np.zeros((*mask.shape,8))
                wave[row_ids,col_ids] = data[key][:,:8]
                return wave-wave.mean(-1,keepdims=True)
            truth = centered("dense")
            def estimate(key):
                return intervals(mask, centered(key), truth, hdraw, fdraw)[:3]
            active = estimate("active_same")
            for control in differences:
                ref = estimate(control+"_same")
                differences[control].append(tuple(a-b for a,b in zip(active,ref)))
            for arm in ("active_same", "active_cap", "rotated_same", "rotated_cap", "fixed_same", "fixed_cap"):
                pred = data[arm].astype(float)
                row[arm] = dict(early_dense=skill(pred[:,:8],data["dense"][:,:8].astype(float),data["atom"]),
                                late_donor=skill(pred[:,8:],data["donor"][:,8:].astype(float),data["atom"]))
                if arm+"_ratio" in data:
                    row[arm]["median_length_ratio"] = float(np.median(data[arm+"_ratio"]))
            with (OUT/"span_inverse.json").open() as f:
                primary = json.load(f)["bolt"]["daily"]
            for arm in ("active_same","active_cap"):
                np.testing.assert_allclose(row[arm]["early_dense"],primary["early_dense"][arm]["value"],atol=1e-7)
            rows[seed] = row
    pooled = {ctrl: entry(tuple(np.mean(draws,axis=0) for draws in zip(*by_seed)))
              for ctrl,by_seed in differences.items()}
    result = dict(seeds=rows, primary_active_minus_mean_control=pooled,
                  histories=len(daily), events=int(mask.sum()), features=len(atoms))
    (out/"span_controls.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["score", *map(str,SEEDS)])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.stage == "score": score(a.out)
    else: run(int(a.stage), a.out)
