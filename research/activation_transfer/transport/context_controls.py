"""Break source-context identity and separate background-specific transfer from common templates."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.transport.analyze import fit_bridge, metrics, pooled_evaluation, scaled_like
from research.activation_transfer.transport.capture import REVISIONS, load_adapter, signed_response
from research.activation_transfer.transport.prepare import CONCEPTS, DEFAULT_CACHE


def donors_and_clusters(count, seed):
    """Disjoint two-item swaps create explicit independent bootstrap clusters."""
    if count % 2:
        raise ValueError("the paired donor test needs an even item count")
    pairs = np.random.default_rng(seed).permutation(count).reshape(-1, 2)
    donor = np.empty(count, dtype=int)
    donor[pairs[:, 0]] = pairs[:, 1]
    donor[pairs[:, 1]] = pairs[:, 0]
    return donor, pairs


def bootstrap_summary(predictions, truth, pairs, n_boot=4000):
    """Recenter each concept/lead within every series-cluster bootstrap resample."""
    rng = np.random.default_rng(2026112007)
    draws = pairs[rng.integers(len(pairs), size=(n_boot, len(pairs)))].reshape(n_boot, -1)
    n = len(truth)
    # Integer multiplicities implement resampling without creating [draw,series,concept,lead] arrays.
    weights = np.zeros((n_boot + 1, n), dtype=float)
    weights[0, :] = 1
    for i, sample in enumerate(draws, start=1):
        weights[i] = np.bincount(sample, minlength=n)
    out = {}
    for view, horizon_centered, background_centered in [
            ("full", False, False), ("horizon_shape", True, False),
            ("background", False, True), ("background_shape", True, True)]:
        y = truth - truth.mean(axis=-1, keepdims=True) if horizon_centered else truth
        y = y.reshape(n, -1).astype(float)
        ysum = weights @ y if background_centered else None
        yy = weights @ np.einsum("ij,ij->i", y, y)
        if background_centered:
            yy -= np.einsum("ij,ij->i", ysum, ysum) / n
        boot = {}
        for key, val in predictions.items():
            f = val - val.mean(axis=-1, keepdims=True) if horizon_centered else val
            f = f.reshape(n, -1).astype(float)
            fx = weights @ np.einsum("ij,ij->i", f, y)
            ff = weights @ np.einsum("ij,ij->i", f, f)
            if background_centered:
                fsum = weights @ f
                fx -= np.einsum("ij,ij->i", fsum, ysum) / n
                ff -= np.einsum("ij,ij->i", fsum, fsum) / n
                ff = np.maximum(0, ff)
            boot[key] = metrics(np.column_stack((fx, yy, ff)))
        def summarize(values):
            return {"value": float(values[0]), "ci95": np.quantile(values[1:], [.025, .975]).tolist()}
        out[view] = dict(methods={k: {metric: summarize(v) for metric, v in vals.items()}
                                  for k, vals in boot.items()},
                         bridge_minus={k: summarize(boot["bridge"]["recovery"] - vals["recovery"])
                                       for k, vals in boot.items() if k != "bridge"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=REVISIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    target_name = args.target
    source_name = next(name for name in REVISIONS if name != target_name)
    stem = source_name + "_to_" + target_name
    with np.load(args.cache / "data.npz") as data:
        x, split, ids = data["x"], data["split"], data["ids"]
    with np.load(args.cache / (target_name + ".npz")) as data:
        target = dict(data)
    with np.load(args.cache / (source_name + ".npz")) as data:
        source = dict(data)
    original = dict(np.load(args.out / (stem + ".npz")))
    train = split == "train"
    bridge, _ = fit_bridge(source["clean"][train], target["clean"][train])
    adapter = load_adapter(target_name)
    rows = {}
    summaries = {}
    for group in ("daily", "weekly"):
        idx = np.flatnonzero(split == group)
        selected = original["split"] == group
        np.testing.assert_array_equal(ids[idx], original["ids"][selected])
        donor, pairs = donors_and_clusters(len(idx), 2026112005 if group == "daily" else 2026112006)
        original_effects = {k: v[selected] for k, v in original.items()
                            if k not in ("ids", "split", "concepts", "native")}
        transported = bridge.transport(source["delta"][idx])
        vectors = {"bridge_donor": transported[donor],
                   "bridge_donor_energy": scaled_like(transported[donor], transported),
                   "target_native_donor": target["delta"][idx][donor]}
        measured = {}
        for name, edit in vectors.items():
            print(group, target_name, name, flush=True)
            measured[name] = signed_response(adapter, x[idx], edit)
        template = target["native"][train].mean(axis=0, dtype=float)
        measured["forecast_template"] = np.broadcast_to(template, original["native"][selected].shape)
        summaries[group] = bootstrap_summary(
            {**original_effects, **measured}, original["native"][selected], pairs)
        rows[group] = {**measured, "donor_ids": ids[idx][donor], "ids": ids[idx]}
    all_keys = rows["daily"].keys()
    np.savez_compressed(args.out / (stem + "_controls.npz"),
                        **{k: np.concatenate([rows["daily"][k], rows["weekly"][k]]) for k in all_keys})
    (args.out / (stem + "_controls.json")).write_text(json.dumps(summaries, separators=(",", ":"),
                                                                allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
