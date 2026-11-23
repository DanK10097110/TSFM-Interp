"""Patch a predicted target-window displacement and measure the actual frozen forecast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.extraction.hooks import token_patch

from research.activation_transfer.boundary.analyze import SPLITS, ci, metric_vectors, read_data
from research.activation_transfer.boundary.capture import ALL_MODELS, DEFAULT_SCRATCH, FAMILIES, H, adapter_for


def last_window_slice(adapter, positions):
    expected = np.arange(positions[0], positions[-1] + 1)
    np.testing.assert_array_equal(positions, expected)
    def select(live_length):
        valid = np.arange(live_length)[adapter.token_slice(live_length)]
        selected = valid[positions]
        np.testing.assert_array_equal(np.diff(selected), np.ones(len(selected)-1, dtype=int))
        return slice(int(selected[0]), int(selected[-1])+1)
    return select


def change_forecasts(enc, layer, inputs, tokens, displacement, selector, batch=8):
    """A common vector is added to each native token covered by the final study window."""
    outputs = []
    for start in range(0, len(inputs), batch):
        xs = inputs[start:start+batch]
        if displacement is None:
            y = enc.predict(xs, enc.data_cfg.horizon, [.5])["quantiles"][:, :H, 0]
        else:
            replacement = torch.as_tensor(tokens[start:start+batch] +
                                          displacement[start:start+batch, None, :])
            with token_patch(enc.module, layer, selector, replacement):
                y = enc.predict(xs, enc.data_cfg.horizon, [.5])["quantiles"][:, :H, 0]
        outputs.append(y)
    return np.concatenate(outputs)


def compare(pred, truth):
    absolute = metric_vectors(pred, truth)
    shape_p = pred-pred.mean(-1, keepdims=True)
    shape_q = truth-truth.mean(-1, keepdims=True)
    scores = dict(absolute)
    scores["centered_cosine"] = metric_vectors(shape_p, shape_q)["cosine"]
    scores["rms"] = np.sqrt(np.mean(pred.astype(np.float64)**2, axis=-1))
    results = {key: ci(values) for key, values in scores.items()}
    pred64, ref64 = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    results["pooled_relative_rmse"] = float(np.sqrt(np.sum((pred64-ref64)**2) /
                                                    max(np.sum(ref64**2), 1e-24)))
    results["pooled_gain"] = float(np.sum(pred64*ref64)/max(np.sum(ref64**2), 1e-24))
    results["series_cosine"] = scores["cosine"].mean(axis=1).tolist()
    return results


def patch_dataset(dataset, model, scratch, suffix=""):
    states = read_data(scratch, dataset, model)
    test = slice(*SPLITS["test"]) if dataset == "m4" else slice(None)
    ids, inputs, clean = states.ids[test], states.raw[test], states.tokens[test]
    enc = adapter_for(model)
    layer = ALL_MODELS[model][-1]
    selector = last_window_slice(enc, states.positions)
    with np.load(scratch/f"{dataset}{suffix}_predictions.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(ids, f["ids"])
        moves = {k.removeprefix(model+"_"): f[k] for k in f.files if k.startswith(model+"_")}
    baseline = change_forecasts(enc, layer, inputs, clean, None, selector)
    zeros = np.zeros((len(inputs), clean.shape[-1]), dtype=np.float32)
    replay = change_forecasts(enc, layer, inputs, clean, zeros, selector)
    floor = np.max(np.abs(replay-baseline))
    if floor > 1e-5:
        raise RuntimeError(f"self-patch replay error {floor} exceeds threshold")
    deltas = {}
    for name, dx in moves.items():
        group = []
        for j in range(len(FAMILIES)):
            group.append(change_forecasts(enc, layer, inputs, clean, dx[:,j], selector)-replay)
        deltas[name] = np.stack(group, axis=1)
    group = []
    for j in range(len(FAMILIES)):
        perturbed = inputs.copy()
        perturbed[:, -64:] += .03*states.cues[test, j]
        group.append(change_forecasts(enc, layer, perturbed, clean, None, selector)-baseline)
    deltas["ordinary_input_change"] = np.stack(group, axis=1)
    reference = deltas["native_dense"]
    results = {name: compare(v, reference) for name,v in deltas.items()}
    pairs = [("native_fixed_decoded","native_decoded"), ("cross_dense","cross_code"),
             ("cross_fixed_code","cross_code"), ("cross_dense","cross_input"),
             ("cross_dense","cross_shuffle")]
    for arm in ("shoulder","subtract"):
        if f"cross_{arm}" in results:
            pairs.extend([(f"cross_{arm}","cross_code"), (f"native_{arm}","native_decoded")])
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        results[a+"_minus_"+b] = ci(np.asarray(results[a]["series_cosine"]) -
                                    np.asarray(results[b]["series_cosine"]))
    results.update(
        self_patch_max=float(floor), n_native_tokens=int(clean.shape[1]),
        reference_near_floor_count=int((np.linalg.norm(reference,axis=-1) < 1e-7).sum()),
        forecast_norm_lt_1e_5=int((np.linalg.norm(reference,axis=-1) < 1e-5).sum()),
        n=int(len(ids)), ids=ids.tolist(), layer=layer, dataset=dataset,
        units="forecast in clean-history standard deviations",
        uncertainty="paired item bootstrap, 4000 percentile draws, averaged over two cues",
        forecast_changes={k:v.tolist() for k,v in deltas.items()})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ALL_MODELS, required=True)
    parser.add_argument("--dataset", default="m4", choices=["m4", "machines", "fast", "weekly", "reserve"])
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--suffix", default="", help="select independently exported transfer maps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = patch_dataset(args.dataset, args.model, args.scratch, args.suffix)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
