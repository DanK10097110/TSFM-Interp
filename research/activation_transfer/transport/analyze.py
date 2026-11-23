"""Fit clean-state bridges and compare central, intervened forecast vectors on held-out series."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import eigh

from research.activation_transfer.transport.prepare import CONCEPTS, DEFAULT_CACHE, DOSE


@dataclass
class Ridge:
    weights: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray

    def predict(self, x):
        """Map original states with the fitted intercept."""
        return (x - self.source_mean) @ self.weights + self.target_mean

    def transport(self, contrasts):
        """An affine intercept cancels under a central difference."""
        return contrasts @ self.weights


class KernelFit:
    """Solve multivariate ridge in a train-item kernel, applying one ridge per full map."""
    def __init__(self, x, y):
        self.source_mean = x.mean(axis=0, dtype=np.float64)
        self.target_mean = y.mean(axis=0, dtype=np.float64)
        self.x = x - self.source_mean
        y_center = y - self.target_mean
        k = self.x @ self.x.T
        eigen, basis = eigh(k)
        self.eigen = np.maximum(eigen, 0)
        self.scale = max(np.trace(k) / len(k), 1e-12)
        self.basis = basis
        self.projected_y = basis.T @ y_center

    def coef(self, relative_penalty):
        return self.basis @ (self.projected_y / (self.eigen[:, None] + relative_penalty * self.scale))

    def validation_mse(self, x, y, relative_penalty):
        projected_x = (x - self.source_mean) @ self.x.T
        predicted = projected_x @ self.coef(relative_penalty) + self.target_mean
        return float(np.mean(np.square(predicted - y)))

    def model(self, relative_penalty):
        return Ridge(self.x.T @ self.coef(relative_penalty), self.source_mean, self.target_mean)


def fit_bridge(x, y):
    """Select the regularizer by four series-wise folds scored only on clean target states."""
    penalties = np.asarray([1e-4, 1e-3, .01, .1, 1, 10])
    folds = np.arange(len(x)) % 4
    losses = np.empty((4, len(penalties)))
    for k in range(4):
        fit = KernelFit(x[folds != k], y[folds != k])
        losses[k] = [fit.validation_mse(x[folds == k], y[folds == k], p) for p in penalties]
    index = int(np.argmin(losses.mean(axis=0)))
    best = float(penalties[index])
    return KernelFit(x, y).model(best), dict(relative_penalty=best, penalties=penalties.tolist(),
                                             mean_fold_mse=losses.mean(axis=0).tolist())


def scaled_like(edits, reference):
    """Scale each individual alternative to a common positive per-item, per-contrast energy."""
    norms = np.linalg.norm(edits, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("a zero candidate cannot be norm matched")
    return edits * np.linalg.norm(reference, axis=-1, keepdims=True) / norms


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-8)


def make_edits(source, target, x, waves, train, test, seed=2026112003):
    """Create each intervention without inspecting any held-out native target edit."""
    bridge, settings = fit_bridge(source["clean"][train], target["clean"][train])
    perm = np.random.default_rng(seed).permutation(train)
    shuffled, sh_settings = fit_bridge(source["clean"][train], target["clean"][perm])
    raw, raw_settings = fit_bridge(x[train], target["clean"][train])
    transported = bridge.transport(source["delta"][test])
    shuffled_delta = shuffled.transport(source["delta"][test])
    raw_delta = (zscore(x[test, None, :] + DOSE * waves) - zscore(x[test, None, :] - DOSE * waves)) / 2
    raw_delta = raw.transport(raw_delta)
    train_mean = np.broadcast_to(target["delta"][train].mean(axis=0), transported.shape)
    alternatives = dict(bridge=transported, shuffle=shuffled_delta, raw=raw_delta, family_mean=train_mean)
    for label, vector in list(alternatives.items())[1:]:
        alternatives[label + "_energy"] = scaled_like(vector, transported)
    clean_fit = bridge.predict(source["clean"][test])
    centred = target["clean"][test] - target["clean"][train].mean(axis=0)
    extra = dict(settings=dict(bridge=settings, shuffled=sh_settings, raw=raw_settings),
                 clean_errors=(np.sum((clean_fit - target["clean"][test]) ** 2, axis=1),
                               np.sum(centred ** 2, axis=1)))
    return alternatives, extra


def sufficient(predicted, actual):
    """Per-series inner products allow paired cluster bootstrap of all energy-based metrics."""
    axes = tuple(range(1, predicted.ndim))
    return np.stack([np.sum(predicted * actual, axis=axes), np.sum(actual * actual, axis=axes),
                     np.sum(predicted * predicted, axis=axes)], axis=1, dtype=np.float64)


def metrics(stat):
    xy, yy, xx = np.moveaxis(stat, -1, 0)
    eps = 1e-20
    return {"recovery": (2 * xy - xx) / np.maximum(yy, eps),
            "gain": xy / np.maximum(yy, eps), "cosine": xy / np.sqrt(np.maximum(yy * xx, eps)),
            "rms_ratio": np.sqrt(xx / np.maximum(yy, eps))}


def pooled_evaluation(predictions, native, draws):
    """Summarize paired uncertainty and separate forecast shape from a common level shift."""
    groups, bootstraps = {}, {}
    for label, estimate in predictions.items():
        groups[label], bootstraps[label] = {}, {}
        for name, strip_level in (("full", False), ("shape", True)):
            p = estimate - estimate.mean(axis=-1, keepdims=True) if strip_level else estimate
            n = native - native.mean(axis=-1, keepdims=True) if strip_level else native
            stat = sufficient(p.astype(float), n.astype(float))
            point = metrics(stat.sum(axis=0))
            boot = metrics(stat[draws].sum(axis=1))
            groups[label][name] = {key: {"value": float(val),
                                         "ci95": np.quantile(boot[key], [.025, .975]).tolist()}
                                    for key, val in point.items()}
            bootstraps[label][name] = boot["recovery"]
    paired = {}
    for label in predictions:
        if label == "bridge":
            continue
        paired[label] = {}
        for name in ("full", "shape"):
            score = groups["bridge"][name]["recovery"]["value"] - groups[label][name]["recovery"]["value"]
            diff = bootstraps["bridge"][name] - bootstraps[label][name]
            paired[label][name] = {"value": score, "ci95": np.quantile(diff, [.025, .975]).tolist()}
    return dict(methods=groups, bridge_minus=paired)


def main():
    from research.activation_transfer.transport.capture import REVISIONS, load_adapter, signed_response
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=REVISIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target_name = args.target
    source_name = next(name for name in REVISIONS if name != target_name)
    with np.load(args.cache / "data.npz") as data:
        x, waves, split, ids = data["x"], data["patterns"], data["split"], data["ids"]
    with np.load(args.cache / (target_name + ".npz")) as data:
        target = dict(data)
    with np.load(args.cache / (source_name + ".npz")) as data:
        source = dict(data)
    train, test = np.flatnonzero(split == "train"), np.flatnonzero(split != "train")
    if set(ids[train]) & set(ids[test]) or len(set(ids)) != len(ids):
        raise ValueError("IDs overlap")
    edits, extra = make_edits(source, target, x, waves, train, test)
    adapter = load_adapter(target_name)
    result = dict(source=source_name, target=target_name, metadata=json.loads(str(target["metadata"])),
                  source_metadata=json.loads(str(source["metadata"])), cv=extra["settings"],
                  data_manifest="manifest.json", splits={})
    forecasts, hidden_stats = {}, {}
    for label, edit in edits.items():
        print(f"{source_name}->{target_name}: patching {label}", flush=True)
        forecasts[label] = signed_response(adapter, x[test], edit, batch_size=args.batch_size)
        hidden_stats[label] = sufficient(edit, target["delta"][test])
    forecasts["native"] = target["native"][test]
    forecasts["input_edit"] = target["input_effect"][test]
    forecasts["source_native"] = source["native"][test]
    archive = args.out / (source_name + "_to_" + target_name + ".npz")
    np.savez_compressed(archive, ids=ids[test], split=split[test], concepts=CONCEPTS, **forecasts)
    result["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    for group in ("daily", "weekly"):
        mask = split[test] == group
        draws = np.random.default_rng(2026112004).integers(mask.sum(), size=(4000, mask.sum()))
        scored = {k: v[mask] for k, v in forecasts.items() if k != "native"}
        native = forecasts["native"][mask]
        point = pooled_evaluation(scored, native, draws)
        point["by_concept"] = {c: pooled_evaluation({k: v[:, i] for k, v in scored.items()},
                                                    native[:, i], draws)
                               for i, c in enumerate(CONCEPTS)}
        point["n_series"] = int(mask.sum())
        point["hidden"] = {k: {m: float(v) for m, v in metrics(s[mask].sum(axis=0)).items()}
                           for k, s in hidden_stats.items()}
        source_reference = pooled_evaluation(
            {"bridge": scored["bridge"], "native_target": native, "shuffle": scored["shuffle"]},
            scored["source_native"], draws)
        point["versus_source_native"] = source_reference
        err, baseline = extra["clean_errors"]
        point["clean_state_recovery"] = float(1 - err[mask].sum() / baseline[mask].sum())
        result["splits"][group] = point
    with (args.out / (source_name + "_to_" + target_name + ".json")).open("w") as handle:
        json.dump(result, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    shutil.copyfile(args.cache / "manifest.json", args.out / "manifest.json")
    for group, item in result["splits"].items():
        print(group, {k: round(v["full"]["recovery"]["value"], 3) for k, v in item["methods"].items()},
              flush=True)


if __name__ == "__main__":
    main()
