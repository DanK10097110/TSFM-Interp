"""Assess local dictionary stability and held-out transfer of hidden-state changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import linalg
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.capture import (DEFAULT_SCRATCH, DOSES, FAMILIES, HERE, MODELS,
                     dictionary_path)

SPLITS = {"train": (0, 128), "valid": (128, 176), "test": (176, 256)}
RANK = 32
GRID = [0., .001, .01, .1, 1., 10.]
BOOT = 4000
EPS = 1e-12


def read_data(scratch, dataset, model):
    with np.load(scratch/f"{dataset}_{model}.npz", allow_pickle=False) as f:
        data = SimpleNamespace(**{k: f[k] for k in f.files})
    data.clean = data.tokens.mean(axis=1)
    return data


def load_study(scratch, dataset):
    data = {m: read_data(scratch, dataset, m) for m in MODELS}
    np.testing.assert_array_equal(data["bolt"].ids, data["timesfm"].ids)
    np.testing.assert_array_equal(data["bolt"].raw, data["timesfm"].raw)
    return data


def metric_vectors(prediction, reference):
    """Return per-observation scores. Zero predictions receive zero cosine/gain."""
    p = np.asarray(prediction, dtype=np.float64)
    q = np.asarray(reference, dtype=np.float64)
    dot = (p*q).sum(-1)
    pn, qn = np.linalg.norm(p, axis=-1), np.linalg.norm(q, axis=-1)
    return {
        "cosine": dot/np.maximum(pn*qn, EPS),
        "gain": dot/np.maximum(qn**2, EPS),
        "relative_error": np.linalg.norm(p-q, axis=-1)/np.maximum(qn, EPS),
    }


def ci(item_scores, seed=20261120):
    values = np.asarray(item_scores).reshape(len(item_scores), -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(BOOT, len(values)))
    lo, hi = np.quantile(np.mean(values[draws], axis=1), [.025, .975])
    return {"mean": float(np.mean(values)), "ci95": [float(lo), float(hi)],
            "n_items": len(values)}


def pca(clean, rank=RANK):
    centered = clean.astype(np.float64) - clean.mean(0, dtype=np.float64)
    _, _, vt = linalg.svd(centered, full_matrices=False, check_finite=False)
    return vt[:rank].astype(np.float32)


@torch.inference_mode()
def representations(sae, h, hp):
    """Exact TopK change and the same dictionary with the clean support held fixed."""
    h0 = torch.as_tensor(h)
    h1 = torch.as_tensor(hp)
    z0 = sae.encode(h0)
    pre1 = sae.pre_activations(h1)
    z1 = sae.sparsify(pre1)
    zlocked = torch.relu(pre1) * (z0 > 0)
    dz = z1-z0
    dzlocked = zlocked-z0
    decoded = dz @ sae.W_dec
    fixed = dzlocked @ sae.W_dec
    return {"dense": (h1-h0).numpy(), "code": dz.numpy(), "fixed_code": dzlocked.numpy(),
            "decoded": decoded.numpy(), "fixed_decoded": fixed.numpy(),
            "switched": ((z0 > 0) != (z1 > 0)).any(-1).numpy(),
            "fraction_left": (((z0 > 0) & (z1 <= 0)).sum(-1) /
                              (z0 > 0).sum(-1).clamp_min(1)).numpy()}


def rep_for_dose(sae, data, dose):
    hp = data.states[:, :, dose, :]
    h = np.broadcast_to(data.clean[:, None, :], hp.shape).copy()
    result = representations(sae, h, hp)
    result["input"] = data.cues*DOSES[dose]
    return result


def shape_average(rows):
    return np.asarray(rows).mean(axis=1)


def within(sae, ds, test, bases, data_fingerprint):
    clean = ds.clean[test]
    z = sae.encode(torch.tensor(clean)).detach().numpy()
    rec = sae.decode(torch.tensor(z)).detach().numpy()
    prior = np.roll(z, 1, axis=0)
    repeat = ds.repeat[test] - clean
    info = {
        "dictionary_size": int(sae.dict_size), "k": int(sae.k), "d": int(sae.d_in),
        "sha256": str(ds.sae_sha256.item()), "clean_contexts_sha256": data_fingerprint,
        "reconstruction_fvu": float(((rec-clean)**2).sum() /
                                    np.maximum(((clean-clean.mean(0))**2).sum(), EPS)),
        "clean_mean_firing_count": float(np.mean((z > 0).sum(-1))),
        "unrelated_fraction_features_left": float(np.mean(
            ((z > 0) & (prior <= 0)).sum(-1)/np.maximum((z > 0).sum(-1), 1))),
        "repeat_max_abs_dense": float(np.abs(repeat).max()),
        "repeat_max_abs_code": float(np.max(np.abs(
            sae.encode(torch.tensor(ds.repeat[test])).detach().numpy() - z))),
        "doses": {}, "series": {},
    }
    rng = np.random.default_rng(9719)
    for idx, dose in enumerate(DOSES):
        r = rep_for_dose(sae, ds, idx)
        dx = r["dense"][test]
        free, fixed = r["decoded"][test], r["fixed_decoded"][test]
        pca_dx = (dx @ bases.T) @ bases
        changed = r["switched"][test]
        denom = np.linalg.norm(dx, axis=-1)
        switch_dx = free-fixed
        metrics = {
            "changed_support": changed, "fraction_left": r["fraction_left"][test],
            "dense_change_over_clean_norm": denom / np.maximum(
                np.linalg.norm(clean, axis=-1)[:, None], EPS),
            "switch_change_over_dense_norm": np.linalg.norm(switch_dx, axis=-1)/
                                             np.maximum(denom, EPS),
        }
        sign = rng.choice([-1., 1.], size=dx.shape[:-1]+(1,))
        for arm, pred in (("natural", free), ("fixed", fixed), ("pca", pca_dx),
                          ("random_sign", dx*sign)):
            for metric, arr in metric_vectors(pred, dx).items():
                metrics[f"{arm}_{metric}"] = arr
        diff = metrics["fixed_cosine"]-metrics["natural_cosine"]
        metrics["fixed_minus_natural_cosine"] = diff
        metrics["fixed_minus_natural_error"] = metrics["fixed_relative_error"]-metrics[
            "natural_relative_error"]
        summary = {key: ci(value) for key, value in metrics.items()}
        summary["near_zero_dense_fraction"] = float((denom < 1e-8).mean())
        summary["conditional"] = {
            label: ({"count": int(mask.sum()), "natural_cosine": float(
                metrics["natural_cosine"][mask].mean()), "fixed_cosine": float(
                    metrics["fixed_cosine"][mask].mean()),
                     "difference": float(diff[mask].mean())} if np.any(mask) else {"count": 0})
            for label, mask in [("switch", changed), ("stable", ~changed)]}
        info["doses"][str(dose)] = summary
        if idx == 1:
            info["series"] = {key: shape_average(values).tolist() for key, values in metrics.items()
                              if key in {"natural_cosine", "fixed_cosine", "pca_cosine",
                                         "changed_support", "fraction_left"}}
    return info


class Ridge:
    """Rank-constrained ridge; form a train Gram matrix so all feature widths use one solver."""
    def __init__(self, X, Y, rank):
        x, y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        self.mean_x, self.mean_y = x.mean(0), y.mean(0)
        self.train = x-self.mean_x
        gram = self.train @ self.train.T
        vals, U = linalg.eigh(gram, check_finite=False)
        keep = np.flatnonzero(vals > max(1e-12, vals[-1]*1e-12))[-rank:]
        self.vals = vals[keep]
        self.U = U[:, keep]
        self.y = self.U.T @ (y-self.mean_y)
        self.scale = float(self.vals.mean()) if len(self.vals) else 1.

    def transform(self, X):
        x = np.asarray(X, np.float64)
        return ((x-self.mean_x) @ self.train.T) @ self.U

    def predict(self, projected, penalty):
        return projected @ (self.y / (self.vals + penalty*self.scale)[:, None]) + self.mean_y


def train_validate_one(source, target, evaluation, rank, permute=None):
    """Fit on the first 128 items, fix the penalty on the next 48, predict once."""
    ntrain = SPLITS["train"][1]
    endval = SPLITS["valid"][1]
    def rows(data):
        return data.reshape(-1, data.shape[-1])

    X = source[:ntrain] if permute is None else source[:ntrain][permute]
    model = Ridge(rows(X), rows(target[:ntrain]), rank)
    projected = model.transform(rows(source[ntrain:endval]))
    yval = rows(target[ntrain:endval])
    losses = [float(np.mean((model.predict(projected, a)-yval)**2)) for a in GRID]
    penalty = GRID[int(np.argmin(losses))]
    return model.predict(model.transform(rows(evaluation)), penalty), penalty, len(model.vals)


def fit_maps(train_rep, train_target, test_rep, test_target, rank=RANK, export=None):
    """Validate every input representation with identical loss, rank, splits and penalty grid."""
    ntrain = SPLITS["train"][1]
    perms = np.random.default_rng(147).permutation(ntrain)
    results, primary = {}, {}
    arms = ["dense", "code", "fixed_code", "decoded", "fixed_decoded", "input", "shuffle"]
    arms.extend(k for k in ("shoulder", "subtract") if k in train_rep)
    for arm in arms:
        X = train_rep["dense" if arm == "shuffle" else arm]
        other = test_rep["dense" if arm == "shuffle" else arm]
        pred, penalty, fitted_rank = train_validate_one(
            X, train_target, other, rank, perms if arm == "shuffle" else None)
        preds = pred.reshape(test_target.shape)
        if export is not None:
            export[arm] = preds.astype(np.float32)
        vals = metric_vectors(preds, test_target)
        entry = {k: ci(v) for k, v in vals.items()}
        entry.update(penalty=penalty, rank=fitted_rank,
                     relative_pooled_mse=float(np.sum((preds-test_target)**2) /
                                               np.maximum(np.sum(test_target**2), EPS)),
                     series_cosine=shape_average(vals["cosine"]).tolist())
        results[arm] = entry
        primary[arm] = vals["cosine"]
    pairs = [("dense", "code"), ("fixed_code", "code"), ("dense", "input"),
             ("dense", "shuffle"), ("fixed_decoded", "decoded")]
    pairs.extend((arm, "code") for arm in ("shoulder", "subtract") if arm in arms)
    for a, b in pairs:
        results[f"{a}_minus_{b}"] = ci(primary[a]-primary[b])
    results["target_norm_lt_1e-8"] = int(np.sum(np.linalg.norm(test_target, axis=-1) < 1e-8))
    return results


def study(dataset, scratch, rank=RANK):
    import hashlib
    import scipy
    import sys
    reference = load_study(scratch, "m4")
    new = reference if dataset == "m4" else load_study(scratch, dataset)
    test = slice(*SPLITS["test"]) if dataset == "m4" else slice(None)
    ids = new["bolt"].ids[test]
    digest_raw = hashlib.sha256(new["bolt"].raw[test].tobytes()).hexdigest()
    out = {"dataset": dataset, "revision": {m: MODELS[m][3] for m in MODELS},
           "sources": {"m4": str(reference["bolt"].source_sha256.item()),
                       dataset: str(new["bolt"].source_sha256.item())},
           "ids": {"m4_training": reference["bolt"].ids[:128].tolist(),
                   "m4_validation": reference["bolt"].ids[128:176].tolist(),
                   "evaluation": ids.tolist()}, "split_unit": "GIFT item",
           "rank": rank, "bootstraps": BOOT, "valid_grid": GRID,
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__},
           "windows_end_exclusive": new["bolt"].ends[test].tolist(),
           "within": {}, "transfer": {}}
    train_rep, test_rep = {}, {}
    for m in MODELS:
        sae = load_sae_checkpoint(str(dictionary_path(m)))
        bases = pca(reference[m].clean[:128], sae.k)
        out["within"][m] = within(sae, new[m], test, bases, digest_raw)
        train_rep[m] = rep_for_dose(sae, reference[m], 1)
        fresh = train_rep[m] if dataset == "m4" else rep_for_dose(sae, new[m], 1)
        test_rep[m] = {k: v[test] for k, v in fresh.items()}
    exports = {"ids": ids}
    for a, b in (("bolt", "timesfm"), ("timesfm", "bolt")):
        pred = {}
        out["transfer"][f"{a}_to_{b}"] = fit_maps(train_rep[a], train_rep[b]["dense"],
                                                   test_rep[a], test_rep[b]["dense"], rank, pred)
        for method, values in pred.items():
            exports[f"{b}_cross_{method}"] = values
        for key in ("dense", "decoded", "fixed_decoded"):
            exports[f"{b}_native_{key}"] = test_rep[b][key]
    np.savez(scratch / f"{dataset}_predictions.npz", **exports)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="m4", choices=["m4", "machines", "fast"])
    p.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=RANK)
    ns = p.parse_args()
    torch.set_num_threads(4)
    ns.output.write_text(json.dumps(study(ns.dataset, ns.scratch, ns.rank), indent=2,
                                    allow_nan=False) + "\n")
