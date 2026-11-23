"""Compare globally fixed 32-coordinate bases with per-state SAE selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import RANK, SPLITS, ci, load_study, metric_vectors, pca, train_validate_one
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS, dictionary_path
from research.activation_transfer.boundary.smooth import smooth_reps


def normalized_columns(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)


def popular_atoms(clean_codes, count):
    active = (clean_codes > 0).sum(axis=0)
    if np.count_nonzero(active) < count:
        raise ValueError("too few active SAE columns for the matched fixed budget")
    return np.lexsort((np.arange(len(active)), -clean_codes.mean(axis=0), -active))[:count]


@torch.inference_mode()
def fixed_bases(sae, clean):
    codes = sae.encode(torch.tensor(clean)).numpy()
    enc = sae.W_enc.detach().numpy()
    pool = np.flatnonzero((codes > 0).any(axis=0))
    frequent = popular_atoms(codes, RANK)
    basis = {"frequent_sae": [normalized_columns(enc[:, frequent])],
             "clean_pca": [pca(clean).T], "random_sae": [], "isotropic": []}
    for seed in range(8):
        rng = np.random.default_rng(2026112090 + seed)
        random_atoms = rng.choice(pool, RANK, replace=False)
        basis["random_sae"].append(normalized_columns(enc[:, random_atoms]))
        q, _ = np.linalg.qr(rng.standard_normal((len(enc), RANK)))
        basis["isotropic"].append(q.astype(np.float32))
    meta = {"train_active_pool": len(pool), "popular_indices": frequent.tolist(),
            "popular_train_fraction": (codes[:, frequent] > 0).mean(axis=0).tolist()}
    return basis, meta


def score(arm, target, other, evaluation):
    pred, penalty, fitted_rank = train_validate_one(arm, target, other, RANK)
    pred = pred.reshape(evaluation.shape)
    vec = metric_vectors(pred, evaluation)
    pooled = np.square(pred-evaluation).sum()/np.square(evaluation).sum()
    return vec, float(pooled), penalty, fitted_rank, pred.astype(np.float32)


def evaluate(train, selected, a, b, cached, basis, exports=None):
    src, dst = train[a], train[b]
    te = {k: v[selected] for k,v in cached[a].items()}
    goal = cached[b]["dense"][selected]
    arms = {k: [(src[k], te[k])] for k in ("dense", "code", "shoulder", "subtract", "input")}
    for name, matrices in basis.items():
        arms[name] = [(src["dense"] @ mat, te["dense"] @ mat) for mat in matrices]
    results, vectors = {}, {}
    for name, rows in arms.items():
        scores = [score(x, dst["dense"], xte, goal) for x, xte in rows]
        if exports is not None:
            for index, result in enumerate(scores):
                key = name if len(scores) == 1 else f"{name}_{index}"
                exports[f"{b}_cross_{key}"] = result[4]
        averaged = {key:np.mean([v[0][key] for v in scores],axis=0) for key in scores[0][0]}
        vectors[name] = averaged["cosine"]
        results[name] = {**{key:ci(value) for key,value in averaged.items()},
                         "pooled_mse_ratio":float(np.mean([s[1] for s in scores])),
                         "penalties":[s[2] for s in scores], "ranks":[s[3] for s in scores],
                         "replicate_mean_cosines":[float(s[0]["cosine"].mean()) for s in scores]}
    for x, y in [("frequent_sae","isotropic"), ("random_sae","isotropic"),
                 ("clean_pca","isotropic"), ("frequent_sae","random_sae"),
                 ("frequent_sae","clean_pca"), ("isotropic","code"), ("dense","isotropic")]:
        results[x + "_minus_" + y] = ci(vectors[x]-vectors[y])
    return results


def study(scratch):
    m4 = load_study(scratch, "m4")
    saes = {m:load_sae_checkpoint(str(dictionary_path(m))) for m in MODELS}
    train = {m:smooth_reps(saes[m],m4[m])[0] for m in MODELS}
    bases, metadata = {}, {}
    for m in MODELS:
        bases[m], metadata[m] = fixed_bases(saes[m], m4[m].clean[:SPLITS["train"][1]])
    out = {"metadata":metadata, "datasets":{}}
    for name in ("reserve","m4","weekly","fast"):
        data = m4 if name == "m4" else load_study(scratch, name)
        selected = slice(*SPLITS["test"]) if name == "m4" else slice(None)
        te = train if name == "m4" else {m:smooth_reps(saes[m],data[m])[0] for m in MODELS}
        results = {"ids": data["bolt"].ids[selected].tolist()}
        exports = {"ids":data["bolt"].ids[selected]} if name == "reserve" else None
        for a,b in [("bolt","timesfm"),("timesfm","bolt")]:
            results[f"{a}_to_{b}"] = evaluate(train, selected, a, b, te, bases[a], exports)
            if exports is not None:
                exports[f"{b}_native_dense"] = te[b]["dense"][selected]
        if exports is not None:
            np.savez(scratch/"reserve_fixed_predictions.npz", **exports)
        out["datasets"][name] = results
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.write_text(json.dumps(study(args.scratch), indent=2, allow_nan=False) + "\n")
