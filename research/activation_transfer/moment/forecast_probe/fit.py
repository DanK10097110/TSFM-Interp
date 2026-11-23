"""Fit training-centered multi-output ridge in the dual, with validation-only regularization."""
import gc
import json

import numpy as np

from research.activation_transfer.moment.forecast_probe.capture import (
    CACHE, HEADS, MOMENT, OUT, features, rows,
)
from research.activation_transfer.moment.forecast_quality.data import hash_array

GRID = np.logspace(-6, 1, 8)


def ridge_arms(x, train, labels, last):
    """Use a common eigendecomposition for direct and persistence-residual predictions."""
    if np.unique(train).size != 2:
        raise ValueError("The fitting and regularization splits must both exist")
    x = np.asarray(x, dtype=np.float64)
    mean_x = x[train].mean(0)
    xc = x[train] - mean_x
    val_k = (x[~train] - mean_x) @ xc.T
    eig, basis = np.linalg.eigh(xc @ xc.T)
    eig = eig.clip(0)
    scale = max(eig[-1], 1e-18)
    output = {}
    for arm, target in (("direct", labels), ("residual", labels - last[:, None])):
        mean_y = target[train].mean(0)
        cross = basis.T @ (target[train] - mean_y)
        loss, dual = [], []
        for precision in GRID:
            coeff = basis @ (cross / (eig[:, None] + scale * precision))
            val = val_k @ coeff + mean_y
            loss.append(float(np.mean((val - target[~train])**2)))
            dual.append(coeff)
        best = int(np.argmin(loss))
        w = xc.T @ dual[best]
        b = mean_y - mean_x @ w
        output[arm] = dict(w=w, b=b, dual=dual[best], mean_y=mean_y,
            meta=dict(relative=float(GRID[best]), max_gram_eigenvalue=float(scale),
                      selected_validation_mse=loss[best], normalized_validation_mse=loss))
    return output


def design(raw, mu, sd):
    return (raw - mu[:, None]) / sd[:, None]


def raw_forecast(z, w, b, mu, sd, last_raw, residual):
    value = (z @ w + b) * sd[:, None] + mu[:, None]
    if residual:
        value += (last_raw - mu)[:, None]
    return value


def fit():
    ids, lag, target, split = rows("train")
    manifest = json.loads((MOMENT / "head_test" / "results" / "manifest.json").read_text())
    np.testing.assert_array_equal(ids, manifest["ids"])
    np.testing.assert_array_equal(split, manifest["split"])
    assert manifest["context_sha256"] == hash_array(np.concatenate([lag, target], axis=1))
    tr = split == "train"
    assert (int(tr.sum()), int((~tr).sum())) == (1024, 128)
    h, mu, sd = features("train")
    past = design(lag, mu, sd)
    labels = design(target, mu, sd)
    with np.load(HEADS / "large_0/terminal.npz") as masked_train:
        np.testing.assert_array_equal(masked_train["ids"], ids)
        np.testing.assert_array_equal(masked_train["split"], split)
        np.testing.assert_array_equal(masked_train["mu"], mu)
        np.testing.assert_array_equal(masked_train["sd"], sd)
        np.testing.assert_array_equal(masked_train["y"].reshape(labels.shape), labels)
    weights, compact, meta = {}, dict(training_ids=ids[tr]), {
        "matching_previous_terminal_targets": {"n_histories":len(ids), "exact_labels":int(labels.size),
                                                 "target_and_normalization_max_discrepancy":0.0}}

    for desc, x in (("hidden", h), ("lag464", past[:, -464:]), ("lag32", past[:, -32:])):
        models = ridge_arms(x, tr, labels, past[:, -1])
        for arm, result in models.items():
            tag = desc + "_" + arm
            weights[tag + "_w"] = result["w"]
            weights[tag + "_b"] = result["b"]
            compact[tag + "_dual"] = result["dual"]
            compact[tag + "_mean_y"] = result["mean_y"]
            meta[tag] = result["meta"]
        print(desc, [(k, a["meta"]["relative"], a["meta"]["selected_validation_mse"]) for k, a in models.items()], flush=True)
    np.savez(CACHE / "weights.npz", **weights)
    np.savez_compressed(OUT / "dual.npz", **compact)
    (OUT / "fit.json").write_text(json.dumps(meta, indent=2) + "\n")


def predict():
    with np.load(CACHE / "weights.npz") as f:
        heads = dict(f)
    for cohort in ("test", "weather"):
        ids, lag, target, domain = rows(cohort)
        h, mu, sd = features(cohort)
        raw = lag.reshape(-1, lag.shape[-1])
        past = design(raw, mu, sd)
        pred = {}
        for desc, x in (("hidden", h), ("lag464", past[:, -464:]), ("lag32", past[:, -32:])):
            for arm in ("direct", "residual"):
                tag = desc + "_" + arm
                output = raw_forecast(x, heads[tag + "_w"], heads[tag + "_b"], mu, sd, raw[:, -1], arm == "residual")
                pred[tag] = output.reshape(target.shape)
        np.savez_compressed(OUT / f"{cohort}_pred.npz", ids=ids, domain=domain, **pred)
        del h, x
        gc.collect()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["fit", "predict"])
    if p.parse_args().mode == "fit":
        fit()
    else:
        predict()
