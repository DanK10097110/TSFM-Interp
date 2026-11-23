"""Describe previously identified SAE features with train-selected univariate input statistics."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.replication.data import CACHE, HERE
from tsfm_lens.sae.train import load_sae_checkpoint


def run(cache):
    with np.load(cache / "data.npz") as f:
        x, split, ids = f["x"].astype(float), f["split"], f["ids"]
    recent = x[:, -32:]
    t = np.arange(32) - 15.5
    statistics = dict(last_value=x[:, -1], recent_mean=recent.mean(1),
                      recent_linear_change=(recent @ t) * 31 / np.sum(t*t), recent_std=recent.std(1))
    training = split == "train"
    rows = {}
    report = {}
    for model, atoms in (("timesfm", [742]), ("bolt", [5432, 4813])):
        with np.load(cache / (model + ".npz")) as f:
            np.testing.assert_array_equal(ids, f["ids"])
            h = torch.from_numpy(f["tokens"].mean(1))
        sae = load_sae_checkpoint(str(dictionary_path(model)))
        with torch.inference_mode():
            codes = sae.encode(h).numpy()
        for atom in atoms:
            z = codes[:, atom].astype(float)
            fired = z > 0
            raw_auc = {k: float(roc_auc_score(fired[training], v[training])) for k, v in statistics.items()}
            best = max(raw_auc, key=lambda k: abs(raw_auc[k] - .5))
            sign = 1 if raw_auc[best] >= .5 else -1
            score = sign * statistics[best]
            monotone = IsotonicRegression(out_of_bounds="clip").fit(score[training], z[training])
            predicted = monotone.predict(score)
            key = f"{model}_{atom}"
            rows[key] = z, predicted
            item = dict(training_auc_in_positive_orientation=raw_auc, selected=best, sign=sign, domains={})
            for cohort in ("daily", "machines"):
                keep = split == cohort
                truth = fired[keep]
                error = np.sum((predicted[keep] - z[keep])**2)
                energy = np.sum((z[keep] - z[training].mean())**2)
                item["domains"][cohort] = dict(n=int(keep.sum()), firings=int(truth.sum()),
                    auc=float(roc_auc_score(truth, score[keep])) if np.unique(truth).size==2 else None,
                    coefficient_skill_vs_training_mean=float(1-error/energy) if energy else None)
            report[key] = item
    daily = split == "daily"
    pair = [rows[k] for k in ("timesfm_742", "bolt_5432")]
    observed = [z[daily] for z, p in pair]
    residuals = [(z-p)[daily] for z, p in pair]
    return dict(features=report,training_ids=ids[training].tolist(),daily_ids=ids[daily].tolist(),
                machine_ids=ids[split=="machines"].tolist(),
                daily_pair_coefficient_pearson=float(np.corrcoef(*observed)[0,1]),
                daily_pair_residual_pearson=float(np.corrcoef(*residuals)[0,1]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--output", type=Path, default=HERE / "results" / "feature_inputs.json")
    args = p.parse_args()
    torch.set_num_threads(2)
    args.output.write_text(json.dumps(run(args.cache), indent=2, allow_nan=False)+"\n", encoding="utf-8")
