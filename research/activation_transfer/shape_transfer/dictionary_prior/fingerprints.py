"""After the primary match, isolate centered temporal shape and unsigned magnitude for self identification."""
import json

import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.run import OUT, load_target
from research.activation_transfer.shape_transfer.feature_groups.score import weights
from research.activation_transfer.shape_transfer.sparse_stability.score import interval, rate, rat


def analyze(model):
    data = dict(np.load(OUT / f"{model}_to_{model}.npz"))
    receiver = load_target(model)
    np.testing.assert_array_equal(data["ids"], receiver["ids"])
    live = data["active"]
    hit_shape = np.zeros_like(live)
    hit_rms = np.zeros_like(live)
    hit_primary = np.zeros_like(live)
    degenerate = np.zeros_like(live)
    for row, support in enumerate(receiver["support"]):
        predicted = -receiver["doses"][row, :, None] * (receiver["decoder"][support] @ receiver["jac"][row].T)
        norms = np.linalg.norm(predicted, axis=1)
        centered = predicted - predicted.mean(1, keepdims=True)
        lengths = np.linalg.norm(centered, axis=1)
        for j in np.flatnonzero(live[row]):
            actual = data["measured_source"][row, j, :8].astype(float)
            temporal = actual - actual.mean()
            true_id = data["atoms"][j]
            cosines = (centered @ temporal) / np.maximum(lengths * np.linalg.norm(temporal), 1e-25)
            hit_shape[row, j] = support[cosines.argmax()] == true_id
            hit_rms[row, j] = support[np.abs(norms - np.linalg.norm(actual)).argmin()] == true_id
            hit_primary[row, j] = support[data["natural_rank"][row, j, 0]] == true_id
            degenerate[row, j] = min(lengths.min(), np.linalg.norm(temporal)) <= 1e-12
    results = {}
    for cohort in ("daily", "extra_daily"):
        rows = data["split"] == cohort
        draw = weights(int(rows.sum()))
        active = live[rows]
        denom = draw @ active.sum(1)
        points = {}
        for key, hit in (("shape_only", hit_shape), ("rms_only", hit_rms), ("natural", hit_primary)):
            points[key] = rate(hit[rows], active, draw)
        delta = (hit_shape[rows].astype(int) - hit_rms[rows]).sum(1)
        points["shape_minus_rms"] = interval(rat(draw @ delta, denom))
        points["degenerate_cases"] = int(degenerate[rows].sum())
        results[cohort] = points
    return results


if __name__ == "__main__":
    scores = {model: analyze(model) for model in ("bolt", "timesfm")}
    (OUT / "fingerprints.json").write_text(json.dumps(scores, indent=2) + "\n")
