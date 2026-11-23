"""Paired phase and weekly contrasts on the same 96 sensors under three physical policies."""
import json

import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.score import weights
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import N, OUT as ORIGINAL
from research.activation_transfer.shape_transfer.rolling.fixed_2622.factorial.data import OFFSETS, OUT
from research.activation_transfer.shape_transfer.rolling.fixed_2622.factorial.run import ARMS
from research.activation_transfer.shape_transfer.sparse_stability.score import interval


def calendar_contrasts(values):
    """Columns index hours before the tail schedule in the order 0, 12, 168 and 180."""
    base, half, week, both = values.T
    return {
        "halfday_in_tail_week": half - base, "halfday_in_prior_week": both - week,
        "week_at_base_phase": week - base, "week_at_shifted_phase": both - half,
        "halfday_main": .5 * (half - base + both - week),
        "week_main": .5 * (week - base + both - half),
        "interaction": (both - week) - (half - base),
    }


def score():
    data = dict(np.load(OUT / "effects.npz"))
    old_fire = data["source_anchor"].reshape(4, N) > 0
    now_fire = data["source_now"].reshape(4, N) > 0
    common = (old_fire & now_fire).all(0)
    if not common.any():
        raise ValueError("No common firing sensor")
    ref = data["native"].reshape(4, N, 8).astype(float)[:, common]
    groups = [data["ids"][i*N:(i+1)*N] for i in range(4)]
    labels = [[s.split("/", 2)[-1] for s in group] for group in groups]
    if not all(g == labels[0] for g in labels):
        raise ValueError("Sensor pairing differs")
    draws = weights(int(common.sum()))
    out = dict(
        offsets=OFFSETS, common=int(common.sum()), sensors=N,
        anchor_donor_firing=old_fire.sum(1).tolist(), current_donor_firing=now_fire.sum(1).tolist(),
        per_cell_persistent=(old_fire & now_fire).sum(1).tolist(),
        target_anchor_firing=(data["receiver_anchor"].reshape(4, N)[:, common] > 0).sum(1).tolist(),
        target_current_firing=(data["receiver_now"].reshape(4, N)[:, common] > 0).sum(1).tolist(),
        fresh_selected_2622=(data["fresh_id"].reshape(4, N)[:, common] == 2622).sum(1).tolist(),
        median_fixed_bound=np.median(data["anchor_bound"].reshape(4, N)[:, common], axis=1).tolist(),
    )
    for part in ("full", "shape"):
        want = ref if part == "full" else ref - ref.mean(2, keepdims=True)
        denom = draws @ np.square(want).sum(2).T
        results = {}
        decomposition = {}
        for arm in ARMS:
            obs = data[arm + "_effect"].reshape(4, N, 8).astype(float)[:, common]
            if part == "shape":
                obs = obs - obs.mean(2, keepdims=True)
            results[arm] = 1. - (draws @ np.square(want - obs).sum(2).T) / denom
            source_energy = np.square(want).sum((1, 2))
            effect_energy = np.square(obs).sum((1, 2))
            overlap = (want * obs).sum((1, 2))
            cosine = overlap / np.sqrt(source_energy * effect_energy)
            ratio = np.sqrt(effect_energy / source_energy)
            np.testing.assert_allclose(2. * cosine * ratio - ratio**2, results[arm][0])
            decomposition[arm] = {
                "cosine": cosine.tolist(), "norm_ratio": ratio.tolist(),
                "source_rms": np.sqrt(np.square(want).mean((1, 2))).tolist(),
                "effect_rms": np.sqrt(np.square(obs).mean((1, 2))).tolist(),
            }
        results["fixed_minus_fresh"] = results["fixed"] - results["fresh"]
        results["fixed_minus_sham"] = results["fixed"] - results["sham"]
        results["fixed_minus_fresh_common_denominator"] = (
            results["fixed_minus_fresh"] * denom / denom.mean(1, keepdims=True)
        )
        points = {key: [interval(values[:, col]) for col in range(4)] for key, values in results.items()}
        contrasts = {}
        for key, values in results.items():
            diffs = calendar_contrasts(values)
            contrasts[key] = {name: interval(delta) for name, delta in diffs.items()}
        out[part] = {"cells": points, "contrasts": contrasts, "decomposition": decomposition}
    # Direct archive comparison on the offset zero 96 sensors, including every actual output.
    with np.load(ORIGINAL / "effects.npz") as old, np.load(ORIGINAL / "norm_null.npz") as sham:
        idx = list(old["atoms"]).index(int(data["donor"]))
        errors = {}
        for arm, array in (
            ("native", old["native"]), ("fresh_effect", old["fresh_effect"]),
            ("fixed_effect", old["anchor_effect"]), ("sham_effect", sham["norm1_effect"]),
        ):
            err = float(np.max(np.abs(data[arm][:N] - array[:N, idx])))
            errors[arm] = err
            if err > 8e-5:
                raise ValueError(f"Tail {arm} changed by {err}")
        out["tail_prior_archive_max_abs"] = errors
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    score()
