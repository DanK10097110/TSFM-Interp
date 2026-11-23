"""Whole-series conditional bootstrap of support retention, identity switches and observed forecast skill."""
import json

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.sparse_stability.prepare import DOSES, N, OUT


def rat(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 1e-16)


def interval(draws):
    if not np.isfinite(draws[0]):
        return {"value": None, "ci95": None, "defined_bootstraps": 0}
    valid = draws[1:][np.isfinite(draws[1:])]
    return {"value": float(draws[0]), "ci95": np.quantile(valid, [.025, .975]).tolist(),
            "defined_bootstraps": len(valid)}


def rate(numer, denom, draws):
    def rows(arr):
        return np.asarray(arr, dtype=float).reshape(len(arr), -1).sum(1)
    a, b = rows(numer), rows(denom)
    return dict(numerator=int(a.sum()), denominator=int(b.sum()), **interval(rat(draws @ a, draws @ b)))


def score(mask, reference, arms, draws):
    total = {"pairs": int(mask.sum()), "series": int(mask.any(1).sum())}
    for center in (False, True):
        def transform(a):
            v = a.astype(float)
            return v - v.mean(-1, keepdims=True) if center else v
        ref = transform(reference)
        denom = draws @ (mask[:, :, None] * ref**2).sum((1, 2))
        sims = {}
        for method, wave in arms.items():
            loss = (transform(wave) - ref)**2
            sims[method] = 1. - rat(draws @ (mask[:, :, None] * loss).sum((1, 2)), denom)
        if {"reselect", "carry"} <= sims.keys():
            sims["reselect_minus_carry"] = sims["reselect"] - sims["carry"]
        total["shape" if center else "full"] = {k: interval(v) for k, v in sims.items()}
    return total


def partition(common, previous, current, prior_support, new_support):
    old_survives = np.any(previous[:, :, None] == new_support[:, None, :], axis=-1)
    new_existed = np.any(current[:, :, None] == prior_support[:, None, :], axis=-1)
    switched = common & (previous != current)
    return dict(switched=switched, lost=switched & (previous >= 0) & ~old_survives,
                surviving_to_new=switched & old_survives & ~new_existed & (current >= 0),
                both_surviving=switched & old_survives & new_existed,
                new_existed=switched & new_existed, old_survives=switched & old_survives,
                noop_edge=switched & ((previous < 0) | (current < 0)))


def summarize(source):
    with np.load(OUT / (source + ".npz")) as f:
        data = dict(f)
    draws = np.vstack([np.ones(N), resampling_counts(N)])
    original_active = data["clean_codes"] > 0
    old_target = data["clean_support"]
    output = {}
    for k, amplitude in enumerate(DOSES):
        sl = slice(k * N, (k + 1) * N)
        alive = data["perturbed_codes"][sl] > 0
        common = original_active & alive
        current = data["winner"][sl]
        support = data["perturbed_support"][sl]
        overlap = (old_target[:, :, None] == support[:, None, :]).any(-1)
        parts = partition(common, data["clean_winner"], current, old_target, support)
        switch = parts["switched"]
        if np.any(parts["noop_edge"]):
            raise ValueError("Observed a no-op; inspect overlapping attribution categories before reporting")
        np.testing.assert_array_equal(parts["lost"] | parts["both_surviving"] | parts["surviving_to_new"], switch)
        arms = {"reselect": data["actual_reselect"][sl], "carry": data["actual_carry"][sl],
                "source_linear": data["source_linear"][sl]}
        row = dict(source_survival=rate(common, original_active, draws),
                   new_source_firings=rate(alive & ~original_active, alive, draws),
                   target_members_retained=rate(overlap, np.ones_like(overlap), draws),
                   target_support_changed=rate(~overlap.all(1), np.ones(N), draws),
                   all_switches=rate(switch, common, draws), switch_breakdown={},
                   clean_reference=score(common, data["clean_native"], {"clean_selected": data["clean_effect"]}, draws))
        for group, subset in parts.items():
            if group != "switched":
                row["switch_breakdown"][group] = rate(subset, switch, draws)
        for key, mask in (("all", common), ("switches", switch), ("both_surviving", parts["both_surviving"]),
                          ("old_left", parts["lost"]), ("old_survived_new_entered", parts["surviving_to_new"])):
            row[key] = score(mask, data["actual_native"][sl], arms, draws)
        output[str(amplitude)] = row
    return output


if __name__ == "__main__":
    results = {source: summarize(source) for source in ("bolt", "timesfm")}
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
