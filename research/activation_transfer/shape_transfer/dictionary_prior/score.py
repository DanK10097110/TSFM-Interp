"""Score self identification and real early- and late-horizon responses."""
import json

import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.run import ARMS, OUT
from research.activation_transfer.shape_transfer.feature_groups.score import recover, transform, weights
from research.activation_transfer.shape_transfer.sparse_stability.score import interval, rate, rat

COMPARATORS = (
    ("natural_corrected", "dense"), ("natural_corrected", "wrong_corrected_mean"),
    ("natural_corrected", "natural"), ("natural", "dense"), ("natural", "wrong_mean"),
    ("attenuated_corrected", "dense"), ("attenuated_corrected", "natural_corrected"),
    ("attenuated", "natural"),
)


def compare(data, mask, draws, arms, macro):
    eligible = mask.sum(0) >= 8 if macro else mask.any(0)
    active = mask[:, eligible]
    result = {
        "atoms": data["atoms"][eligible].tolist(),
        "cases": int(active.sum()),
        "histories": int(active.any(1).sum()),
    }
    for segment, leads in (("early", slice(0, 8)), ("late", slice(8, 16))):
        for part in ("full", "shape"):
            goal = transform(data["measured_source"][:, eligible, leads], part)
            scores = {}
            for arm in arms:
                predicted = transform(data[arm + "_effect"][:, eligible, leads], part)
                scores[arm] = recover(active, goal, predicted, draws, pooled=not macro)
            scores["wrong_mean"] = .5 * (scores["wrong1"] + scores["wrong2"])
            scores["wrong_corrected_mean"] = .5 * (scores["wrong1_corrected"] + scores["wrong2_corrected"])
            endpoints = {name: interval(values) for name, values in scores.items()}
            for first, second in COMPARATORS:
                delta = scores[first] - scores[second]
                estimate = interval(delta)
                # A one-sided sensitivity across two central contrasts, four ordered pairs and two cohorts.
                if estimate["value"] is not None and (segment, part) == ("late", "shape") and (first, second) in COMPARATORS[:2]:
                    estimate["family_lower_16"] = float(np.nanquantile(delta[1:], .05 / 16))
                endpoints[first + "_minus_" + second] = estimate
            result[segment + "_" + part] = endpoints
    return result


def identification(data, live, draws):
    membership = data["receiver_support"][:, None, :] == data["atoms"][None, :, None]
    if not np.all(membership.any(-1)[live]):
        raise ValueError("Missing true active self candidate")
    true_local = membership.argmax(-1)
    ident = {}
    hits = {}
    for name in ("natural", "attenuated", "level"):
        ranks = data[name + "_rank"]
        pos = (ranks == true_local[:, :, None]).argmax(-1)
        hits[name] = live & (pos == 0)
        per_atom = {
            str(data["atoms"][j]): rate(hits[name][:, j], live[:, j], draws)
            for j in np.flatnonzero(live.any(0))
        }
        ident[name] = dict(
            top1=rate(hits[name], live, draws), top3=rate(live & (pos < 3), live, draws),
            median_rank=float(np.median(pos[live] + 1)),
            mean_rank=interval(rat(draws @ ((pos + 1) * live).sum(1), draws @ live.sum(1))),
            per_atom=per_atom,
        )

    denominator = draws @ live.sum(1)
    for other in ("level", "attenuated"):
        delta = hits["natural"].astype(int) - hits[other].astype(int)
        ident["top1_natural_minus_" + other] = interval(rat(draws @ delta.sum(1), denominator))

    picks = data["natural_rank"][:, :, 0]
    bank_size = live.sum(1)
    # A uniform guess over only the active evaluation bank is expected to get one of its queries right per history.
    informed = interval(rat(draws @ (bank_size > 0).astype(float), draws @ bank_size))
    largest_code = np.argmax(data["receiver_strength"], axis=1)
    ident.update(
        uniform_all32=1/32, bank_informed_uniform=informed,
        largest_current_coefficient=rate(live & (true_local == largest_code[:, None]), live, draws),
        accidental_true_wrong1=rate(live & (((picks + 7) % 32) == true_local), live, draws),
        accidental_true_wrong2=rate(live & (((picks + 17) % 32) == true_local), live, draws),
    )
    return ident, hits["natural"]


def report(source, target):
    with np.load(OUT / f"{source}_to_{target}.npz") as archive:
        all_data = dict(archive)
    arms = list(ARMS) + (["known_capped"] if source == target else [])
    metadata = {"atoms", "radius", "shape_weight", "unique_physical_writes"}
    result = {}
    for domain in ("daily", "extra_daily"):
        rows = all_data["split"] == domain
        data = {key: (values if key in metadata else values[rows]) for key, values in all_data.items()}
        live = data["active"]
        draws = weights(len(live))
        cohort = dict(
            n=len(live), firing_counts=live.sum(0).tolist(), all_firings=int(live.sum()),
            macro=compare(data, live, draws, arms, True), pooled=compare(data, live, draws, arms, False),
            clip_count={arm: int((data[arm + "_clipped"] & live).sum()) for arm in arms},
            attenuation_noop=rate(live & (data["attenuation_fraction"] == 0), live, draws),
            attenuation_median=float(np.median(data["attenuation_fraction"][live])),
        )
        if source == target:
            cohort["identification"], correct = identification(data, live, draws)
            for name, group in (
                ("correct", correct), ("wrong", live & ~correct),
                ("correct_uncapped", correct & ~data["known_capped_clipped"]),
            ):
                cohort[name] = compare(data, group, draws, arms, False)
        result[domain] = cohort
    return result


if __name__ == "__main__":
    scores = {
        f"{source}_to_{target}": report(source, target)
        for source in ("bolt", "timesfm") for target in ("bolt", "timesfm")
    }
    (OUT / "results.json").write_text(json.dumps(scores, allow_nan=False, separators=(",", ":")) + "\n")
