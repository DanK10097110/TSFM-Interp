"""The powered per-feature causal test of the ablation battery (ROADMAP.md sec 42 task B).

Why it exists. The legacy battery rule (`response._score_channel_against_null`) asks
whether a feature's MEAN |effect| over its k rows exceeds the p95 of POOLED per-row null
values. Under exchangeability a mean over k rows clears a pooled per-row p95 well below
5% of the time (ROADMAP.md sec 38 round 5: empirical chance 0.0246 per cell), so the rule
is conservative in a way no one can state as a significance level, and with k = 8 rows and
16 null draws it has little power. This module replaces the question by one with a stated
level: how extreme is the feature's observed SERIES-MEAN signed effect among the same
statistic computed for each null draw on the SAME series?

What it inherits. The per-series signed deltas and the per-draw signed null deltas that
`feature_ablation_fingerprints` already computes (row-matched, one null direction per
draw, same forward-pass conventions); nothing is re-run. The series is the resampling
unit throughout (invariant 2): every statistic below is a mean over a feature's own
series, the bootstrap resamples series, and no window is ever a unit.

Per channel. The feature's series-mean `O` is compared with the `D` null series-means
(one per null draw, over the same series). Two p-values are written for every scored
channel, because the one the design first called for cannot be reached:

  * `p_rank`: the rank p. Pool `O` with the `D` null series-means (`D + 1` exchangeable
    values under the null); each member's tail share is the share of the pool at least as
    extreme in ITS OWN direction (above the pool median if it sits above it, else
    below), doubled for a signed channel (two-sided, capped at 1). The observed p is
    `(1 + #null series-means as extreme) / (D + 1)`: floor `1 / (D + 1)` (2/(D+1) for a
    signed channel).
  * `p_t`: the predictive-t p. `z = (O - mean(N)) / sd(N)` over the null series-means `N`
    (`ddof=1`); under a normal null `z / sqrt(1 + 1/D)` is Student-t with `D - 1` degrees
    of freedom, and the p is its two-sided tail (upper tail only for a distance channel).
    It has no floor, which is the point: it can rank an effect many null sds out. A
    distance channel (`horizon_shape_*`, a mean |delta|, >= 0 for the effect AND the null)
    is scored on the log of the series-mean (right-skewed, positive) when every value in
    the pool is positive, and carries no sign. The normality of the null series-means is
    an assumption; the target-level diagnostic `loo_cell_rate_p_t_le_05` (every null draw
    scored against the other `D - 1`, rate of `p_t <= 0.05`, expected near 0.05) and the
    known-answer false-positive rate on the control layer
    (`run_powered_known_answer.py`) are what test it.

Feature level. With `m` scored channels, Holm's first step (the Bonferroni minimum):
`p_holm_t = min(1, m * min_c p_t)` and `p_holm_rank = min(1, m * min_c p_rank)`, then BH
(q 0.05) across the target's features on each (`causal`, `causal_rank`). Why both:
`p_holm_rank` can never fall below `m / (D + 1)` (0.1385 at D = 64, m = 9), so no feature
passes it at the planned design, and BH across a target's `F` features needs about
`F * (1/(D+1)) / q` features at the floor before it rejects any (77 of 249 at D = 64),
which a handful of planted atoms cannot supply. The literal reading is reported so that
fact is a number; `p_holm_t` is the verdict (`causal`). Channels are dependent (columns
of the same draws); Bonferroni needs no independence, at some cost in power.

Degenerate channels. A null whose series-means have no spread (every draw moved the
channel by the same amount, e.g. exactly 0) cannot refer an observation: any movement at
all would sit in the tail by itself (CLAUDE.md sec 8, "Absolute epsilons"). The channel
is excluded and the reason recorded, so a constant channel produces no p-value, not a
ramp of tied ones.

Effect size. `signed_effect / null_sd`: `O` over the standard deviation of the null
series-means, with a 95% percentile CI. For a distance channel it is a ratio of
non-negative quantities and the sign carries no meaning (`direction` is None). The CI bootstraps the observed mean over series and the null sd over null draws
(`_mean_ci` says why the null means are not re-formed from resampled series).

Evidence class: behavioral / causal within-model, exactly the legacy battery's; this is
a better test of the same intervention, not a different one.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import false_discovery_control, t as student_t

DISTANCE_CHANNELS = ("horizon_shape_near", "horizon_shape_far")
MIN_DRAWS = 3
DEFAULT_N_BOOT = 500
DEFAULT_Q = 0.05
CALIBRATION_ALPHA = 0.05


def rank_pvalue(obs: float, null: np.ndarray, distance: bool) -> float:
    """The rank p of `obs` against the null series-means: `(1 + #null as extreme) / (D + 1)`
    in the observation's own direction (above or below the pool median), doubled and capped
    at 1 for a signed channel, upper-tailed only for a distance channel. Ties count as
    extreme (conservative)."""
    pool = np.concatenate([[obs], np.asarray(null, dtype=np.float64)])
    upper = float((pool >= obs).sum()) / pool.size
    if distance:
        return upper
    lower = float((pool <= obs).sum()) / pool.size
    return min(1.0, 2.0 * (upper if obs >= np.median(pool) else lower))


def predictive_t_pvalue(obs: float, null: np.ndarray, distance: bool) -> float:
    """The predictive-t p of `obs` against the null series-means (see the module
    docstring). The null must have spread; the caller checks. Two-sided for a signed
    channel, upper-tailed for a distance channel."""
    null = np.asarray(null, dtype=np.float64)
    d = null.size
    z = (obs - null.mean()) / (null.std(ddof=1) * np.sqrt(1.0 + 1.0 / d))
    return float(student_t.sf(z, d - 1) if distance else 2.0 * student_t.sf(abs(z), d - 1))


def loo_calibration(null: np.ndarray, distance: bool) -> tuple:
    """`(n_cells, n_p_t_le_alpha)`: every null series-mean scored against the other `D - 1`
    by `predictive_t_pvalue`, counted at `CALIBRATION_ALPHA`. Under a null drawn from the
    t model the share is about alpha; a clearly larger share says the null's tails are
    heavier than the t model and that `p_t` is anti-conservative."""
    null = np.asarray(null, dtype=np.float64)
    n_hit = 0
    for j in range(null.size):
        rest = np.delete(null, j)
        if rest.std(ddof=1) > 0:
            n_hit += int(predictive_t_pvalue(null[j], rest, distance) <= CALIBRATION_ALPHA)
    return int(null.size), n_hit


def _mean_ci(delta: np.ndarray, null_means: np.ndarray, n_boot: int, rng) -> tuple:
    """Percentile CI of `observed series-mean / null sd`. The observed mean is bootstrapped
    over SERIES (the resampling unit, invariant 2); the null sd is bootstrapped over null
    DRAWS (its own sampling error, about 9% at D = 64). The null series-means are not
    re-formed from resampled series: a resample with repeated series has unequal weights,
    which inflates the sd of a null mean and puts the point estimate outside its own
    interval (measured on the first version). `(lo, hi)` or `(None, None)`."""
    n, d = delta.size, null_means.size
    idx = rng.integers(0, n, size=(n_boot, n))
    jd = rng.integers(0, d, size=(n_boot, d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        obs_b = np.nanmean(delta[idx], axis=1)
        sd_b = null_means[jd].std(axis=1, ddof=1)
    ok = np.isfinite(obs_b) & np.isfinite(sd_b) & (sd_b > 0)
    if ok.sum() < max(20, n_boot // 5):
        return None, None
    lo, hi = np.quantile(obs_b[ok] / sd_b[ok], [0.025, 0.975])
    return float(lo), float(hi)


def powered_feature(delta_by_channel: dict, draws_by_channel: dict, channels: tuple,
                    n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> dict:
    """The powered verdict inputs of ONE feature.

    `delta_by_channel[ch]`: `[n]` signed per-series effect on the feature's own series.
    `draws_by_channel[ch]`: `[D, n]` signed per-series effect of each null draw on the SAME
    series. A channel missing from either, or with fewer than `MIN_DRAWS` draws, or with a
    null of no spread, is recorded unscored with its reason.

    Returns `{"scored": bool, "n_null": D, "n_channels_scored": m, "channels": {...},
    "p_holm_t": ..., "p_holm_rank": ..., "min_possible_p_holm_rank": m / (D + 1),
    "loo_cells": ..., "loo_cells_p_t_le_05": ...}`; `scored` False carries a `reason`
    and no p.
    """
    per_channel, scored_ch, p_rank, p_t = {}, [], [], []
    n_null = loo_n = loo_hit = 0
    for ci, ch in enumerate(channels):
        delta = delta_by_channel.get(ch)
        draws = draws_by_channel.get(ch)
        if delta is None or draws is None or len(draws) < MIN_DRAWS:
            per_channel[ch] = {"scored": False,
                               "reason": "channel unavailable or fewer than "
                                         f"{MIN_DRAWS} null draws"}
            continue
        delta = np.asarray(delta, dtype=np.float64)
        draws = np.asarray(draws, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            obs = float(np.nanmean(delta))
            null_means = np.nanmean(draws, axis=1)
        if not np.isfinite(obs) or not np.isfinite(null_means).all():
            per_channel[ch] = {"scored": False,
                               "reason": "no finite series-mean on this feature's series"}
            continue
        sd = float(null_means.std(ddof=1))
        if not (sd > 0.0):
            per_channel[ch] = {"scored": False,
                               "reason": "every null draw moved this channel by the same "
                                         "amount, so there is no spread to refer an effect to"}
            continue
        distance = ch in DISTANCE_CHANNELS
        use_log = bool(distance and obs > 0 and (null_means > 0).all())
        t_obs, t_null = ((np.log(obs), np.log(null_means)) if use_log else (obs, null_means))
        if not (t_null.std(ddof=1) > 0.0):
            per_channel[ch] = {"scored": False,
                               "reason": "the null series-means have no spread on the scale "
                                         "the t p-value is computed on"}
            continue
        rng = np.random.default_rng([int(seed), ci])
        lo, hi = _mean_ci(delta, null_means, n_boot, rng)
        n_null = int(null_means.size)
        pr = rank_pvalue(obs, null_means, distance)
        pt = predictive_t_pvalue(t_obs, t_null, distance)
        n_cells, n_hit = loo_calibration(t_null, distance)
        loo_n += n_cells
        loo_hit += n_hit
        per_channel[ch] = {
            "scored": True, "p_rank": pr, "p_t": pt,
            "direction": (None if distance else int(np.sign(obs - np.median(null_means)))),
            "signed_effect": obs, "null_mean": float(null_means.mean()),
            "null_sd": sd, "effect_over_null_sd": obs / sd,
            "effect_over_null_sd_ci95": [lo, hi],
            "distance_channel": distance, "t_scale": "log" if use_log else "raw"}
        scored_ch.append(ch)
        p_rank.append(pr)
        p_t.append(pt)
    if not scored_ch:
        return {"scored": False, "n_null": n_null, "n_channels_scored": 0,
                "channels": per_channel,
                "reason": "no channel has a null with spread on this feature's series"}
    m = len(scored_ch)
    return {"scored": True, "n_null": n_null, "n_channels_scored": m,
            "channels": per_channel,
            "p_holm_t": float(min(1.0, m * min(p_t))),
            "p_holm_rank": float(min(1.0, m * min(p_rank))),
            "min_possible_p_holm_rank": float(m / (n_null + 1)),
            "min_channel_t": scored_ch[int(np.argmin(p_t))],
            "loo_cells": int(loo_n), "loo_cells_p_t_le_05": int(loo_hit)}


def annotate_target(blocks: list, q: float = DEFAULT_Q) -> dict:
    """BH across a target's features, in place, on `p_holm_t` (`causal`) and `p_holm_rank`
    (`causal_rank`). `blocks` are the `powered_feature` dicts of the target's scorable
    features; an unscored block gets `causal` False with `q_bh` None. Returns the target
    summary: counts, the level, the smallest p the rank design can reach and the
    calibration diagnostic of the t model."""
    tested = [b for b in blocks if b.get("scored")]
    for b in blocks:
        b.update({"q_bh": None, "causal": False, "q_bh_rank": None, "causal_rank": False})
    if tested:
        for key, qk, ck in (("p_holm_t", "q_bh", "causal"),
                            ("p_holm_rank", "q_bh_rank", "causal_rank")):
            adj = false_discovery_control(np.array([b[key] for b in tested]), method="bh")
            for b, a in zip(tested, adj):
                b[qk] = float(a)
                b[ck] = bool(a <= q)
    loo_n = sum(b.get("loo_cells", 0) for b in tested)
    loo_hit = sum(b.get("loo_cells_p_t_le_05", 0) for b in tested)
    return {"q": float(q), "n_features": len(blocks), "n_tested": len(tested),
            "n_causal": sum(int(b["causal"]) for b in blocks),
            "n_causal_rank": sum(int(b["causal_rank"]) for b in blocks),
            "min_possible_p_holm_rank": (min(b["min_possible_p_holm_rank"] for b in tested)
                                         if tested else None),
            "loo_cells": int(loo_n),
            "loo_cell_rate_p_t_le_05": (loo_hit / loo_n if loo_n else None)}


def powered_causal(candidate: dict) -> bool:
    """Whether a candidate record carries a positive powered verdict (`causal`)."""
    return bool((candidate.get("powered") or {}).get("causal"))
