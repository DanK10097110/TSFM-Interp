"""ROADMAP.md sec 40 / sec 41 V3-C -- the subspace agreement test: an ESTIMATE of how similar two
models' causal effects of one concept are, on the same series, with a confidence interval.

Why this module exists. `sae/shared_input_agreement.py` (the L5 rung) ablates ONE feature per
side and returns a verdict from two point statistics. FINDINGS MN-30: on a planted pair of
different architectures it called a truly shared concept "same" in 0.4 (one direction) and 0.0
(3-4 directions) of seeds, so its absences across architectures say nothing. Two defects, both
structural: a distributed concept is not one feature (single-feature ablation removes a
fraction of it), and a verdict from a point statistic against a floor gives no way to say
"agree" or "differ" with a stated confidence, nor to say "not enough evidence".

What it inherits. The unit format of `build_units` (a source feature set, a destination feature
set, their targets), `TargetContext` (adapter, SAE, reach gate), `battery_for_set` (the same
9-channel battery, level and shape separated, against the same full-reconstruction baseline,
seeded identically on a sampled model), and `_score_channel_against_null` (the repo's one
channel-gating rule). Nothing there changes.

What is new.

1. **Subspace ablation.** Each side ablates its whole member set together (the SAE
   reconstruction with those features zeroed). Its removal at token `t` is
   `r_t = sum_f z_f(t) w_f`, a vector in the span of the members' decoder rows. Orthonormalize
   that span (`Q`, `d x r`, `r` its numerical rank) and `r_t = Q c_t`.
2. **Subspace-matched null.** `_subspace_null_replacement` removes `Q' c_t` instead, with `Q'` a
   random orthonormal `r`-frame drawn from random combinations of decoder rows. This is an
   isometry of the set's own removal: the SAME per-token removed amount, the SAME within-token
   geometry (angles between its coordinates), a random subspace of equal dimension. For one
   feature it is exactly `_profile_matched_null_replacement`'s construction. It decides whether
   a side's effect is real (`clears` its own null, by default at the level of draws:
   `_score_draw_level`) and is also the matched floor.
3. **Estimate, not verdict.** On the shared series `U` (the union of the two top-series sets,
   as in the rung) two statistics, each with a series-bootstrap CI (series are the resampling
   unit, invariant 2). Each is a co-directed effect in null units, `sign(T) sqrt(|T|)` with `T`
   the mean product of the two sides' effects, each in units of its own null scale (the pooled
   p95 of the null's absolute effect): (i) level, `T` averaged over series on the signed level
   effect; (ii) shape, `T` averaged over the level-removed shape channels that clear in either
   side, each side's channel effect being its mean over `U`. The matched floor of a statistic is
   its value when ONE side's real effect is replaced by a draw of the matched null, against the
   other side's real effect (both directions), so it carries exactly what an equally-sized,
   equally-profiled removal agrees by chance. Correlation and cosine were tried first and
   dropped: a profile-matched null's across-series pattern is the set's profile up to a random
   sign, so its correlation with any real effect is about +-1 and no real effect can clear it
   (seed 0: floor q95 1.0 against an observed 0.999).
4. **Calls.** A statistic is `agree` when its CI lower bound is above the floor's q95,
   `differ` when its CI upper bound is below the floor's q05, else `inconclusive`. It is
   `not scorable` (never `differ`) when a side does not clear its own null on the channel it
   measures (MN-30's diagnosis: concordance of two sub-null residuals is noise, and a deterministic
   sign pattern in it read as disagreement). The overall call is `differ` when any scorable
   statistic differs, `agree` when every scorable statistic agrees, `not scorable` when none is
   scorable, else `inconclusive`. Any `differ` decides because the shape channels are mostly
   unsigned magnitudes: an effect with the opposite sign still moves the same channels by the
   same amounts, so shape agreement cannot undo a signed contradiction (`mixed` records that it
   happened). `equivalence` (TOST on the per-series difference of the two null-unit level
   effects, margin `delta`, 90% CI) is reported beside and does not gate `agree`.

Evidence class: causal WITHIN each model (an ablation against that model's own null), compared
across models only on the same corpus inputs, never a transplant (invariant 5). The test is
opt-in, wired into no stage, and written to its own artifact `sae/subspace_agreement.json`. A
registered claim built on it must first pass `analysis/l5_subspace_gate.py` (sec 40 step 3).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..extraction.hooks import token_patch
from ..utils import log, save_json
from .ground_truth import load_ground_truth_table
from .response import _score_channel_against_null, battery_statistics
from .shared_input_agreement import (SHAPE_CHANNELS, TargetContext, _baseline_for_rows,
                                     _channel_deltas, _side_channel_scores,
                                     _target_key, battery_for_set, matched_dst_set, series_sets,
                                     verify_auc_reproduces)
from .transfer import _seed

__all__ = ["subspace_agreement_path", "run_subspace_agreement", "agreement_estimate",
           "decoder_subspace", "SCHEMA_VERSION", "STATUSES"]

SCHEMA_VERSION = 1
STATUSES = ("agree", "differ", "inconclusive", "not scorable")
TOST_ALPHA = 0.05
MIN_VALID_BOOT_FRACTION = 0.8

_null_cache: dict = {}

CLEARING_MODES = ("draw_level", "row_pooled")


def _score_draw_level(delta: np.ndarray, null_draws: list) -> dict:
    """One channel's real effect scored against the null at the level of DRAWS.

    `_score_channel_against_null` compares the MEAN over the shared series of |delta| with the
    p95 of the POOLED single-row |null| values, a mismatch the repo already documents
    (`response._lodo_pseudo_clear_rate`: the clear rate under exchangeability is far below
    nominal), so a real effect that is plainly above the null's typical mean is not called real.
    Here the real statistic (mean |delta| over the series) is compared with the same statistic of
    every null draw: it clears when it exceeds the q95 of the draws' means, a permutation-style
    test on exchangeable draws. `null_p95` stays the pooled single-row p95, the null UNIT the
    estimate divides each series' effect by.
    """
    base = _score_channel_against_null(delta, null_draws)
    draws = np.array([np.nanmean(d) for d in null_draws], dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    effect = base["effect"]
    q95 = float(np.quantile(draws, 0.95)) if draws.size else None
    degenerate = q95 is not None and not (q95 > 0.0)
    base.update({
        "clears_null": bool(q95 is not None and not degenerate and effect > q95),
        "null_draw_q95": q95,
        "null_draw_rank_p": (float((np.sum(draws >= effect) + 1) / (draws.size + 1))
                             if draws.size else None),
        "clearing": "draw_level"})
    return base


def _side_scores(real_raw: dict, real_shape: dict, null: list, clearing: str) -> dict:
    """`{"level": score, "shape": {channel: score}}` for one side under `clearing`
    (`CLEARING_MODES`); `row_pooled` is the rung's own `_side_channel_scores`."""
    if clearing not in CLEARING_MODES:
        raise ValueError(f"unknown clearing mode {clearing!r}: expected one of {CLEARING_MODES}")
    if clearing == "row_pooled":
        return _side_channel_scores(real_raw, real_shape, null)
    level_real = _channel_deltas(real_raw, "level")
    level_null = [np.abs(_channel_deltas(sr, "level")) for sr, _ in null
                  if _channel_deltas(sr, "level") is not None]
    level = (_score_draw_level(level_real, level_null) if level_real is not None else
             {"available": False, "effect": None, "signed_effect": None, "null_p95": None,
              "clears_null": False, "reason": "level channel unavailable"})
    shape = {}
    for ch in SHAPE_CHANNELS:
        real_delta = _channel_deltas(real_shape, ch)
        if real_delta is None:
            shape[ch] = {"available": False, "effect": None, "signed_effect": None,
                         "null_p95": None, "clears_null": False,
                         "reason": "channel unavailable on this row set"}
            continue
        draws = [np.abs(_channel_deltas(ss, ch)) for _, ss in null
                 if _channel_deltas(ss, ch) is not None]
        shape[ch] = _score_draw_level(real_delta, draws)
    return {"level": level, "shape": shape}


def _clearing(side: dict) -> list:
    out = ["level"] if side["level"].get("clears_null") else []
    return out + [c for c in SHAPE_CHANNELS if side["shape"].get(c, {}).get("clears_null")]



def subspace_agreement_path(run_dir) -> Path:
    return Path(run_dir) / "sae" / "subspace_agreement.json"


def reset_caches() -> None:
    _null_cache.clear()


# ---------------------------------------------------------------------------
# 1. Subspace removal and its matched null.
# ---------------------------------------------------------------------------

def decoder_subspace(W_dec: torch.Tensor, features) -> torch.Tensor:
    """`[d, r]` orthonormal basis of the span of `features`' decoder rows (`r` its numerical
    rank: singular values below `1e-6` of the largest are dropped, so duplicated or
    collinear members do not inflate the dimension)."""
    idx = sorted(int(f) for f in features)
    M = W_dec[idx].to(torch.float64)
    _u, s, vh = torch.linalg.svd(M, full_matrices=False)
    r = int((s > s[0] * 1e-6).sum()) if s.numel() and s[0] > 0 else 0
    return vh[:r].T.to(W_dec.dtype)


@torch.no_grad()
def _subspace_null_replacement(clean_tokens: torch.Tensor, sae, device, f_idx,
                               Q_null: torch.Tensor) -> torch.Tensor:
    """Token-level reconstruction with a RANDOM subspace removed in place of the set's own.

    The set's removal at token `t` is `r_t = sum_f z_f(t) w_f = Q c_t` (`Q` the orthonormal
    basis of its decoder span, `c_t = Q^T r_t`). The null removes `Q_null c_t`: the same
    coordinates through a random orthonormal frame of the same dimension, so the removed amount
    at every token and the angles within it are the set's exactly and only the subspace is
    random. The baseline stays the full reconstruction (`_feature_ablated_replacement`'s).
    """
    b, t, d = clean_tokens.shape
    flat = clean_tokens.reshape(-1, d).to(device)
    features = sae.encode(flat)
    recon = sae.decode(features)
    W = sae.W_dec.detach()
    idx = sorted(int(i) for i in f_idx)
    Q = decoder_subspace(W, idx).to(device=W.device, dtype=W.dtype)
    removal = features[:, idx] @ W[idx]
    coords = removal @ Q
    null_removal = coords @ Q_null.to(device=W.device, dtype=W.dtype).T
    return (recon - null_removal).reshape(b, t, d).cpu()


def _random_frame(sae, dim: int, rng: np.random.Generator) -> torch.Tensor:
    """`[d, dim]` random orthonormal frame spanned by `dim` random combinations of decoder rows
    (on the dictionary's own span, as `profile_matched` draws its direction)."""
    W = sae.W_dec.detach()
    code = torch.as_tensor(rng.normal(size=(dim, W.shape[0])), dtype=W.dtype, device=W.device)
    rows = (code @ W).T
    q, _r = torch.linalg.qr(rows.to(torch.float64))
    return q.to(W.dtype)


def subspace_null(ctx: TargetContext, features, U_key: tuple, contexts_u: np.ndarray,
                  targets_u: np.ndarray, periods_u: np.ndarray, baseline_seed: int,
                  direction_seed: int, n_null: int) -> list:
    """`n_null` subspace-matched null passes for `features` on `U`: a list of
    `(stats_raw, stats_shape)` tuples, the shape `battery_for_set` returns. The forward pass
    is reseeded with `baseline_seed`, the seed of the side's cached baseline (a sampled model's
    null must carry the baseline's own sampling noise, ROADMAP sec 37.8), and `direction_seed`
    seeds only the frame draws. Cached by `(model, layer, sorted(features), U, n_null)`."""
    feat_list = sorted(int(f) for f in features)
    key = (ctx.model, ctx.layer, tuple(feat_list), U_key, int(n_null), int(direction_seed))
    if key in _null_cache:
        return _null_cache[key]
    clean_tokens, baseline_fc, baseline_q = _baseline_for_rows(ctx, contexts_u, U_key, baseline_seed)
    dim = int(decoder_subspace(ctx.sae.W_dec.detach(), feat_list).shape[1])
    rng = np.random.default_rng(direction_seed)
    out = []
    for _ in range(int(n_null)):
        replacement = _subspace_null_replacement(clean_tokens, ctx.sae, ctx.device, feat_list,
                                                 _random_frame(ctx.sae, dim, rng))
        with token_patch(ctx.adapter.module, ctx.layer, ctx.adapter.token_slice, replacement):
            torch.manual_seed(baseline_seed)
            rec = ctx.adapter.predict(contexts_u, ctx.cfg.data.horizon, ctx.cfg.l0.quantiles)
        kw = dict(steered_quantiles=rec.get("quantiles"), baseline_quantiles=baseline_q)
        out.append((battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u, periods_u,
                                       remove_level=False, **kw),
                    battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u, periods_u,
                                       remove_level=True, **kw)))
    _null_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# 2. The estimate (pure numpy: testable without a model).
# ---------------------------------------------------------------------------

def _codirected(product_mean: np.ndarray) -> np.ndarray:
    """`sign(T) sqrt(|T|)` of a mean product `T` of two null-unit effects: the geometric-mean
    effect size, signed by whether the two sides move the same way. A monotone transform, so
    every comparison made on it (CI against floor) is the comparison on `T`."""
    return np.sign(product_mean) * np.sqrt(np.abs(product_mean))


def _quantile(values, q: float):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return float(np.quantile(v, q)) if v.size else None


def _call(lo, hi, observed, q95, q05, n_valid: int, n_boot: int) -> str:
    """`agree` / `differ` / `inconclusive` from a CI and the matched floor's tails."""
    if observed is None or lo is None or n_valid < MIN_VALID_BOOT_FRACTION * n_boot:
        return "inconclusive"
    if q95 is not None and lo > q95:
        return "agree"
    if q05 is not None and hi < q05:
        return "differ"
    return "inconclusive"


def _ci(samples: np.ndarray, level: float) -> tuple:
    v = samples[np.isfinite(samples)]
    if not v.size:
        return None, None
    tail = (1.0 - level) / 2.0
    return float(np.quantile(v, tail)), float(np.quantile(v, 1.0 - tail))


def _statistic(x_real: np.ndarray, y_real: np.ndarray, x_null: np.ndarray, y_null: np.ndarray,
               idx: np.ndarray, ci_level: float, n_boot: int) -> dict:
    """The co-directed effect of two null-unit effect arrays `x`, `y` over their last axis
    (series for the level statistic, mask channels for the shape statistic), its
    series-bootstrap CI (`idx` `[n_boot, n]` the resampled positions along that axis) and its
    matched floor: the same statistic with `x` replaced by each draw of `x_null` `[K, n]`
    (against the real `y`) and with `y` replaced by each draw of `y_null` (against the real
    `x`). The agree threshold is the larger floor q95 and the differ threshold the smaller q05,
    so a call must clear both sides' floors, as the rung's does."""
    obs = float(_codirected(np.mean(x_real * y_real)))
    boot = _codirected((x_real[idx] * y_real[idx]).mean(axis=1))
    lo, hi = _ci(boot, ci_level)
    f_src = _codirected((x_null * y_real[None, :]).mean(axis=1))
    f_dst = _codirected((x_real[None, :] * y_null).mean(axis=1))
    q95 = [_quantile(f_src, 0.95), _quantile(f_dst, 0.95)]
    q05 = [_quantile(f_src, 0.05), _quantile(f_dst, 0.05)]
    t95 = None if any(v is None for v in q95) else max(q95)
    t05 = None if any(v is None for v in q05) else min(q05)
    n_valid = int(np.isfinite(boot).sum())
    obs = obs if np.isfinite(obs) else None
    return {"observed": obs, "ci": [lo, hi], "n_valid_boot": n_valid,
            "floor_q95_src": q95[0], "floor_q95_dst": q95[1], "floor_q05_src": q05[0],
            "floor_q05_dst": q05[1], "agree_threshold": t95, "differ_threshold": t05,
            "n_floor_src": int(np.isfinite(f_src).sum()), "n_floor_dst": int(np.isfinite(f_dst).sum()),
            "call": _call(lo, hi, obs, t95, t05, n_valid, int(n_boot))}


def agreement_estimate(*, level_a, level_b, null_level_a, null_level_b, level_p95_a, level_p95_b,
                       level_clears_a: bool, level_clears_b: bool, shape_a: dict, shape_b: dict,
                       null_shape_a: list, null_shape_b: list, shape_p95_a: dict,
                       shape_p95_b: dict, shape_mask: list, shape_clearing_a: list,
                       shape_clearing_b: list, n_boot: int = 1000, ci_level: float = 0.95,
                       margin: float = 1.0, seed: int = 0) -> dict:
    """The two statistics, their CIs, floors, calls, the overall call and the TOST equivalence,
    from per-series arrays on the shared series `U` (all length `n`).

    `level_*` are the real signed level effects of each side, `null_level_*` `[n_null, n]` the
    matched-null draws' (raw level), `level_p95_*` each side's null scale (the p95 of its pooled
    absolute null level effect: one null unit). `shape_*` map a shape channel to its `[n]` real
    delta, `null_shape_*` are lists (per draw) of such maps, `shape_p95_*` each side's
    per-channel null p95, `shape_mask` the channels to compare, `shape_clearing_*` the channels
    each side clears its own null on. A statistic is `not scorable` unless every side it reads
    clears its own null on it: level needs `level` cleared on both sides, shape needs each side
    to clear at least one channel of the mask.

    Level statistic: the co-directed effect (`_codirected`) of the two sides' per-series level
    effects in null units. Shape statistic: the same over the mask channels, each side's channel
    effect being its mean over the series in null units. (Pearson correlation and cosine were
    tried first and are not used: a null draw with the set's own removal profile has an
    across-series effect pattern fixed by that profile up to a random sign, so its correlation
    with a real effect is about +-1 and no real effect can clear that floor; an effect in null
    units carries the scale that correlation throws away.)
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(level_a, dtype=np.float64)
    b = np.asarray(level_b, dtype=np.float64)
    n = a.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    out: dict = {"n_series": int(n), "n_boot": int(n_boot), "ci_level": float(ci_level),
                 "margin": float(margin)}

    level: dict = {"scorable": bool(level_clears_a and level_clears_b)}
    if level["scorable"]:
        da, db = a / level_p95_a, b / level_p95_b
        level.update(_statistic(da, db, np.asarray(null_level_a, dtype=np.float64) / level_p95_a,
                                np.asarray(null_level_b, dtype=np.float64) / level_p95_b,
                                idx, ci_level, int(n_boot)))
        diff = (da - db)[idx].mean(axis=1)
        tlo, thi = _ci(diff, 1.0 - 2.0 * TOST_ALPHA)
        level["equivalence"] = {
            "test": "TOST on the mean per-series difference of null-unit level effects",
            "mean_difference": float((da - db).mean()), "ci": [tlo, thi], "alpha": TOST_ALPHA,
            "margin": float(margin),
            "equivalent": bool(tlo is not None and tlo > -margin and thi < margin)}
        level["effect_null_units"] = {"src": float(np.mean(np.abs(da))),
                                      "dst": float(np.mean(np.abs(db)))}
    else:
        level["call"] = "not scorable"
        level["reason"] = "level does not clear its own subspace-matched null on " + \
            ("both sides" if not (level_clears_a or level_clears_b) else
             "the source side" if not level_clears_a else "the destination side")
    out["level_effect"] = level

    mask = list(shape_mask)
    scorable_shape = bool(mask) and bool(set(shape_clearing_a) & set(mask)) and \
        bool(set(shape_clearing_b) & set(mask))
    shape: dict = {"scorable": scorable_shape, "mask": mask}
    if scorable_shape:
        def vec(deltas: dict, p95: dict, rows=None):
            """`[len(mask)]` mean effect per mask channel in null units (`[B, len(mask)]` for
            the resampled rows `rows`)."""
            cols = []
            for c in mask:
                d = np.asarray(deltas[c], dtype=np.float64)
                p = p95.get(c)
                if p in (None, 0.0):
                    cols.append(np.zeros(1 if rows is None else rows.shape[0]))
                else:
                    cols.append(np.atleast_1d(np.nanmean(d) if rows is None
                                              else np.nanmean(d[rows], axis=1)) / p)
            return np.stack(cols, axis=-1)

        va, vb = vec(shape_a, shape_p95_a)[0], vec(shape_b, shape_p95_b)[0]
        boot_a, boot_b = vec(shape_a, shape_p95_a, idx), vec(shape_b, shape_p95_b, idx)
        null_va = np.concatenate([vec(d, shape_p95_a) for d in null_shape_a])
        null_vb = np.concatenate([vec(d, shape_p95_b) for d in null_shape_b])
        obs = float(_codirected(np.mean(va * vb)))
        boot = _codirected((boot_a * boot_b).mean(axis=1))
        lo, hi = _ci(boot, ci_level)
        f_src = _codirected((null_va * vb[None, :]).mean(axis=1))
        f_dst = _codirected((va[None, :] * null_vb).mean(axis=1))
        q95 = [_quantile(f_src, 0.95), _quantile(f_dst, 0.95)]
        q05 = [_quantile(f_src, 0.05), _quantile(f_dst, 0.05)]
        t95 = None if any(v is None for v in q95) else max(q95)
        t05 = None if any(v is None for v in q05) else min(q05)
        n_valid = int(np.isfinite(boot).sum())
        obs = obs if np.isfinite(obs) else None
        shape.update({
            "observed": obs, "ci": [lo, hi], "n_valid_boot": n_valid,
            "floor_q95_src": q95[0], "floor_q95_dst": q95[1], "floor_q05_src": q05[0],
            "floor_q05_dst": q05[1], "agree_threshold": t95, "differ_threshold": t05,
            "call": _call(lo, hi, obs, t95, t05, n_valid, int(n_boot))})
    else:
        shape["call"] = "not scorable"
        shape["reason"] = ("no shape channel clears its own null on both sides"
                           if mask else "no shape channel is available and clears in either side")
    out["shape_effect"] = shape

    calls = [s["call"] for s in (level, shape) if s["call"] != "not scorable"]
    if not calls:
        overall = "not scorable"
    elif all(c == "agree" for c in calls):
        overall = "agree"
    elif "differ" in calls:
        overall = "differ"
        out["mixed"] = "agree" in calls
    else:
        overall = "inconclusive"
    out["verdict"] = overall
    return out


# ---------------------------------------------------------------------------
# 3. Driver.
# ---------------------------------------------------------------------------

def _shape_deltas(stats_shape: dict) -> dict:
    return {c: _channel_deltas(stats_shape, c) for c in SHAPE_CHANNELS
            if _channel_deltas(stats_shape, c) is not None}


def run_subspace_agreement(cfg, run_dir, hub, store, data, device, units: list, *,
                           private_pooled: dict | None = None, private_alive: dict | None = None,
                           ground_truth_path: str | None = None, n_null: int | None = None,
                           n_boot: int | None = None, ci_level: float | None = None,
                           margin: float | None = None, base_seed: int | None = None,
                           clearing: str = "draw_level", write: bool = True) -> dict:
    """Score every `unit` (the dict `shared_input_agreement.build_units` returns, with
    `src_features` / `dst_features` the two sides' member sets; a unit with
    `dst_set_request == "matched_set"` has its destination set resolved as the rung does) and
    return the artifact, written to `sae/subspace_agreement.json` unless `write=False`.

    How an orchestrator registers it: freeze the `units` (source set, destination set, targets,
    `k_top_series`) in the registry exactly as the rung's units are frozen, pass the private
    corpus's `private_pooled` / `private_alive` and `ground_truth_path`, set `write=False`, and
    read each test's `verdict` (`agree` / `differ` / `inconclusive` / `not scorable`) and its
    two statistics' `ci`, `agree_threshold` and `differ_threshold`. The defaults reproduce the
    dev run. The artifact's `tests[i]` keep the unit's identity fields.
    """
    t0 = time.monotonic()
    reset_caches()
    from . import shared_input_agreement as sia
    sia.reset_caches()
    c = cfg.concepts
    n_null = int(n_null if n_null is not None else getattr(c, "subspace_agreement_n_null", 50))
    n_boot = int(n_boot if n_boot is not None else getattr(c, "subspace_agreement_n_boot", 1000))
    ci_level = float(ci_level if ci_level is not None else getattr(c, "subspace_agreement_ci_level", 0.95))
    margin = float(margin if margin is not None else getattr(c, "subspace_agreement_margin", 1.0))
    base_seed = int(base_seed if base_seed is not None else cfg.run.seed)

    periods_full = None
    try:
        gt = load_ground_truth_table(ground_truth_path if ground_truth_path is not None
                                     else cfg.data.path)
        periods_full = gt.reindex(data.meta["series_id"].to_numpy())[
            "seasonal_period_dominant"].to_numpy(dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 -- degrade the seasonal channel, not the module
        log.warning("subspace_agreement: no ground-truth periods (%s); seasonal channel "
                    "unavailable for every test", exc)

    contexts: dict = {}

    def _ctx(target: str) -> TargetContext:
        if target not in contexts:
            model, layer = _target_key(target)
            contexts[target] = TargetContext(cfg, hub, store, data, run_dir, model, layer, device,
                                             pooled=(private_pooled or {}).get(target),
                                             alive_mask=(private_alive or {}).get(target))
        return contexts[target]

    tests, reach_records, verified = [], {}, set()
    for unit in units:
        ctx_src, ctx_dst = _ctx(unit["src_target"]), _ctx(unit["dst_target"])
        reach_records.setdefault(unit["src_target"], ctx_src.reach)
        reach_records.setdefault(unit["dst_target"], ctx_dst.reach)
        record = {"concept": unit["concept"], "src_target": unit["src_target"],
                  "src_model": unit["src_model"], "src_features": list(unit["src_features"]),
                  "dst_target": unit["dst_target"], "dst_model": unit["dst_model"],
                  "dst_feature": unit["dst_feature"], "dst_features": list(unit["dst_features"]),
                  "dst_set_kind": unit["dst_set_kind"]}
        for side, ctx in (("src", ctx_src), ("dst", ctx_dst)):
            if not ctx.reach["reachable"]:
                record["verdict"] = "not scorable"
                record["reason"] = (f"{unit[side + '_target']} does not causally reach the forecast "
                                    f"(reach_probe): {ctx.reach['reason']}")
                break
        if "verdict" in record:
            tests.append(record)
            continue

        S_src, S_dst, _ = series_sets(ctx_src, ctx_dst, unit)
        if unit.get("dst_set_request") == "matched_set":
            unit = {**unit, "dst_features": matched_dst_set(ctx_dst, S_src, len(unit["src_features"])),
                    "dst_set_kind": "matched_set"}
            record["dst_features"], record["dst_set_kind"] = list(unit["dst_features"]), "matched_set"
        pair = (unit["src_model"], unit["dst_model"])
        if (pair not in verified and unit.get("test_auc") is not None
                and unit.get("transfer_k_top_series", unit["k_top_series"]) == unit["k_top_series"]):
            verify_auc_reproduces(ctx_dst, S_src, unit["dst_feature"], unit["test_auc"])
            verified.add(pair)

        U = np.array(sorted(set(S_src.tolist()) | set(S_dst.tolist())), dtype=int)
        U_key = tuple(int(u) for u in U)
        contexts_u, targets_u = data.contexts()[U], data.targets()[U]
        periods_u = periods_full[U] if periods_full is not None else np.full(U.size, np.nan)

        seed_src = _seed("shared_input_baseline", unit["src_target"], U_key, base=base_seed)
        seed_dst = _seed("shared_input_baseline", unit["dst_target"], U_key, base=base_seed)
        raw_a, shape_a = battery_for_set(ctx_src, unit["src_features"], U_key, contexts_u,
                                         targets_u, periods_u, seed_src)
        raw_b, shape_b = battery_for_set(ctx_dst, unit["dst_features"], U_key, contexts_u,
                                         targets_u, periods_u, seed_dst)
        dseed_a = _seed("subspace_null", unit["src_target"], tuple(sorted(unit["src_features"])),
                        U_key, base=base_seed)
        dseed_b = _seed("subspace_null", unit["dst_target"], tuple(sorted(unit["dst_features"])),
                        U_key, base=base_seed)
        null_a = subspace_null(ctx_src, unit["src_features"], U_key, contexts_u, targets_u, periods_u,
                               seed_src, dseed_a, n_null)
        null_b = subspace_null(ctx_dst, unit["dst_features"], U_key, contexts_u, targets_u, periods_u,
                               seed_dst, dseed_b, n_null)
        side_a = _side_scores(raw_a, shape_a, null_a, clearing)
        side_b = _side_scores(raw_b, shape_b, null_b, clearing)
        clear_a, clear_b = _clearing(side_a), _clearing(side_b)
        mask = [ch for ch in SHAPE_CHANNELS
                if side_a["shape"][ch]["available"] and side_b["shape"][ch]["available"]
                and (side_a["shape"][ch]["clears_null"] or side_b["shape"][ch]["clears_null"])]
        record.update({
            "U": list(U_key), "n_shared_series": int(U.size),
            "dimension": {"src": int(decoder_subspace(ctx_src.sae.W_dec.detach(),
                                                      unit["src_features"]).shape[1]),
                          "dst": int(decoder_subspace(ctx_dst.sae.W_dec.detach(),
                                                      unit["dst_features"]).shape[1])},
            "null": {"kind": "subspace_matched", "n": n_null},
            "side_src": {"clearing_channels": clear_a, "level": side_a["level"], "shape": side_a["shape"]},
            "side_dst": {"clearing_channels": clear_b, "level": side_b["level"], "shape": side_b["shape"]},
        })

        level_a, level_b = _channel_deltas(raw_a, "level"), _channel_deltas(raw_b, "level")
        if level_a is None or level_b is None:
            record["verdict"] = "not scorable"
            record["reason"] = "level channel unavailable on the shared series"
            tests.append(record)
            continue
        est = agreement_estimate(
            level_a=level_a, level_b=level_b,
            null_level_a=np.stack([_channel_deltas(sr, "level") for sr, _ in null_a]),
            null_level_b=np.stack([_channel_deltas(sr, "level") for sr, _ in null_b]),
            level_p95_a=side_a["level"]["null_p95"], level_p95_b=side_b["level"]["null_p95"],
            level_clears_a="level" in clear_a, level_clears_b="level" in clear_b,
            shape_a=_shape_deltas(shape_a), shape_b=_shape_deltas(shape_b),
            null_shape_a=[_shape_deltas(ss) for _, ss in null_a],
            null_shape_b=[_shape_deltas(ss) for _, ss in null_b],
            shape_p95_a={ch: side_a["shape"][ch]["null_p95"] for ch in SHAPE_CHANNELS},
            shape_p95_b={ch: side_b["shape"][ch]["null_p95"] for ch in SHAPE_CHANNELS},
            shape_mask=mask, shape_clearing_a=clear_a, shape_clearing_b=clear_b,
            n_boot=n_boot, ci_level=ci_level, margin=margin,
            seed=_seed("subspace_boot", unit["src_target"], unit["dst_target"], U_key, base=base_seed))
        record.update(est)
        tests.append(record)

    counts: dict = {}
    for r in tests:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    out = {"schema_version": SCHEMA_VERSION,
           "params": {"n_null": n_null, "n_boot": n_boot, "ci_level": ci_level, "margin": margin,
                      "tost_alpha": TOST_ALPHA, "clearing": clearing, "shape_channels": list(SHAPE_CHANNELS),
                      "statuses": list(STATUSES), "null": "subspace_matched",
                      "evidence_class": "causal within model; compared across models on shared inputs"},
           "n_tests": len(tests), "tests": tests, "verdict_counts": counts, "reach": reach_records,
           "runtime_seconds": time.monotonic() - t0}
    if not write:
        log.info("subspace agreement: %d test(s), verdicts=%s, %.1fs (not written)", len(tests),
                 counts, out["runtime_seconds"])
        return out
    path = subspace_agreement_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, out)
    log.info("subspace agreement: wrote %s (%d test(s), verdicts=%s, %.1fs)", path, len(tests),
             counts, out["runtime_seconds"])
    return out
