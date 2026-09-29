"""Ground-truth feature-alignment score (ROADMAP.md §2.1, §6.2, §6.3).

This repo's distinctive advantage over most SAE interpretability work: every
synthetic benchmark series carries exact generative ground truth (trend
order, seasonality amplitude/period, AR order, changepoints, anomalies,
...), so "is this feature interpretable" can be checked against a real
target instead of falling back to qualitative max-activating-example
inspection first. For each learned feature, this correlates its per-series
activation against every available ground-truth component and reports the
best match and its strength.

Deliberately series-level, not window-level: `GroundTruth.generative_params`
stores scalar recipe parameters (one trend order, one AR coefficient list,
...) for the whole series, not a per-timestep decomposition aligned to this
pipeline's pooling windows. A finer window-level version -- matching a
feature's own per-window activation trace against a pooled version of
`GroundTruth.components`' per-timestep arrays -- is a natural extension not
built this session (`CLAUDE.md` §2.5: this coarser version is a complete,
useful answer on its own; scope was not stretched to also build the finer
one on the same pass).

Ground truth is real only for the `parametric`/`random_parametric`
generators; real-derived series (`mixture`, `block_bootstrap`,
`sequential_par`) carry deliberately empty `GroundTruth` objects
(`CLAUDE.md` §4.1), so they contribute no rows here rather than
misleadingly-empty ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from tsfm_benchmark.build_pipeline.seal import load_sealed

from ..config import PipelineConfig
from ..extraction.store import ActivationStore, load_meta
from ..utils import batch_slices, log, sample_rows

_MIN_VALID = 10
_PROVENANCE_PREFIXES = ("tier_", "generator_", "archetype_")


def _scalar_ground_truth(gt) -> dict:
    """Flatten one sample's `GroundTruth` into scalar fields comparable across series.

    `has_gt` distinguishes "no ground truth at all" (real-derived tier) from
    "genuinely zero of this structural property" (a fully-specified
    synthetic recipe with e.g. 0 changepoints) -- both previously collapsed
    to the same `0`/`False` via `len([])`/`"x" in {}`, so a real-derived
    series silently counted as a valid negative example for
    `n_seasonalities`/`ar_order`/`n_changepoints`/`n_anomalies`/`has_*` too,
    not just the `archetype` dummy this item was named for (`ROADMAP.md` sec
    15 A10 -- found via the item's own real-checkpoint verification: several
    features that appeared to correlate with `n_seasonalities` at rho~-0.55
    turned out, once a real `tier` field existed to compete against, to
    actually be tier detectors at rho~0.94, exactly this same defect one
    level removed from the archetype dummy it was named for).

    Gated on `"trend" in p`, not `bool(p)`: `CLAUDE.md` §4.1 says real-derived
    `GroundTruth` is "deliberately empty," but `mixture` actually sets
    `generative_params={"mode": ...}` (`generators.py:450`) -- non-empty, but
    with none of the structural keys `parametric`/`random_parametric` always
    set together via `_generative_params()` (`generators.py:152`, `"trend"`
    included unconditionally). `bool(p)` alone would have missed exactly the
    generator this item is most concerned with; checked directly against a
    real `mixture` sample rather than assumed, per `CLAUDE.md` §2.4.
    """
    p = gt.generative_params or {}
    has_gt = "trend" in p
    trend = p.get("trend") or {}
    seasonalities = p.get("seasonalities") or []
    ar = p.get("ar_coeffs") or []
    dominant = max(seasonalities, key=lambda s: s.get("amplitude", 0.0)) if seasonalities else None
    return {
        "trend_order": trend.get("order"),
        "trend_scale": trend.get("scale"),
        "n_seasonalities": float(len(seasonalities)) if has_gt else None,
        "seasonal_amplitude_max": dominant.get("amplitude") if dominant else None,
        "seasonal_period_dominant": dominant.get("period") if dominant else None,
        "ar_order": float(len(ar)) if has_gt else None,
        "ar_coeff_sum": float(sum(ar)) if ar else None,
        "noise_scale": p.get("noise_scale"),
        "n_changepoints": float(len(gt.changepoints)) if has_gt else None,
        "n_anomalies": float(len(gt.anomalies)) if has_gt else None,
        "has_random_walk": float("random_walk_scale" in p) if has_gt else None,
        "has_intermittency": float("intermittency" in p) if has_gt else None,
        "has_heteroskedastic": float("heteroskedastic" in p) if has_gt else None,
        "archetype": p.get("archetype"),
    }


def _add_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Archetype/tier/generator one-hot dummies (`ROADMAP.md` sec 15 A10).

    Pure transform on an already-assembled raw table (`archetype`, `tier`,
    `generator` string columns) -- kept separate from `load_ground_truth_table`'s
    `load_sealed` I/O so this logic, which is where the actual bug lived, can
    be unit tested directly with a synthetic DataFrame.

    Archetype dummies previously read `df["archetype"] == a` for *every* row,
    including real-derived series whose `GroundTruth` never had an archetype
    concept to begin with (`archetype` is `NaN` there, not "not archetype
    `a`") -- so every real-derived series silently became a valid negative
    example for every archetype, and a feature that merely fires more on
    `random_parametric` than on `mixture`/`block_bootstrap` (i.e. detects
    *tier*, not any specific archetype) could score a spuriously high ρ
    against an archetype dummy. Fixed: rows with no archetype get `NaN` on
    every `archetype_*` dummy (excluded from that field's own `valid` mask in
    `best_ground_truth_matches`, exactly like any other inapplicable field),
    not `0.0`. `tier`/`generator` get their own one-hot dummies too (always
    defined -- every sample has both), so "this feature detects the
    real-derived tier" or "detects `mixture` specifically" is now a
    detectable, nameable match instead of one that could only masquerade as
    an archetype match.
    """
    has_archetype = df["archetype"].notna()
    for a in sorted(df["archetype"].dropna().unique()):
        df[f"archetype_{a}"] = np.where(has_archetype, (df["archetype"] == a).astype(float), np.nan)
    for t in sorted(df["tier"].unique()):
        df[f"tier_{t}"] = (df["tier"] == t).astype(float)
    for g in sorted(df["generator"].unique()):
        df[f"generator_{g}"] = (df["generator"] == g).astype(float)
    return df.drop(columns=["archetype", "tier"])


def load_ground_truth_table(corpus_path: str) -> pd.DataFrame:
    """One row per sealed sample with scalar ground-truth fields, indexed by sample_id.

    Samples with empty `generative_params` (every real-derived generator)
    get all-`None` scalar fields and are dropped downstream by the
    per-column validity filter, not silently included as false zeros.
    """
    samples, _ = load_sealed(corpus_path, verify=False)
    rows = []
    for s in samples:
        row = _scalar_ground_truth(s.ground_truth)
        row["sample_id"] = s.sample_id
        row["generator"] = s.provenance.generator
        row["tier"] = s.provenance.generator_params.get("tier", "unknown")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("sample_id")
    return _add_dummy_columns(df)


@torch.no_grad()
def encode_series_level(sae, store: ActivationStore, model: str, layer: str,
                        rows: np.ndarray, device) -> np.ndarray:
    pooled = store.load(model, layer, level="series", rows=rows).astype(np.float32)
    features = sae.encode(torch.from_numpy(pooled).to(device))
    return features.cpu().numpy()


def best_ground_truth_matches(features: np.ndarray, gt: pd.DataFrame, series_ids: np.ndarray,
                              gt_cols: list, min_valid: int = _MIN_VALID,
                              top_features: int = 50,
                              must_include_fields: tuple = ()) -> dict:
    """Pure per-feature best-match search: no I/O, so this is the part unit tests exercise directly.

    `features` is `[N, F]` (one row per series, in the same order as
    `series_ids`); `gt` is indexed by sample_id and reindexed onto
    `series_ids` here so a series with no ground-truth row still gets an
    aligned (all-NaN) row rather than silently shifting every later row.

    `top_features` truncates the returned `features` *list* only -- the
    headline `n_features_matched`/`mean_abs_rho_matched` are always over the
    full population. That distinction matters to any consumer that reads the
    list as a candidate pool rather than as a display top-N (V0's cross-model
    matcher does exactly that, `sae/matching.py`), so `top_features <= 0`
    returns every feature. The default is the value every existing caller was
    hardcoded to, so their artifacts are unchanged.

    `must_include_fields` (ROADMAP.md sec 16 C1): each named field's own
    single best full-population match (searched over the untruncated
    `results`, before the `top_features` slice) is appended to the returned
    list if it isn't already inside the top-`top_features` slice. This is
    the fix for a real bug: `sae/train.py::run_sae` used to run this same
    widening search over the already-truncated top-50 list, which cannot
    find a field's best match when that match sits outside the top 50 by
    |rho| entirely -- exactly the case that left every steering feature's
    `direction_match` null for both models (C1's diagnosis). Doing the
    search here, against `results` rather than `ranked[:top_features]`, is
    the only way to see the full population. Default `()` is a no-op --
    every existing caller's artifact is unchanged.
    """
    joined = gt.reindex(series_ids)
    n_with_gt = int(joined[gt_cols].notna().any(axis=1).sum())
    if n_with_gt < min_valid:
        log.info(f"sae ground-truth alignment: only {n_with_gt} series with any ground "
                 f"truth (real-derived tiers carry none); skipping")
        return {"n_series_with_ground_truth": n_with_gt, "n_features": features.shape[1],
               "n_features_matched": 0, "mean_abs_rho_matched": 0.0,
               "abs_rho_matched": [], "features": []}

    results = []
    for f_idx in range(features.shape[1]):
        col = features[:, f_idx]
        best = None
        if np.std(col) == 0:
            results.append({"feature": f_idx, "best_field": None, "rho": 0.0, "n": 0})
            continue
        for field in gt_cols:
            gvals = joined[field].to_numpy(dtype=np.float64)
            valid = ~np.isnan(gvals)
            if valid.sum() < min_valid or np.std(gvals[valid]) == 0 or np.std(col[valid]) == 0:
                continue
            rho, _ = spearmanr(col[valid], gvals[valid])
            if not np.isfinite(rho):
                continue
            if best is None or abs(rho) > abs(best["rho"]):
                best = {"feature": f_idx, "best_field": field, "rho": float(rho),
                       "n": int(valid.sum())}
        results.append(best or {"feature": f_idx, "best_field": None, "rho": 0.0, "n": 0})

    matched = [r for r in results if r["best_field"] is not None]
    ranked = sorted(results, key=lambda r: -abs(r["rho"]))
    out_features = ranked if top_features <= 0 else list(ranked[:top_features])

    widened_with = []
    if top_features > 0 and must_include_fields:
        seen = {r["feature"] for r in out_features}
        for field in must_include_fields:
            field_matches = [r for r in matched if r["best_field"] == field]
            if not field_matches:
                continue
            best_for_field = max(field_matches, key=lambda r: abs(r["rho"]))
            if best_for_field["feature"] not in seen:
                out_features.append(best_for_field)
                seen.add(best_for_field["feature"])
                widened_with.append((field, best_for_field["feature"]))

    return {
        "n_series_with_ground_truth": n_with_gt,
        "n_features": len(results),
        "n_features_matched": len(matched),
        "mean_abs_rho_matched": float(np.mean([abs(r["rho"]) for r in matched])) if matched else 0.0,
        "abs_rho_matched": [float(abs(r["rho"])) for r in matched],
        "features": out_features,
        "widened_with": widened_with,
    }


def is_provenance_field(field: str) -> bool:
    """True for a corpus-provenance dummy (`tier_*`/`generator_*`/`archetype_*`).

    These are defined on every series (ROADMAP.md §25.1 (3)) and encode
    which generator produced a series / whether it is synthetic or
    real-derived -- a confound flag for an interpretability reader, not a
    structural generative property (trend/seasonality/AR/changepoint/
    anomaly/intermittency). The one place this repo needs the distinction
    by name rather than by re-deriving it from `_add_dummy_columns`'
    prefixes at each call site.
    """
    return field.startswith(_PROVENANCE_PREFIXES)


def _ridge_cross_fitted_residuals(x: np.ndarray, y: np.ndarray, n_folds: int,
                                  alpha: float, seed: int) -> np.ndarray:
    """Residuals of `y` on one-hot provenance columns `x`, each row predicted out of fold.

    Closed-form ridge on the normal equations, the same pattern
    `analysis/error_fingerprint.py::_cross_fitted_residuals` (dev branch)
    already uses for
    exactly the reason ROADMAP.md sec 11.36 names: a control that silently
    explains nothing is indistinguishable from one that works unless it is
    scored out of fold, and this residualization has precisely that shape
    (few provenance columns, many rows -- the opposite failure direction
    from sec 11.36's ~200-column basis, but the same principle: publish the
    OOF R2, do not assume the fit worked).
    """
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % n_folds
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    resid = np.empty(n, dtype=np.float64)
    eye = np.eye(xb.shape[1])
    eye[-1, -1] = 0.0
    for f in range(n_folds):
        te = folds == f
        tr = ~te
        if tr.sum() < 2 or not te.any():
            resid[te] = y[te]
            continue
        xt = xb[tr]
        beta = np.linalg.solve(xt.T @ xt + alpha * eye, xt.T @ y[tr])
        resid[te] = y[te] - xb[te] @ beta
    return resid


def residualize_against_provenance(gt: pd.DataFrame, structural_field: str,
                                   provenance_cols: list, valid: np.ndarray,
                                   n_folds: int = 5, alpha: float = 0.01,
                                   seed: int = 0) -> tuple:
    """Residual of one structural field after regressing out the provenance one-hots.

    A feature whose entire correlation with a structural field is really
    "this is real-derived data" (ROADMAP.md §25.1 (2)-(3)) scores near zero
    here even if its raw correlation with that field was large, because the
    provenance signal has already been removed from the field itself, not
    from the feature. Returns `(residual, oof_r2)` -- `oof_r2` is the
    provenance basis's own out-of-fold R2 predicting the structural field, so
    a caller can tell "this field has no provenance confound to remove" (R2
    near 0) apart from "the regression silently did nothing" (also R2 near
    0) only by checking whether removing it changes anything; recorded so
    neither is asserted silently (§11.36's rule).

    `valid` is a boolean mask over `gt`'s row order selecting rows where
    `structural_field` is non-NaN -- provenance columns are always defined,
    so the regression is fit only on rows the structural field can itself be
    evaluated on.

    `alpha` defaults far lower than `analysis/error_fingerprint.py`'s (dev
    branch) same-shaped `_cross_fitted_residuals` (1.0, tuned for a
    ~200-column basis): the provenance basis here is a handful of one-hot
    columns, and a
    fixed `alpha=1.0` ridge penalty measurably under-corrects it. Confirmed
    directly (not assumed) with a fully-confounded synthetic case (a binary
    `tier` dummy explaining a structural field almost exactly,
    `oof_r2 > 0.99`): at `alpha=1.0` the cross-fitted residual still
    correlates with `tier` at 0.61 -- the ridge shrinkage pulls each fold's
    fitted coefficient toward zero, systematically under-removing the
    confound on every held-out point -- while `alpha=0.01` drops that
    leakage to 0.02 with no cost to the independent (no-confound) case. This
    is the opposite failure direction from `CLAUDE.md` §11.36's own basis
    (many columns, needs real shrinkage to avoid overfitting a tiny sample);
    a shared default across both would have been wrong for one of them.

    `provenance_cols` includes `archetype_*` dummies, which are `NaN` (not
    0) for a series with no archetype concept at all (`_add_dummy_columns`'
    documented distinction between "not this archetype" and "not
    archetype-bearing"). That NaN is a real, meaningful "not applicable" --
    but `np.linalg.solve` propagates any NaN in `x` to every entry of the
    fitted coefficient, silently returning an all-NaN residual for the
    *entire* fold rather than raising (caught empirically: 130 of 555
    `trend_order`-valid rows carry at least one `archetype_*` NaN on
    `runs/full_report_run_large`'s real corpus, which a 1-provenance-column
    synthetic test has no way to exercise). Filled with 0 here, consistent
    with every other row's own `archetype_*` encoding of "not this
    archetype" -- a non-archetype-bearing row already reads as all-zero
    across every `archetype_*` column it CAN take a value on, so this only
    extends that same convention to columns it cannot.
    """
    y = gt.loc[valid, structural_field].to_numpy(dtype=np.float64)
    x = gt.loc[valid, list(provenance_cols)].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    if len(y) < n_folds or np.std(y) == 0:
        return y - y.mean() if len(y) else y, 0.0
    resid = _ridge_cross_fitted_residuals(x, y, n_folds, alpha, seed)
    oof_r2 = 1.0 - float(np.sum(resid ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))
    return resid, oof_r2


def _average_rank_columns(a: np.ndarray) -> np.ndarray:
    """Column-wise average rank (`scipy.stats.rankdata(..., method="average")`'s
    tie convention), vectorized across all columns at once.

    A first version used plain ordinal rank (`argsort` twice, no tie
    averaging) and was silently wrong on tied data -- checked directly
    against `scipy.stats.spearmanr` (`CLAUDE.md` §2.4) rather than assumed
    correct because it "looked like the standard rank trick": max abs
    difference 0.047 on integer-valued test data, versus <1e-12 once average
    ranking was implemented. Real SAE dictionary activations are continuous
    and rarely exactly tied, but ground-truth fields like `n_seasonalities`/
    `ar_order`/`n_changepoints` are small integers with many genuine ties, so
    this is not a hypothetical edge case for this module's actual inputs.

    Implementation: sort each column, use the sorted order's inverse to place
    ordinal ranks, then average the ordinal ranks within each run of equal
    values (found via `np.diff` on the sorted values) -- O(n log n) per
    column via one vectorized sort across all columns at once, not a
    Python-level loop over columns.
    """
    order = np.argsort(a, axis=0, kind="mergesort")
    sorted_a = np.take_along_axis(a, order, axis=0)
    n = a.shape[0]
    ordinal = np.arange(1, n + 1, dtype=np.float64)
    if a.ndim == 1:
        sorted_a = sorted_a[:, None]
        order = order[:, None]
    # For each column, find run boundaries where consecutive sorted values
    # differ, then average the ordinal ranks within each run.
    out = np.empty_like(sorted_a, dtype=np.float64)
    for j in range(sorted_a.shape[1]):
        col = sorted_a[:, j]
        is_new = np.empty(n, dtype=bool)
        is_new[0] = True
        is_new[1:] = col[1:] != col[:-1]
        group_id = np.cumsum(is_new) - 1
        group_sum = np.bincount(group_id, weights=ordinal)
        group_count = np.bincount(group_id)
        avg_rank = group_sum / group_count
        out[:, j] = avg_rank[group_id]
    ranks = np.empty_like(out)
    np.put_along_axis(ranks, order, out, axis=0)
    return ranks[:, 0] if a.ndim == 1 else ranks


def _vectorized_spearman_all_features(features: np.ndarray, valid: np.ndarray,
                                      target: np.ndarray) -> np.ndarray:
    """Spearman rho of every column of `features[valid]` against `target`, vectorized.

    `features` is the FULL `[n_rows, n_features]` array; `valid` selects the
    rows to use. `target` must already be pre-sliced to those same `valid`
    rows (matching every existing caller's convention -- `resid`/`gvals[valid]`
    are already row-selected before this is called).

    Average-rank Spearman (`_average_rank_columns`) computed once across all
    features simultaneously via a single Pearson-on-ranks matrix computation,
    replacing an O(n_features) Python loop of `scipy.stats.spearmanr` calls --
    each of which carries real per-call overhead that dominates at dictionary
    sizes up to 10240 (§25.9 Stage 1's own real-corpus offline check took
    several minutes per target before this change).

    Returns `nan` for a constant column (correlation undefined), matching
    `scipy.stats.spearmanr`'s own behavior there.
    """
    x = features[valid]
    y = target
    n = x.shape[0]
    if n < 2 or n != y.shape[0]:
        return np.full(x.shape[1], np.nan)
    rx = _average_rank_columns(x)
    ry = _average_rank_columns(y)
    rx_c = rx - rx.mean(axis=0, keepdims=True)
    ry_c = ry - ry.mean()
    num = (rx_c * ry_c[:, None]).sum(axis=0)
    den = np.sqrt((rx_c ** 2).sum(axis=0) * (ry_c ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    rho[den == 0] = np.nan
    return rho


def best_ground_truth_matches_separated(features: np.ndarray, gt: pd.DataFrame,
                                        series_ids: np.ndarray, gt_cols: list,
                                        min_valid: int = _MIN_VALID,
                                        top_features: int = 50,
                                        n_folds: int = 5, alpha: float = 0.01,
                                        seed: int = 0,
                                        min_residual_scale: float = 0.01) -> dict:
    """Per-feature best match, reported as separate `structural`/`provenance` columns.

    ROADMAP.md §25.5(a): stop running one argmax over ~30 mixed fields
    (§25.1 shows this structurally favors provenance dummies, which are
    defined on every series and carry the corpus's single largest variance
    axis, over structural fields that are `None` for every real-derived
    series). Instead:

    - `provenance`: this feature's best match among `tier_*`/`generator_*`/
      `archetype_*` fields, on the RAW (non-residualized) correlation --
      exactly `best_ground_truth_matches`' existing search, restricted to
      provenance columns.
    - `structural`: this feature's best match among every other field, on
      the RESIDUAL correlation after regressing that field on the
      provenance one-hots (`residualize_against_provenance`) -- so a feature
      whose apparent structural match is really a provenance detector in
      disguise scores near zero here, on purpose (§25.5(a)'s stated design).
    - `n`: each match's own valid sample size, carried through rather than
      dropped, since a rho on 374 series and one on 965 are not the same
      claim (§25.5(a) -- structural fields are `None` for every real-derived
      series by construction, so their `n` is always <= the provenance
      fields' `n`).
    - `top3_structural`/`top3_provenance`: each feature's top-3 field
      matches per competition, not just the argmax -- §25.1 (4) shows a
      single best-of name discards real differentiation between atoms that
      an argmax collapses onto one label.

    Pure function, zero forward passes (`ROADMAP.md` §25.9 Stage 1's own
    stated cost bound) -- everything here is a transform of already-encoded
    `features` and the corpus's own ground-truth table.
    """
    provenance_cols = [c for c in gt_cols if is_provenance_field(c)]
    structural_cols = [c for c in gt_cols if not is_provenance_field(c)]
    joined = gt.reindex(series_ids)
    n_with_gt = int(joined[gt_cols].notna().any(axis=1).sum())
    if n_with_gt < min_valid:
        log.info(f"sae ground-truth alignment (separated): only {n_with_gt} series with any "
                 f"ground truth; skipping")
        return {"n_series_with_ground_truth": n_with_gt, "n_features": features.shape[1],
               "n_features_matched": 0, "mean_abs_rho_structural": 0.0,
               "mean_abs_rho_provenance": 0.0, "residualization_oof_r2": {}, "features": []}

    # Cache each structural field's provenance-residual once (shared across
    # every feature) rather than refitting the same regression per feature.
    residual_cache: dict = {}
    oof_r2_by_field: dict = {}
    not_separable: list = []
    for field in structural_cols:
        gvals = joined[field].to_numpy(dtype=np.float64)
        valid = ~np.isnan(gvals)
        if valid.sum() < min_valid or np.std(gvals[valid]) == 0:
            continue
        resid, oof_r2 = residualize_against_provenance(
            joined, field, provenance_cols, valid, n_folds=n_folds, alpha=alpha, seed=seed)
        # ROADMAP.md sec 26 A3. A field the provenance one-hots predict
        # PERFECTLY has no residual left to correlate against -- and
        # Spearman, being rank-based, does not degrade gracefully there: it
        # ranks whatever floating-point rounding survives and returns a
        # large, confident-looking rho computed on numerical noise.
        #
        # This is not hypothetical. On this repo's own corpus the three
        # binary flags `has_intermittency` / `has_random_walk` /
        # `has_heteroskedastic` are determined by the archetype that
        # generated the series, so oof_r2 is exactly 1.0000, the residual's
        # scale collapses to ~3e-4 of the field's own, and those three
        # fields supplied 82 of the 88 (93%) top matches the report
        # rendered -- at rho +0.43..+0.63, where the RAW-field correlation
        # was only +0.16..+0.20. Residualizing inflated a weak real
        # association into a strong fake one by dividing out everything
        # real and ranking the remainder.
        #
        # The gate is on the residual's surviving SCALE rather than on
        # oof_r2, because scale is the quantity that actually decides
        # whether there is anything to correlate with; oof_r2 is a proxy for
        # it. Measured separation on that corpus is unambiguous -- every
        # usable field keeps >= 0.42 of its sd, every degenerate one <=
        # 0.0004, a factor of ~1200 with nothing in between -- so the
        # default sits well inside the gap rather than on a knife edge.
        scale = float(np.std(resid) / np.std(gvals[valid]))
        if scale < min_residual_scale:
            not_separable.append({"field": field, "oof_r2": float(oof_r2),
                                  "residual_scale": scale})
            log.info(f"sae ground-truth (separated): dropping `{field}` from the "
                     f"structural competition -- provenance predicts it at "
                     f"oof_r2={oof_r2:.4f}, leaving {scale:.2e} of its own scale, "
                     f"so any residual correlation would be rounding noise")
            continue
        residual_cache[field] = (valid, resid)
        oof_r2_by_field[field] = oof_r2

    n_features = features.shape[1]
    # One vectorized pass per field (not per feature-per-field): rho_by_field
    # is {field: (rho_array[n_features], n)}, computed once and sliced per
    # feature below -- the same total work as the old per-feature Python
    # loop, done in O(n_fields) numpy calls instead of O(n_features * n_fields)
    # scipy calls.
    struct_rho_by_field: dict = {}
    for field, (valid, resid) in residual_cache.items():
        rho = _vectorized_spearman_all_features(features, valid, resid)
        struct_rho_by_field[field] = (rho, int(valid.sum()))

    prov_rho_by_field: dict = {}
    for field in provenance_cols:
        gvals = joined[field].to_numpy(dtype=np.float64)
        valid = ~np.isnan(gvals)
        if valid.sum() < min_valid or np.std(gvals[valid]) == 0:
            continue
        rho = _vectorized_spearman_all_features(features, valid, gvals[valid])
        prov_rho_by_field[field] = (rho, int(valid.sum()))

    results = []
    for f_idx in range(n_features):
        if np.std(features[:, f_idx]) == 0:
            results.append({"feature": f_idx, "structural": None, "provenance": None,
                            "top3_structural": [], "top3_provenance": []})
            continue

        struct_candidates = []
        for field, (rho_arr, n) in struct_rho_by_field.items():
            rho = rho_arr[f_idx]
            if not np.isfinite(rho):
                continue
            struct_candidates.append({"field": field, "rho": float(rho), "n": n})
        struct_candidates.sort(key=lambda r: -abs(r["rho"]))

        prov_candidates = []
        for field, (rho_arr, n) in prov_rho_by_field.items():
            rho = rho_arr[f_idx]
            if not np.isfinite(rho):
                continue
            prov_candidates.append({"field": field, "rho": float(rho), "n": n})
        prov_candidates.sort(key=lambda r: -abs(r["rho"]))

        results.append({
            "feature": f_idx,
            "structural": struct_candidates[0] if struct_candidates else None,
            "provenance": prov_candidates[0] if prov_candidates else None,
            "top3_structural": struct_candidates[:3],
            "top3_provenance": prov_candidates[:3],
        })

    struct_matched = [r["structural"]["rho"] for r in results if r["structural"] is not None]
    prov_matched = [r["provenance"]["rho"] for r in results if r["provenance"] is not None]
    ranked = sorted(results, key=lambda r: -abs(r["structural"]["rho"]) if r["structural"] else 0.0)
    out_features = ranked if top_features <= 0 else list(ranked[:top_features])

    return {
        "n_series_with_ground_truth": n_with_gt,
        "n_features": len(results),
        "n_features_matched": sum(1 for r in results if r["structural"] or r["provenance"]),
        "n_features_structural_matched": len(struct_matched),
        "n_features_provenance_matched": len(prov_matched),
        "mean_abs_rho_structural": float(np.mean(np.abs(struct_matched))) if struct_matched else 0.0,
        "mean_abs_rho_provenance": float(np.mean(np.abs(prov_matched))) if prov_matched else 0.0,
        "residualization_oof_r2": {k: float(v) for k, v in oof_r2_by_field.items()},
        # Reported, never silently dropped: a field excluded here is one the
        # corpus CANNOT separate from provenance, which is a fact about the
        # benchmark's design worth surfacing rather than an empty slot.
        "fields_not_separable_from_provenance": not_separable,
        "features": out_features,
    }


def permutation_null_alignment(features: np.ndarray, gt: pd.DataFrame, series_ids: np.ndarray,
                               gt_cols: list, seed: int, n_perm: int = 3,
                               max_features: int = 2000,
                               min_valid: int = _MIN_VALID) -> dict:
    """Label-permutation null for `mean_abs_rho_matched` (ROADMAP.md sec 16 E9).

    `best_ground_truth_matches` picks, per feature, the *best* of ~len(gt_cols)
    candidate fields -- a real multiple-comparisons inflation even under pure
    noise, since the max of many weak correlations is not itself weak. This
    reruns the identical search with each feature's row permuted (breaking
    every feature-to-series correspondence while leaving each field's own and
    each feature's own marginal distribution untouched) `n_perm` times, so
    the real `mean_abs_rho_matched` can be read against how large that number
    gets by chance alone, not against zero.

    Subsamples to `max_features` features per permutation (logged, never
    silent, per `ROADMAP.md` sec 15's no-silent-cap precedent) since a
    dictionary can hold 10k+ features and the null does not need every one
    to give an informative summary; `n_perm=0` disables this entirely.
    """
    if n_perm <= 0:
        return {"n_perm": 0, "max_features": 0, "mean_abs_rho_null_mean": 0.0,
               "mean_abs_rho_null_p95": 0.0, "mean_abs_rho_null_values": []}
    rng = np.random.default_rng(seed)
    n_features = features.shape[1]
    use_n = min(max_features, n_features)
    if use_n < n_features:
        log.info("sae ground-truth permutation null: subsampling %d/%d features "
                 "per permutation (n_perm=%d)", use_n, n_features, n_perm)
    means = []
    for _ in range(n_perm):
        feat_idx = rng.choice(n_features, size=use_n, replace=False)
        perm_rows = rng.permutation(len(series_ids))
        shuffled = features[:, feat_idx][perm_rows]
        sub = best_ground_truth_matches(shuffled, gt, series_ids, gt_cols, min_valid=min_valid)
        means.append(sub["mean_abs_rho_matched"])
    means = np.asarray(means, dtype=np.float64)
    return {"n_perm": int(n_perm), "max_features": int(use_n),
            "mean_abs_rho_null_mean": float(means.mean()),
            "mean_abs_rho_null_p95": float(np.quantile(means, 0.95)),
            "mean_abs_rho_null_values": [float(m) for m in means]}


def ground_truth_alignment(cfg: PipelineConfig, store: ActivationStore, model: str,
                           layer: str, sae, device,
                           must_include_fields: tuple = ()) -> dict:
    """I/O wrapper: load the corpus's ground truth + this model/layer's encoded features.

    `must_include_fields` passes through to `best_ground_truth_matches` --
    see its docstring and ROADMAP.md sec 16 C1. Default `()` is a no-op, so
    every caller but `sae/train.py::run_sae`'s steering step is unaffected.
    """
    meta = load_meta(cfg.run_dir())
    gt = load_ground_truth_table(cfg.data.path)
    gt_cols = [c for c in gt.columns if c != "generator"]

    # A `[:n]` head slice here silently drew from whichever generator/tier a
    # corpus written grouped by task happens to put first (ROADMAP.md sec 15
    # A4, compounding A10's tier-as-negative-label issue) -- stratified by
    # family instead, so which archetypes are present doesn't depend on task
    # order.
    rows = sample_rows(len(meta), cfg.sae.ground_truth_max_series, cfg.run.seed + 12,
                       strata=meta["family"].to_numpy())
    series_ids = meta["series_id"].to_numpy()[rows]
    features = encode_series_level(sae, store, model, layer, rows, device)
    result = best_ground_truth_matches(features, gt, series_ids, gt_cols,
                                       must_include_fields=must_include_fields)
    # ROADMAP.md sec 26 A1: the legacy single-argmax result above is KEPT
    # byte-for-byte (every recorded number stays regenerable, sec 2.1), and
    # the structural/provenance-separated view is ADDED beside it under
    # `separated` -- the sec 11.39 rule, "add a canonical key, leave the
    # legacy keys untouched, read new-then-legacy". The report reads
    # `separated` as its headline: on `runs/full_report_run_large`, 445 of
    # 566 (78.6%) of the legacy matches -- and 11 of 11 headline rows --
    # were `generator_*`/`tier_*`/`archetype_*` provenance dummies, i.e. a
    # feature that detects which generator wrote the series, which is a
    # corpus artifact and not a model finding. `best_ground_truth_matches_
    # separated` has existed and been unit-tested since Stage 1 (sec 25.22)
    # but only ever ran as a standalone check; nothing consumed it.
    result["separated"] = best_ground_truth_matches_separated(
        features, gt, series_ids, gt_cols, seed=cfg.run.seed + 14)
    result["n_requested"] = int(cfg.sae.ground_truth_max_series)
    result["n_realized"] = int(len(rows))
    result["rows"] = [int(r) for r in rows]
    result["permutation_null"] = permutation_null_alignment(
        features, gt, series_ids, gt_cols, seed=cfg.run.seed + 13,
        n_perm=cfg.sae.ground_truth_permutation_repeats,
        max_features=cfg.sae.ground_truth_permutation_max_features)
    return result
