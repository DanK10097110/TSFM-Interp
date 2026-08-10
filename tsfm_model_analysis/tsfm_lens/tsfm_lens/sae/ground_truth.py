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
                              gt_cols: list, min_valid: int = _MIN_VALID) -> dict:
    """Pure per-feature best-match search: no I/O, so this is the part unit tests exercise directly.

    `features` is `[N, F]` (one row per series, in the same order as
    `series_ids`); `gt` is indexed by sample_id and reindexed onto
    `series_ids` here so a series with no ground-truth row still gets an
    aligned (all-NaN) row rather than silently shifting every later row.
    """
    joined = gt.reindex(series_ids)
    n_with_gt = int(joined[gt_cols].notna().any(axis=1).sum())
    if n_with_gt < min_valid:
        log.info(f"sae ground-truth alignment: only {n_with_gt} series with any ground "
                 f"truth (real-derived tiers carry none); skipping")
        return {"n_series_with_ground_truth": n_with_gt, "n_features": features.shape[1],
               "n_features_matched": 0, "mean_abs_rho_matched": 0.0, "features": []}

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
    return {
        "n_series_with_ground_truth": n_with_gt,
        "n_features": len(results),
        "n_features_matched": len(matched),
        "mean_abs_rho_matched": float(np.mean([abs(r["rho"]) for r in matched])) if matched else 0.0,
        "features": sorted(results, key=lambda r: -abs(r["rho"]))[:50],
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
                           layer: str, sae, device) -> dict:
    """I/O wrapper: load the corpus's ground truth + this model/layer's encoded features."""
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
    result = best_ground_truth_matches(features, gt, series_ids, gt_cols)
    result["n_requested"] = int(cfg.sae.ground_truth_max_series)
    result["n_realized"] = int(len(rows))
    result["rows"] = [int(r) for r in rows]
    result["permutation_null"] = permutation_null_alignment(
        features, gt, series_ids, gt_cols, seed=cfg.run.seed + 13,
        n_perm=cfg.sae.ground_truth_permutation_repeats,
        max_features=cfg.sae.ground_truth_permutation_max_features)
    return result
