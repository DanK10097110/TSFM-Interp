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
from ..utils import batch_slices, log

_MIN_VALID = 10


def _scalar_ground_truth(gt) -> dict:
    """Flatten one sample's `GroundTruth` into scalar fields comparable across series."""
    p = gt.generative_params or {}
    trend = p.get("trend") or {}
    seasonalities = p.get("seasonalities") or []
    ar = p.get("ar_coeffs") or []
    dominant = max(seasonalities, key=lambda s: s.get("amplitude", 0.0)) if seasonalities else None
    return {
        "trend_order": trend.get("order"),
        "trend_scale": trend.get("scale"),
        "n_seasonalities": len(seasonalities),
        "seasonal_amplitude_max": dominant.get("amplitude") if dominant else None,
        "seasonal_period_dominant": dominant.get("period") if dominant else None,
        "ar_order": len(ar),
        "ar_coeff_sum": float(sum(ar)) if ar else None,
        "noise_scale": p.get("noise_scale"),
        "n_changepoints": len(gt.changepoints),
        "n_anomalies": len(gt.anomalies),
        "has_random_walk": float("random_walk_scale" in p),
        "has_intermittency": float("intermittency" in p),
        "has_heteroskedastic": float("heteroskedastic" in p),
        "archetype": p.get("archetype"),
    }


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
        rows.append(row)
    df = pd.DataFrame(rows).set_index("sample_id")
    archetypes = df["archetype"].dropna().unique()
    for a in archetypes:
        df[f"archetype_{a}"] = (df["archetype"] == a).astype(float)
    return df.drop(columns=["archetype"])


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


def ground_truth_alignment(cfg: PipelineConfig, store: ActivationStore, model: str,
                           layer: str, sae, device) -> dict:
    """I/O wrapper: load the corpus's ground truth + this model/layer's encoded features."""
    meta = load_meta(cfg.run_dir())
    gt = load_ground_truth_table(cfg.data.path)
    gt_cols = [c for c in gt.columns if c != "generator"]

    n = min(len(meta), cfg.sae.ground_truth_max_series)
    series_ids = meta["series_id"].to_numpy()[:n]
    features = encode_series_level(sae, store, model, layer, np.arange(n), device)
    return best_ground_truth_matches(features, gt, series_ids, gt_cols)
