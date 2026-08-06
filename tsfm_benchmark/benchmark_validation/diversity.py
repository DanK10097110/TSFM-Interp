"""Quantitative diversity metrics, computed in the catch22 feature space.

These are the numbers that make this a test rather than a picture. They are
deliberately computed on the standardised feature matrix, never on the UMAP
coordinates, because UMAP distorts global geometry and density. The metrics
capture three things: how spread out the benchmark is (effective dimensionality
and total variance), how isolated each sequence is from its nearest neighbour
(the redundancy tail), and which catch22 features actually carry the variation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .features import FeatureMatrix


@dataclass
class DiversityReport:
    n_sequences: int
    effective_dimensionality: float
    total_variance: float
    nn_distance_mean: float
    nn_distance_min: float
    nn_distance_p05: float
    near_collision_fraction: float
    feature_variance_ranking: list[tuple[str, float]]
    # Populated only by `diversity_metrics_by_group` (sec 15 A17) -- a
    # bootstrap CI resampling *this group's own rows*, so a small group's
    # point estimate carries a visible interval instead of the "trust the
    # direction, not the exact number" caveat living only in someone's
    # prose. `None` when not requested (`n_boot=0`) or not computed (the
    # single global `diversity_metrics()` call never sets these).
    effective_dimensionality_ci: tuple | None = None
    near_collision_fraction_ci: tuple | None = None


@dataclass
class InsufficientN:
    """Sentinel for a group below `min_group_size` (sec 15 A17).

    PCA/nearest-neighbour metrics are not meaningful on a handful of
    points; this states that explicitly in the output instead of silently
    omitting the group, so a reader can't mistake "not enough data" for
    "not present in the corpus."
    """
    n_sequences: int
    min_required: int
    status: str = "insufficient_n"


def _participation_ratio(values: np.ndarray) -> float:
    s = values.sum()
    if s <= 0:
        return 0.0
    return float((s ** 2) / (np.square(values).sum() + 1e-12))


def diversity_metrics(fm: FeatureMatrix, near_collision_quantile: float = 0.05) -> DiversityReport:
    """Summarise how much of the feature space the benchmark actually covers.

    Effective dimensionality is the participation ratio of the PCA spectrum: a
    value near the feature count means variation is spread across many
    independent properties, while a low value means the benchmark varies along
    only a few axes. The nearest-neighbour distance tail exposes redundancy that
    a mean distance would hide.
    """
    x = fm.features_scaled
    n, d = x.shape

    pca = PCA(n_components=min(n, d)).fit(x)
    eff_dim = _participation_ratio(pca.explained_variance_)
    total_var = float(pca.explained_variance_.sum())

    nn = NearestNeighbors(n_neighbors=2).fit(x)
    dists, _ = nn.kneighbors(x)
    nn_dist = dists[:, 1]
    threshold = np.quantile(nn_dist, near_collision_quantile)
    near_collision = float((nn_dist <= max(threshold, 1e-9)).mean())

    variances = x.var(axis=0)
    ranking = sorted(zip(fm.feature_names, variances.tolist()), key=lambda t: t[1], reverse=True)

    return DiversityReport(
        n_sequences=n,
        effective_dimensionality=round(eff_dim, 3),
        total_variance=round(total_var, 3),
        nn_distance_mean=round(float(nn_dist.mean()), 4),
        nn_distance_min=round(float(nn_dist.min()), 4),
        nn_distance_p05=round(float(np.quantile(nn_dist, 0.05)), 4),
        near_collision_fraction=round(near_collision, 4),
        feature_variance_ranking=[(name, round(v, 4)) for name, v in ranking],
    )


def _bootstrap_group_ci(sub: FeatureMatrix, n_boot: int, seed: int, ci: float,
                        near_collision_quantile: float) -> tuple:
    """Percentile bootstrap CI for effective dimensionality and near-collision
    fraction, resampling this group's own rows with replacement (sec 15 A17).

    Recomputes the same PCA/nearest-neighbour pipeline `diversity_metrics`
    uses on each resample rather than a closed-form approximation, so the
    CI reflects the actual estimator's sampling variability, not a
    normal-theory stand-in that may not hold at the small group sizes this
    is specifically for.
    """
    x = sub.features_scaled
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    eff_dims = np.empty(n_boot)
    collisions = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        pca = PCA(n_components=min(n, xb.shape[1])).fit(xb)
        eff_dims[b] = _participation_ratio(pca.explained_variance_)
        nn = NearestNeighbors(n_neighbors=min(2, n)).fit(xb)
        dists, _ = nn.kneighbors(xb)
        nn_dist = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
        threshold = np.quantile(nn_dist, near_collision_quantile)
        collisions[b] = float((nn_dist <= max(threshold, 1e-9)).mean())
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    eff_lo, eff_hi = np.quantile(eff_dims, [lo_q, hi_q])
    coll_lo, coll_hi = np.quantile(collisions, [lo_q, hi_q])
    return (round(float(eff_lo), 3), round(float(eff_hi), 3)), \
           (round(float(coll_lo), 4), round(float(coll_hi), 4))


def diversity_metrics_by_group(fm: FeatureMatrix, labels: list[str], min_group_size: int = 5,
                               near_collision_quantile: float = 0.05, n_boot: int = 200,
                               ci: float = 0.95, seed: int = 0) -> dict:
    """Rerun ``diversity_metrics`` separately for each label's rows, with a
    bootstrap CI and an explicit insufficient-n sentinel (sec 15 A17).

    A single global number can average away the failure that actually
    matters: one task or archetype collapsed onto a handful of shapes while
    another is genuinely broad. ``labels`` must be aligned to ``fm.ids``
    (e.g. ``[r.task for r in records]``, ``[r.tier for r in records]``, or
    ``[r.archetype for r in records]``). Groups with fewer than
    ``min_group_size`` members get an explicit ``InsufficientN`` entry
    instead of being silently omitted -- PCA and nearest-neighbour metrics
    are not meaningful on a handful of points, and a missing key is easy to
    misread as "not present in the corpus" rather than "not enough data."
    ``n_boot=0`` skips the bootstrap (cheaper, matches pre-A17 behavior
    exactly -- every existing field byte-identical, just no CI fields set).
    """
    if len(labels) != len(fm.ids):
        raise ValueError(f"labels must align with fm.ids: got {len(labels)} labels for {len(fm.ids)} rows")

    labels_arr = np.asarray(labels)
    out: dict = {}
    for gi, g in enumerate(sorted(set(labels))):
        mask = labels_arr == g
        n = int(mask.sum())
        if n < min_group_size:
            out[g] = InsufficientN(n_sequences=n, min_required=min_group_size)
            continue
        idx = np.where(mask)[0]
        sub = FeatureMatrix(
            features_raw=fm.features_raw[idx],
            features_scaled=fm.features_scaled[idx],
            feature_names=fm.feature_names,
            ids=[fm.ids[i] for i in idx],
            groups=[fm.groups[i] for i in idx],
            n_imputed=0,
        )
        report = diversity_metrics(sub, near_collision_quantile=near_collision_quantile)
        if n_boot > 0:
            report.effective_dimensionality_ci, report.near_collision_fraction_ci = \
                _bootstrap_group_ci(sub, n_boot, seed + gi, ci, near_collision_quantile)
        out[g] = report
    return out
