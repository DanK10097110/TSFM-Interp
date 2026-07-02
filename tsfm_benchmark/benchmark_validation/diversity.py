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


def diversity_metrics_by_group(fm: FeatureMatrix, labels: list[str], min_group_size: int = 5, near_collision_quantile: float = 0.05) -> dict[str, DiversityReport]:
    """Rerun ``diversity_metrics`` separately for each label's rows.

    A single global number can average away the failure that actually
    matters: one task or archetype collapsed onto a handful of shapes while
    another is genuinely broad. ``labels`` must be aligned to ``fm.ids``
    (e.g. ``[r.task for r in records]``, ``[r.tier for r in records]``, or
    ``[r.archetype for r in records]``). Groups with fewer than
    ``min_group_size`` members are skipped -- PCA and nearest-neighbour
    metrics are not meaningful on a handful of points, and a silently
    misleading number is worse than an omitted one.
    """
    if len(labels) != len(fm.ids):
        raise ValueError(f"labels must align with fm.ids: got {len(labels)} labels for {len(fm.ids)} rows")

    labels_arr = np.asarray(labels)
    out: dict[str, DiversityReport] = {}
    for g in sorted(set(labels)):
        mask = labels_arr == g
        if int(mask.sum()) < min_group_size:
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
        out[g] = diversity_metrics(sub, near_collision_quantile=near_collision_quantile)
    return out
