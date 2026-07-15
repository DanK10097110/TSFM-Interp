"""catch22 feature extraction.

Passes every sequence through the catch22 set of 22 canonical, low-redundancy
time-series features (or catch24, which adds mean and standard deviation). The
features live on very different scales and catch22 can emit non-finite values
for degenerate inputs, so this module returns the raw matrix plus a robustly
scaled matrix (median/IQR) with non-finite entries imputed to the per-feature
median, and it reports how many sequences needed imputation. Distance-based
diversity metrics and the embedding both consume the scaled matrix.

RobustScaler alone isn't enough to make features comparable, though: it scales
each column so its *typical* spread (the interquartile range) becomes 1, but
says nothing about the tails. A feature like ``CO_trev_1_num`` (time
reversibility) is tightly concentrated for most series -- its IQR can be on
the order of 1e-2 -- while a handful of series with near-constant runs push
it to values 1000x the bulk. Dividing by that tiny IQR turns those few
outliers into scaled values in the thousands, and because every downstream
multivariate metric (variance ranking, PCA effective dimensionality,
nearest-neighbour distance) is an L2 quantity, a handful of such values on one
column silently swamp the other 21-23 features combined. Winsorizing the
scaled matrix to +/-``clip_scaled`` keeps those series flagged as outliers
(they still sit at the clip boundary, more extreme than everything else) without
letting them single-handedly define the feature space.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import pycatch22

    _HAVE_CATCH22 = True
except ImportError:
    _HAVE_CATCH22 = False

from sklearn.preprocessing import RobustScaler

from .loaders import SeqRecord

logger = logging.getLogger("tsfm_benchmark.benchmark_validation.features")


@dataclass
class FeatureMatrix:
    features_raw: np.ndarray
    features_scaled: np.ndarray
    feature_names: list[str]
    ids: list[str]
    groups: list[str]
    n_imputed: int
    n_clipped: int = 0
    # Per-sequence anomaly diagnostics, from the *pre-winsorization* scaled
    # matrix: which single feature was most extreme for that sequence, and by
    # how much. Empty for FeatureMatrix instances assembled by hand (e.g. the
    # per-group slices in ``diversity_metrics_by_group``) rather than built by
    # ``extract_features``.
    anomaly_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    anomaly_features: list[str] = field(default_factory=list)
    anomaly_raw_values: list[float] = field(default_factory=list)


def extract_features(records: list[SeqRecord], catch24: bool = False, clip_scaled: float | None = 5.0) -> FeatureMatrix:
    """Build the (n_sequences, n_features) catch22/catch24 matrix.

    Non-finite feature values are replaced with the per-feature median and
    counted, so a few degenerate sequences cannot poison the scaler or the
    embedding. Finite-but-extreme values are then winsorized to
    +/-``clip_scaled`` robust-scale units (see module docstring) for the same
    reason -- pass ``clip_scaled=None`` to disable and get the raw
    RobustScaler output.
    """
    if not _HAVE_CATCH22:
        raise ImportError("install pycatch22 to extract catch22 features")

    n = len(records)
    logger.info("extract_features: computing catch%d features for %d sequences", 24 if catch24 else 22, n)
    t0 = time.perf_counter()
    log_every = max(1, n // 10)

    rows = []
    names: list[str] = []
    for i, r in enumerate(records):
        out = pycatch22.catch22_all(r.values.tolist(), catch24=catch24)
        names = out["names"]
        rows.append(out["values"])
        if (i + 1) % log_every == 0 or i + 1 == n:
            logger.debug("extract_features: %d/%d sequences done", i + 1, n)

    raw = np.asarray(rows, dtype=float)
    finite = np.isfinite(raw)
    n_imputed = int((~finite).any(axis=1).sum())
    medians = np.nanmedian(np.where(finite, raw, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(finite, raw, medians)

    scaled = RobustScaler().fit_transform(filled)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    trigger_col = np.argmax(np.abs(scaled), axis=1)
    rows_idx = np.arange(len(scaled))
    anomaly_scores = np.abs(scaled[rows_idx, trigger_col])
    anomaly_features = [names[c] for c in trigger_col]
    anomaly_raw_values = raw[rows_idx, trigger_col].tolist()

    n_clipped = 0
    if clip_scaled is not None:
        n_clipped = int(np.sum(np.abs(scaled) > clip_scaled))
        if n_clipped:
            worst = np.unravel_index(np.argmax(np.abs(scaled)), scaled.shape)
            logger.info(
                "extract_features: winsorizing %d/%d scaled values to +/-%s (worst: feature=%s value=%.1f)",
                n_clipped, scaled.size, clip_scaled, names[worst[1]], scaled[worst],
            )
        scaled = np.clip(scaled, -clip_scaled, clip_scaled)

    logger.info("extract_features: done in %.2fs (%d/%d sequences needed imputation)", time.perf_counter() - t0, n_imputed, n)

    return FeatureMatrix(
        features_raw=raw,
        features_scaled=scaled,
        feature_names=names,
        ids=[r.seq_id for r in records],
        groups=[r.group for r in records],
        n_imputed=n_imputed,
        n_clipped=n_clipped,
        anomaly_scores=anomaly_scores,
        anomaly_features=anomaly_features,
        anomaly_raw_values=anomaly_raw_values,
    )
