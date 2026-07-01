"""catch22 feature extraction.

Passes every sequence through the catch22 set of 22 canonical, low-redundancy
time-series features (or catch24, which adds mean and standard deviation). The
features live on very different scales and catch22 can emit non-finite values
for degenerate inputs, so this module returns the raw matrix plus a robustly
scaled matrix (median/IQR) with non-finite entries imputed to the per-feature
median, and it reports how many sequences needed imputation. Distance-based
diversity metrics and the embedding both consume the scaled matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pycatch22

    _HAVE_CATCH22 = True
except ImportError:
    _HAVE_CATCH22 = False

from sklearn.preprocessing import RobustScaler

from .loaders import SeqRecord


@dataclass
class FeatureMatrix:
    features_raw: np.ndarray
    features_scaled: np.ndarray
    feature_names: list[str]
    ids: list[str]
    groups: list[str]
    n_imputed: int


def extract_features(records: list[SeqRecord], catch24: bool = False) -> FeatureMatrix:
    """Build the (n_sequences, n_features) catch22/catch24 matrix.

    Non-finite feature values are replaced with the per-feature median and
    counted, so a few degenerate sequences cannot poison the scaler or the
    embedding.
    """
    if not _HAVE_CATCH22:
        raise ImportError("install pycatch22 to extract catch22 features")

    rows = []
    names: list[str] = []
    for r in records:
        out = pycatch22.catch22_all(r.values.tolist(), catch24=catch24)
        names = out["names"]
        rows.append(out["values"])

    raw = np.asarray(rows, dtype=float)
    finite = np.isfinite(raw)
    n_imputed = int((~finite).any(axis=1).sum())
    medians = np.nanmedian(np.where(finite, raw, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(finite, raw, medians)

    scaled = RobustScaler().fit_transform(filled)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return FeatureMatrix(
        features_raw=raw,
        features_scaled=scaled,
        feature_names=names,
        ids=[r.seq_id for r in records],
        groups=[r.group for r in records],
        n_imputed=n_imputed,
    )
