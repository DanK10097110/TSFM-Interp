"""Algorithmic sequence matching for redundancy detection.

Two sequences are compared after being put on equal footing: z-normalised and
resampled to a common length (the "equal sized buckets" alignment), so the
comparison is invariant to length and absolute scale. Two matchers are offered.
Rolling cross-correlation slides one series against the other and takes the best
Pearson correlation over a band of lags, which captures linear shape similarity
under small shifts. Localized DTW takes a windowed (Sakoe-Chiba banded) dynamic
time warp and converts the distance to a similarity, which additionally tolerates
local time warping.

The pairwise similarities are summarised two ways: an equal-frequency bucketed
histogram of all scores (the redundancy profile of the whole benchmark) and an
explicit list of pairs above a redundancy threshold. The pairwise step is
O(n^2); for large benchmarks pass ``max_sequences`` to score a random subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loaders import SeqRecord

try:
    from dtaidistance import dtw as _dtai_dtw

    _HAVE_DTAI = True
except ImportError:
    _HAVE_DTAI = False


def _prepare(values: np.ndarray, length: int) -> np.ndarray:
    """Z-normalise and resample one sequence to the common bucket length."""
    v = np.asarray(values, dtype=float).reshape(-1)
    idx = np.linspace(0, len(v) - 1, length)
    r = np.interp(idx, np.arange(len(v)), v)
    return (r - r.mean()) / (r.std() + 1e-8)


def _max_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Best Pearson correlation over integer lags in [-max_lag, max_lag]."""
    best = -1.0
    n = len(a)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: n - lag]
        else:
            x, y = a[: n + lag], b[-lag:]
        if len(x) < 8:
            continue
        denom = x.std() * y.std()
        if denom < 1e-12:
            continue
        c = float(np.mean((x - x.mean()) * (y - y.mean())) / denom)
        best = max(best, c)
    return best


def _dtw_numpy(a: np.ndarray, b: np.ndarray, window: int) -> float:
    n, m = len(a), len(b)
    w = max(window, abs(n - m))
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(np.sqrt(d[n, m]))


def _dtw_similarity(a: np.ndarray, b: np.ndarray, window: int, length: int) -> float:
    """Localized DTW distance mapped to a (0, 1] similarity."""
    if _HAVE_DTAI:
        dist = float(_dtai_dtw.distance_fast(a, b, window=window, use_pruning=True))
    else:
        dist = _dtw_numpy(a, b, window)
    return float(1.0 / (1.0 + dist / np.sqrt(length)))


@dataclass
class MatchReport:
    method: str
    n_sequences: int
    similarity_matrix: np.ndarray
    ids: list[str]
    bucket_edges: list[float]
    bucket_counts: list[int]
    redundant_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    redundancy_fraction: float = 0.0


def match_all(
    records: list[SeqRecord],
    method: str = "xcorr",
    length: int = 256,
    lag_frac: float = 0.1,
    window_frac: float = 0.1,
    n_buckets: int = 10,
    redundancy_threshold: float = 0.95,
    max_sequences: int | None = None,
    seed: int = 0,
) -> MatchReport:
    """Score every pair of sequences and summarise the redundancy of the set.

    Returns the full similarity matrix, an equal-frequency bucketed histogram of
    the off-diagonal scores, and the pairs whose similarity meets the redundancy
    threshold. A high redundancy fraction means the benchmark is wasting slots on
    near-duplicates.
    """
    if max_sequences and len(records) > max_sequences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(records), size=max_sequences, replace=False)
        records = [records[i] for i in sorted(idx)]

    prepared = [_prepare(r.values, length) for r in records]
    ids = [r.seq_id for r in records]
    n = len(prepared)
    max_lag = max(1, int(lag_frac * length))
    window = max(1, int(window_frac * length))

    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if method == "xcorr":
                s = _max_xcorr(prepared[i], prepared[j], max_lag)
            elif method == "dtw":
                s = _dtw_similarity(prepared[i], prepared[j], window, length)
            else:
                raise ValueError(f"unknown method '{method}'")
            sim[i, j] = sim[j, i] = s

    off = sim[np.triu_indices(n, k=1)]
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(off, quantiles) if len(off) else np.zeros(n_buckets + 1)
    edges = np.unique(edges)
    counts, edges = np.histogram(off, bins=edges) if len(edges) > 1 else (np.array([len(off)]), edges)

    redundant = [(ids[i], ids[j], float(sim[i, j])) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= redundancy_threshold]
    redundant.sort(key=lambda t: t[2], reverse=True)
    frac = float(len(redundant) / max(1, len(off)))

    return MatchReport(
        method=method,
        n_sequences=n,
        similarity_matrix=sim,
        ids=ids,
        bucket_edges=[float(e) for e in edges],
        bucket_counts=[int(c) for c in counts],
        redundant_pairs=redundant,
        redundancy_fraction=frac,
    )
