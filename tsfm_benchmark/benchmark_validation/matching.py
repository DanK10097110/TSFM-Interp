"""Algorithmic sequence matching for redundancy detection.

Two sequences are compared after being put on equal footing: z-normalised and
resampled to a common length (the "equal sized buckets" alignment), so the
comparison is invariant to length and absolute scale. Two matchers are offered.
Rolling cross-correlation slides one series against the other and takes the best
Pearson correlation over a band of lags, which captures linear shape similarity
under a small *uniform* shift only -- two sequences that are the same shape but
locally stretched or compressed (one regime lasting a bit longer, a seasonal
period drifting) score low even though they're genuine near-duplicates.
Localized DTW takes a windowed (Sakoe-Chiba banded) dynamic time warp and
converts the distance to a similarity, which additionally tolerates that local
warping, so it's the more robust matcher for "is this actually a duplicate"
and is the default here. xcorr remains available (``method="xcorr"``) as the
cheaper, shift-only check.

The pairwise similarities are summarised two ways: an equal-frequency bucketed
histogram of all scores (the redundancy profile of the whole benchmark) and an
explicit list of pairs above a redundancy threshold. The pairwise step is
O(n^2); for large benchmarks pass ``max_sequences`` to score a random subset.

Both matchers score every pair, but do it as a handful of batched matrix
operations over *all* pairs at once rather than a Python-level double loop
per pair: xcorr via one BLAS matmul per lag (elementwise-maxed together),
DTW via dtaidistance's parallel C distance matrix when available (the numpy
fallback below is used only if dtaidistance isn't installed, and stays
O(n^2 * length * window)). This is mathematically equivalent to scoring each
pair individually -- same checks, same thresholds -- just without paying
Python-call overhead n^2 (or n^2 * n_lags) times, which is what makes DTW
affordable as the default rather than only a fallback for small corpora.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .loaders import SeqRecord

try:
    from dtaidistance import dtw as _dtai_dtw

    _HAVE_DTAI = True
except ImportError:
    _HAVE_DTAI = False

logger = logging.getLogger("tsfm_benchmark.benchmark_validation.matching")


def _prepare(values: np.ndarray, length: int) -> np.ndarray:
    """Z-normalise and resample one sequence to the common bucket length."""
    v = np.asarray(values, dtype=float).reshape(-1)
    idx = np.linspace(0, len(v) - 1, length)
    r = np.interp(idx, np.arange(len(v)), v)
    return (r - r.mean()) / (r.std() + 1e-8)


def _max_xcorr_matrix(prepared: np.ndarray, max_lag: int) -> np.ndarray:
    """All-pairs best Pearson correlation over lags in [-max_lag, max_lag].

    Equivalent to calling a per-pair ``_max_xcorr`` on every (i, j), but
    batched: for each lag ``L >= 0`` we correlate every row shifted by ``L``
    against every un-shifted row in one matmul (``Xn @ Yn.T``, both rows
    standardised so the dot product over the overlap *is* the Pearson
    correlation), giving the whole n x n matrix for that lag at once. The
    negative-lag matrix for the same ``L`` is just the transpose of the
    positive one (correlation is symmetric under swapping which sequence is
    "shifted"), so only ``max_lag + 1`` matmuls are needed to cover the full
    symmetric lag band.
    """
    n, length = prepared.shape
    best = np.full((n, n), -1.0)
    eps = 1e-12

    for lag in range(0, max_lag + 1):
        X = prepared[:, lag:] if lag else prepared
        Y = prepared[:, : length - lag] if lag else prepared
        w = X.shape[1]
        if w < 8:
            break

        Xc = X - X.mean(axis=1, keepdims=True)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        Xs = Xc.std(axis=1)
        Ys = Yc.std(axis=1)
        degenerate = (Xs < eps) | (Ys < eps)
        Xs_safe = np.where(Xs < eps, 1.0, Xs)
        Ys_safe = np.where(Ys < eps, 1.0, Ys)
        Xn = Xc / Xs_safe[:, None]
        Yn = Yc / Ys_safe[:, None]

        corr = (Xn @ Yn.T) / w
        corr[degenerate, :] = -1.0
        corr[:, degenerate] = -1.0

        if lag == 0:
            np.maximum(best, corr, out=best)
        else:
            np.maximum(best, corr, out=best)
            np.maximum(best, corr.T, out=best)

    return best


def _max_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Best Pearson correlation over integer lags in [-max_lag, max_lag].

    Kept for single-pair use (tests, ad hoc checks); ``match_all`` uses the
    batched ``_max_xcorr_matrix`` instead.
    """
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
    """Localized DTW distance mapped to a (0, 1] similarity.

    Kept for single-pair use; ``match_all`` uses the batched
    ``_dtw_similarity_matrix`` instead.
    """
    if _HAVE_DTAI:
        dist = float(_dtai_dtw.distance_fast(a, b, window=window, use_pruning=True))
    else:
        dist = _dtw_numpy(a, b, window)
    return float(1.0 / (1.0 + dist / np.sqrt(length)))


def _dtw_similarity_matrix(prepared: list[np.ndarray], window: int, length: int) -> np.ndarray:
    """All-pairs localized-DTW similarity.

    When dtaidistance is installed, this hands the whole batch to its
    parallel C distance matrix (``distance_matrix_fast``) instead of making
    n^2 individual Python calls into the C extension. Distances come back
    identical to calling ``distance_fast`` pair by pair (verified against
    it directly); only the calling overhead changes. Without dtaidistance,
    falls back to the pure-Python banded DTW per pair, logging progress
    periodically since that path stays O(n^2 * length * window).
    """
    n = len(prepared)
    if _HAVE_DTAI:
        dist = np.asarray(_dtai_dtw.distance_matrix_fast(prepared, window=window))
    else:
        dist = np.zeros((n, n))
        total_pairs = n * (n - 1) // 2
        done = 0
        last_log = time.perf_counter()
        for i in range(n):
            for j in range(i + 1, n):
                d = _dtw_numpy(prepared[i], prepared[j], window)
                dist[i, j] = dist[j, i] = d
                done += 1
                now = time.perf_counter()
                if now - last_log > 2.0:
                    logger.info(
                        "dtw fallback (dtaidistance not installed) progress: %d/%d pairs (%.1f%%)",
                        done, total_pairs, 100.0 * done / max(1, total_pairs),
                    )
                    last_log = now
    return 1.0 / (1.0 + dist / np.sqrt(length))


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
    method: str = "dtw",
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
    if method not in ("xcorr", "dtw"):
        raise ValueError(f"unknown method '{method}'")

    if max_sequences and len(records) > max_sequences:
        logger.info("subsampling %d sequences down to max_sequences=%d (seed=%d)", len(records), max_sequences, seed)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(records), size=max_sequences, replace=False)
        records = [records[i] for i in sorted(idx)]

    n = len(records)
    n_pairs = n * (n - 1) // 2
    max_lag = max(1, int(lag_frac * length))
    window = max(1, int(window_frac * length))
    logger.info("match_all: n=%d sequences, %d pairs, method=%s, bucket_length=%d", n, n_pairs, method, length)
    logger.debug("match_all params: lag_frac=%s window_frac=%s max_lag=%d window=%d redundancy_threshold=%s", lag_frac, window_frac, max_lag, window, redundancy_threshold)

    t0 = time.perf_counter()
    prepared = np.stack([_prepare(r.values, length) for r in records]) if n else np.zeros((0, length))
    ids = [r.seq_id for r in records]
    logger.debug("prepared %d sequences (z-normalised, resampled to length=%d) in %.2fs", n, length, time.perf_counter() - t0)

    t0 = time.perf_counter()
    if method == "xcorr":
        sim = _max_xcorr_matrix(prepared, max_lag)
        np.fill_diagonal(sim, 1.0)
    else:
        dtai_note = "dtaidistance (parallel C)" if _HAVE_DTAI else "pure-Python fallback"
        logger.debug("computing DTW similarity matrix via %s", dtai_note)
        sim = _dtw_similarity_matrix(list(prepared), window, length)
    logger.info("computed %dx%d similarity matrix in %.2fs", n, n, time.perf_counter() - t0)

    off = sim[np.triu_indices(n, k=1)]
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(off, quantiles) if len(off) else np.zeros(n_buckets + 1)
    edges = np.unique(edges)
    counts, edges = np.histogram(off, bins=edges) if len(edges) > 1 else (np.array([len(off)]), edges)

    tri_i, tri_j = np.triu_indices(n, k=1)
    above = off >= redundancy_threshold
    redundant = [(ids[i], ids[j], float(sim[i, j])) for i, j in zip(tri_i[above], tri_j[above])]
    redundant.sort(key=lambda t: t[2], reverse=True)
    frac = float(len(redundant) / max(1, len(off)))
    logger.info("redundancy: %d/%d pairs (%.2f%%) at or above threshold %.3f", len(redundant), len(off), 100.0 * frac, redundancy_threshold)

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


def redundancy_by_group(match: MatchReport, id_to_label: dict[str, str]) -> dict[str, Any]:
    """Split the pairwise similarity matrix into within-group and cross-group means.

    A global redundancy fraction can't tell you *why* it's high: it looks the
    same whether one tier is internally repetitive or two tiers just happen
    to look alike. This answers that by averaging similarity separately for
    pairs sharing a label (e.g. both 'synthetic') and pairs that don't.
    ``match.ids`` may be a subsample of the full corpus (``max_sequences``);
    ``id_to_label`` only needs to cover whichever ids ended up in ``match``.
    """
    labels = np.array([id_to_label.get(i, "unknown") for i in match.ids])
    sim = match.similarity_matrix
    n = len(labels)

    tri_i, tri_j = np.triu_indices(n, k=1)
    same_group = labels[tri_i] == labels[tri_j]
    sim_pairs = sim[tri_i, tri_j]

    within: dict[str, list[float]] = {}
    for g in np.unique(labels):
        mask = same_group & (labels[tri_i] == g)
        if mask.any():
            within[str(g)] = sim_pairs[mask].tolist()
    across = sim_pairs[~same_group].tolist()
    logger.debug("redundancy_by_group: %d groups, %d within-group pairs, %d across-group pairs", len(within), len(sim_pairs) - len(across), len(across))

    return {
        "within_group_mean_similarity": {g: round(float(np.mean(v)), 4) for g, v in sorted(within.items())},
        "across_group_mean_similarity": round(float(np.mean(across)), 4) if across else None,
        "n_pairs_within": {g: len(v) for g, v in sorted(within.items())},
        "n_pairs_across": len(across),
    }
