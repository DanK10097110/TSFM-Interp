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
from sklearn.cluster import KMeans

from .loaders import SeqRecord

try:
    from dtaidistance import dtw as _dtai_dtw

    _HAVE_DTAI = True
except ImportError:
    _HAVE_DTAI = False

logger = logging.getLogger("tsfm_benchmark.benchmark_validation.matching")


def _stratified_subsample(records: list[SeqRecord], k: int, seed: int) -> list[int]:
    """Stratified-without-replacement subsample by generator (sec 15 A17).

    A plain `rng.choice` over the whole corpus draws proportionally from
    whichever generators happen to be most numerous, which is *random* (not
    a head-slice bias) but still leaves per-group redundancy/matching
    numbers measuring mostly whichever generator dominates the corpus by
    count rather than a representative cross-section. Deliberately a small
    local copy of the same proportional-allocation idea `tsfm_lens`'s
    `utils.sample_rows` uses, not an import of it -- the two packages are
    kept decoupled by design (`loaders.py`'s own module docstring).
    """
    groups = np.array([r.group for r in records])
    labels, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv, minlength=len(labels))
    raw_alloc = counts / counts.sum() * k
    alloc = np.floor(raw_alloc).astype(int)
    remainder = k - int(alloc.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw_alloc - alloc))
        for i in frac_order[:remainder]:
            alloc[i] += 1
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for i in range(len(labels)):
        pool = np.where(inv == i)[0]
        take = min(int(alloc[i]), len(pool))
        if take > 0:
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return sorted(chosen)


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
    # sec 15 A17: the exact O(n^2) pass always covers every pair
    # (coverage_fraction=1.0, blocked=False). The blocked/approximate
    # matcher only scores pairs within the same or a neighboring catch22-
    # shape cluster and leaves the rest unscored (NaN in `similarity_matrix`,
    # excluded from `redundancy_fraction`'s denominator) -- this is the
    # exactness trade-off the fix explicitly requires be stated in the
    # report, not buried in a docstring.
    coverage_fraction: float = 1.0
    blocked: bool = False
    n_blocks: int | None = None


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
    blocked: bool = False,
    n_blocks: int | None = None,
    adjacent_k: int = 2,
) -> MatchReport:
    """Score every pair of sequences and summarise the redundancy of the set.

    Returns the full similarity matrix, an equal-frequency bucketed histogram of
    the off-diagonal scores, and the pairs whose similarity meets the redundancy
    threshold. A high redundancy fraction means the benchmark is wasting slots on
    near-duplicates.

    ``blocked=True`` switches to an approximate O(n*k) matcher for corpora too
    large for the exact O(n^2) pass (sec 15 A17); see `_match_all_blocked`.
    """
    if method not in ("xcorr", "dtw"):
        raise ValueError(f"unknown method '{method}'")

    if max_sequences and len(records) > max_sequences:
        logger.info("subsampling %d sequences down to max_sequences=%d (seed=%d, stratified by generator)",
                   len(records), max_sequences, seed)
        idx = _stratified_subsample(records, max_sequences, seed)
        records = [records[i] for i in idx]

    if blocked:
        return _match_all_blocked(records, method, length, lag_frac, window_frac, n_buckets,
                                  redundancy_threshold, seed, n_blocks, adjacent_k)
    return _match_all_exact(records, method, length, lag_frac, window_frac, n_buckets,
                            redundancy_threshold)


def _match_all_exact(records: list[SeqRecord], method: str, length: int, lag_frac: float,
                     window_frac: float, n_buckets: int, redundancy_threshold: float) -> MatchReport:
    """The original exact O(n^2) pass, unchanged -- every existing call site
    that doesn't pass `blocked=True` reproduces byte-for-byte (sec 15 A17)."""
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


def _match_all_blocked(records: list[SeqRecord], method: str, length: int, lag_frac: float,
                       window_frac: float, n_buckets: int, redundancy_threshold: float,
                       seed: int, n_blocks: int | None, adjacent_k: int) -> MatchReport:
    """Approximate matcher for corpora too large for the exact O(n^2) pass (sec 15 A17).

    Sequences are bucketed by shape (KMeans over the same z-normalised,
    resampled arrays the exact matcher itself compares -- a self-contained
    proxy for "catch22 neighbourhood" that doesn't add a new coupling to
    `features.py`, a deliberate implementation choice, not the literal
    "catch22 neighbourhood" the fix's own wording suggested). Exact
    similarity (the same batched `_max_xcorr_matrix`/`_dtw_similarity_matrix`
    routines the exact path uses -- no new numerical code) is computed only
    for pairs whose two sequences fall in the same bucket or in each other's
    `adjacent_k` nearest buckets by centroid distance; every other pair is
    left unscored (`NaN` in `similarity_matrix`) rather than assumed
    non-redundant, and excluded from `redundancy_fraction`'s denominator.
    `coverage_fraction` states exactly what fraction of all possible pairs
    were actually scored -- the stated exactness trade-off the fix requires.
    """
    n = len(records)
    ids = [r.seq_id for r in records]
    if n_blocks is None:
        n_blocks = max(1, int(np.sqrt(max(1, n) / 2)))
    n_blocks = max(1, min(n_blocks, n))

    if n < 3 or n_blocks <= 1:
        logger.info("match_all(blocked=True): n=%d too small to block meaningfully; "
                   "falling back to the exact O(n^2) pass", n)
        exact = _match_all_exact(records, method, length, lag_frac, window_frac, n_buckets,
                                 redundancy_threshold)
        exact.coverage_fraction, exact.blocked, exact.n_blocks = 1.0, True, 1
        return exact

    prepared = np.stack([_prepare(r.values, length) for r in records])
    max_lag = max(1, int(lag_frac * length))
    window = max(1, int(window_frac * length))

    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit(prepared)
    block_of = km.labels_
    centroids = km.cluster_centers_
    cdist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    neighbor_blocks = []
    for b in range(n_blocks):
        order = [o for o in np.argsort(cdist[b]) if o != b]
        neighbor_blocks.append({b, *order[:adjacent_k]})
    logger.info("match_all(blocked=True): n=%d into %d shape-clusters (adjacent_k=%d) in %.2fs",
               n, n_blocks, adjacent_k, time.perf_counter() - t0)

    sim = np.full((n, n), np.nan)
    np.fill_diagonal(sim, 1.0)
    t0 = time.perf_counter()
    done_block_pairs = set()
    for bi in range(n_blocks):
        for bj in neighbor_blocks[bi]:
            key = (min(bi, bj), max(bi, bj))
            if key in done_block_pairs:
                continue
            done_block_pairs.add(key)
            rows_i = np.where(block_of == bi)[0]
            rows_j = np.where(block_of == bj)[0] if bj != bi else rows_i
            combined = np.unique(np.concatenate([rows_i, rows_j]))
            if method == "xcorr":
                sub_sim = _max_xcorr_matrix(prepared[combined], max_lag)
                np.fill_diagonal(sub_sim, 1.0)
            else:
                sub_sim = _dtw_similarity_matrix(list(prepared[combined]), window, length)
            local_of = {g: l for l, g in enumerate(combined)}
            for gi in rows_i:
                for gj in rows_j:
                    if gi >= gj or not np.isnan(sim[gi, gj]):
                        continue
                    sim[gi, gj] = sim[gj, gi] = sub_sim[local_of[gi], local_of[gj]]
    logger.info("match_all(blocked=True): scored block-adjacency pairs in %.2fs", time.perf_counter() - t0)

    tri_i, tri_j = np.triu_indices(n, k=1)
    pair_sims = sim[tri_i, tri_j]
    scored_mask = ~np.isnan(pair_sims)
    total_pairs = n * (n - 1) // 2
    coverage = float(scored_mask.sum()) / max(1, total_pairs)

    off_scored = pair_sims[scored_mask]
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(off_scored, quantiles) if len(off_scored) else np.zeros(n_buckets + 1)
    edges = np.unique(edges)
    counts, edges = np.histogram(off_scored, bins=edges) if len(edges) > 1 else (np.array([len(off_scored)]), edges)

    above = scored_mask & (pair_sims >= redundancy_threshold)
    redundant = [(ids[i], ids[j], float(pair_sims[k]))
                for k, (i, j) in enumerate(zip(tri_i, tri_j)) if above[k]]
    redundant.sort(key=lambda t: t[2], reverse=True)
    frac = float(above.sum() / max(1, scored_mask.sum()))
    logger.info("match_all(blocked=True): coverage=%.1f%% (%d/%d pairs), redundancy %d/%d "
               "scored pairs (%.2f%%) at or above threshold %.3f", 100 * coverage,
               int(scored_mask.sum()), total_pairs, len(redundant), int(scored_mask.sum()),
               100.0 * frac, redundancy_threshold)

    return MatchReport(
        method=method, n_sequences=n, similarity_matrix=sim, ids=ids,
        bucket_edges=[float(e) for e in edges], bucket_counts=[int(c) for c in counts],
        redundant_pairs=redundant, redundancy_fraction=frac,
        coverage_fraction=round(coverage, 4), blocked=True, n_blocks=n_blocks,
    )


def redundancy_by_group(match: MatchReport, id_to_label: dict[str, str]) -> dict[str, Any]:
    """Split the pairwise similarity matrix into within-group and cross-group means.

    A global redundancy fraction can't tell you *why* it's high: it looks the
    same whether one tier is internally repetitive or two tiers just happen
    to look alike. This answers that by averaging similarity separately for
    pairs sharing a label (e.g. both 'synthetic') and pairs that don't.
    ``match.ids`` may be a subsample of the full corpus (``max_sequences``);
    ``id_to_label`` only needs to cover whichever ids ended up in ``match``.

    Under ``match.blocked=True`` (sec 15 A17), unscored pairs are ``NaN`` in
    ``similarity_matrix`` and excluded here too -- a group whose members
    landed in scattered shape-clusters can end up with very few *scored*
    within-group pairs even though it has many members, which
    ``n_pairs_within`` reports honestly rather than silently averaging over
    fewer pairs than a reader would assume from the group's size.
    """
    labels = np.array([id_to_label.get(i, "unknown") for i in match.ids])
    sim = match.similarity_matrix
    n = len(labels)

    tri_i, tri_j = np.triu_indices(n, k=1)
    same_group = labels[tri_i] == labels[tri_j]
    sim_pairs = sim[tri_i, tri_j]
    scored = ~np.isnan(sim_pairs)

    within: dict[str, list[float]] = {}
    for g in np.unique(labels):
        mask = same_group & (labels[tri_i] == g) & scored
        if mask.any():
            within[str(g)] = sim_pairs[mask].tolist()
    across = sim_pairs[(~same_group) & scored].tolist()
    logger.debug("redundancy_by_group: %d groups, %d within-group pairs, %d across-group pairs", len(within), len(sim_pairs) - len(across), len(across))

    return {
        "within_group_mean_similarity": {g: round(float(np.mean(v)), 4) for g, v in sorted(within.items())},
        "across_group_mean_similarity": round(float(np.mean(across)), 4) if across else None,
        "n_pairs_within": {g: len(v) for g, v in sorted(within.items())},
        "n_pairs_across": len(across),
        "coverage_fraction": match.coverage_fraction,
    }
