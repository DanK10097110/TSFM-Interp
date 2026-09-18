"""Leakage and realism auditing.

This module is what lets the benchmark *claim* anything about leakage rather
than assert it. For every candidate series it measures the shape distance to its
nearest neighbour across the real reference corpora; anything closer than a
threshold is flagged as a possible instance-level leak and rejected by the
builder. It also detects near-duplicates inside the benchmark and reports how
closely the synthetic distribution matches a real target on summary statistics.

The leakage gate is two-stage. A cheap z-normalised Euclidean nearest-neighbour
search prefilters the reference corpus to the ``prefilter_k`` closest series,
then a windowed DTW is computed only against those, and the minimum DTW distance
is the reported score. This keeps DTW's shift-invariance on the decision that
matters while staying close to plain-Euclidean cost: measured at roughly 3 ms
per query against a 1000-series corpus versus 2.4 ms for Euclidean alone, and
~20x faster than DTW against the full corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import TimeSeriesSample

try:
    from dtaidistance import dtw as _dtai_dtw

    _HAVE_DTAI = True
except ImportError:
    _HAVE_DTAI = False


def _normalize(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample to a common length and z-normalise for shape comparison."""
    x = np.asarray(x, dtype=float).reshape(-1)
    idx = np.linspace(0, len(x) - 1, target_len)
    r = np.interp(idx, np.arange(len(x)), x)
    return (r - r.mean()) / (r.std() + 1e-8)


def _dtw_numpy(a: np.ndarray, b: np.ndarray, window: int) -> float:
    """Banded DTW fallback used when dtaidistance is not installed."""
    n, m = len(a), len(b)
    w = max(window, abs(n - m))
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(np.sqrt(d[n, m]))


def _dtw(a: np.ndarray, b: np.ndarray, window: int) -> float:
    """Windowed DTW distance, using the fast C backend when available."""
    if _HAVE_DTAI:
        return float(_dtai_dtw.distance_fast(a, b, window=window, use_pruning=True))
    return _dtw_numpy(a, b, window)


@dataclass
class LeakageReport:
    nearest_distance: float
    nearest_corpus: str
    nearest_item: str
    metric: str
    passed: bool
    threshold: float


class LeakageAuditor:
    """Holds the reference corpus and scores candidates against it.

    ``metric`` is 'dtw' (two-stage, shift-invariant) or 'euclidean'. Distances
    are length-normalised so a threshold is independent of ``target_len``, but
    DTW and Euclidean live on different scales: calibrate ``threshold`` per
    metric against a held-out batch before trusting the gate.
    """

    def __init__(
        self,
        target_len: int = 256,
        threshold: float = 0.35,
        metric: str = "dtw",
        prefilter_k: int = 10,
        dtw_window_frac: float = 0.1,
    ) -> None:
        self.target_len = target_len
        self.threshold = threshold
        self.metric = metric
        self.prefilter_k = prefilter_k
        self.window = max(1, int(dtw_window_frac * target_len))
        self._refs: list[tuple[str, str, np.ndarray]] = []
        self._matrix: np.ndarray | None = None
        self._licenses: dict[str, str] = {}
        self._raw_refs: list[np.ndarray] = []

    def add_reference(self, corpus: str, item_id: str, values: np.ndarray, license: str = "unknown") -> None:
        """Register one real series the benchmark must stay away from.

        ``license`` (from the source's own `SourceRef.license`, ROADMAP.md
        sec 34 item B1) is recorded per corpus name so a persisted audit block
        can state which real corpora the gate compared against and under what
        license -- it is not used by the gate computation itself. The raw
        (pre-normalization) values are kept too, so a realism comparison
        (``audit.realism_report``) can be computed later against the exact
        reference set the gate used, without re-loading it.
        """
        self._refs.append((corpus, item_id, _normalize(values, self.target_len)))
        self._raw_refs.append(np.asarray(values, dtype=float))
        self._licenses[corpus] = license
        self._matrix = None

    def _ensure_matrix(self) -> None:
        if self._matrix is None and self._refs:
            self._matrix = np.stack([r for _, _, r in self._refs])

    def audit(self, sample: TimeSeriesSample) -> LeakageReport:
        """Score one sample against the reference corpus and gate it.

        A higher distance is safer; ``passed`` is True when the nearest-neighbour
        distance meets or exceeds the threshold.
        """
        if not self._refs:
            report = LeakageReport(float("inf"), "", "", self.metric, True, self.threshold)
            sample.leakage_report = report.__dict__
            return report

        self._ensure_matrix()
        q = _normalize(sample.values, self.target_len)
        denom = np.sqrt(self.target_len)
        euclid = np.linalg.norm(self._matrix - q, axis=1)

        if self.metric == "euclidean":
            best = int(np.argmin(euclid))
            dist = float(euclid[best] / denom)
        else:
            k = min(self.prefilter_k, len(self._refs))
            cand = np.argsort(euclid)[:k]
            dtw_d = [(_dtw(q, self._refs[i][2], self.window) / denom, i) for i in cand]
            dist, best = min(dtw_d)

        corpus, item_id, _ = self._refs[best]
        report = LeakageReport(dist, corpus, item_id, self.metric, dist >= self.threshold, self.threshold)
        sample.leakage_report = report.__dict__
        return report


def find_near_duplicates(samples: list[TimeSeriesSample], target_len: int = 256, threshold: float = 0.05) -> list[tuple[str, str, float]]:
    """Return pairs of benchmark samples whose shapes are nearly identical.

    Uses a KD-tree radius search rather than an all-pairs scan: the naive
    O(n^2) comparison becomes the dominant cost once a build produces tens of
    thousands of samples (e.g. a multi-domain, gigabyte-scale run), while a
    tree-based fixed-radius pair query stays tractable at that scale and
    returns the exact same pairs.
    """
    if len(samples) < 2:
        return []

    from scipy.spatial import cKDTree

    denom = np.sqrt(target_len)
    data = np.stack([_normalize(s.values, target_len) for s in samples])
    ids = [s.sample_id for s in samples]

    tree = cKDTree(data)
    candidate_pairs = tree.query_pairs(r=threshold * denom, output_type="ndarray")

    dups: list[tuple[str, str, float]] = []
    for i, j in candidate_pairs:
        d = float(np.linalg.norm(data[i] - data[j]) / denom)
        if d < threshold:
            dups.append((ids[i], ids[j], d))
    return dups


def _quantiles(values: list[float]) -> dict[str, float] | None:
    """Quantile summary of a distance list, or ``None`` when it means nothing.

    ``None`` (never a degenerate all-zero/all-inf dict) is deliberate: with no
    references every sample's ``nearest_distance`` is `inf` (``LeakageReport``'s
    own no-op default), and a quantile block full of ``inf`` would read as a
    measurement rather than as "the gate did not run" (CLAUDE.md sec 11.37 --
    an absent measurement must never render as a passing, or any, one).
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return {
        "min": float(arr.min()),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
    }


def compose_audit_block(
    auditor: "LeakageAuditor",
    kept: list[TimeSeriesSample],
    rejected: list[tuple[str, float]],
    dup_pairs: list[tuple[str, str, float]],
    split_of: dict[str, str],
    near_dup_threshold: float = 0.05,
    near_dup_target_len: int = 256,
    worst_k: int = 50,
    compute_realism: bool = True,
) -> dict[str, Any]:
    """Compose the leakage/near-duplicate/realism verdict into one persistable block.

    This performs **no new gate computation** (ROADMAP.md sec 34 item B1.1):
    every number here was already produced by ``LeakageAuditor.audit`` (set on
    each sample as ``leakage_report``), by ``BenchmarkBuilder._build_split``
    (``rejected``), or by the caller's own already-computed ``find_near_duplicates``
    call (``dup_pairs`` -- this function does not re-run the matcher, per
    B1.3's "do not change the matcher"). It only reads that evidence and labels
    it, instead of letting it print to a terminal and disappear (the defect
    this item exists to fix; CLAUDE.md sec 2.5/sec 11.49).

    ``split_of`` maps each kept sample's ``sample_id`` to ``"public"`` or
    ``"private"``, which is what makes the near-duplicate pairs' ``relation``
    field meaningful -- the cross-split pairs are the only leakage-relevant
    ones (they are the ones that would break sec 4.5's disjoint-seed
    guarantee), and the un-labelled matcher output cannot tell them apart from
    an ordinary within-split near-duplicate.
    """
    references = sorted({corpus for corpus, _, _ in auditor._refs})
    reference_n_series = len(auditor._refs)
    gate_effective = reference_n_series > 0

    accepted_distances = [
        s.leakage_report["nearest_distance"] for s in kept
        if s.leakage_report is not None and s.leakage_report.get("nearest_distance") is not None
    ]
    rejected_distances = [d for _, d in rejected]
    n_candidates = len(kept) + len(rejected)

    gate = {
        "references": references if references else "none",
        "reference_licenses": dict(auditor._licenses),
        "reference_n_series": reference_n_series,
        "gate_effective": gate_effective,
        "metric": auditor.metric,
        "band": auditor.window,
        "normalization": f"resample{auditor.target_len}+znorm",
        "threshold": auditor.threshold,
        "n_candidates": n_candidates,
        "n_rejected": len(rejected),
        "rejection_rate": (len(rejected) / n_candidates) if n_candidates else 0.0,
        "accepted_distance_quantiles": _quantiles(accepted_distances) if gate_effective else None,
        "rejected_distance_quantiles": _quantiles(rejected_distances) if gate_effective else None,
    }

    n_within_public = n_within_private = n_across = 0
    labeled: list[dict[str, Any]] = []
    for a, b, d in dup_pairs:
        sa, sb = split_of.get(a, "?"), split_of.get(b, "?")
        if sa == "public" and sb == "public":
            relation = "within_public"
            n_within_public += 1
        elif sa == "private" and sb == "private":
            relation = "within_private"
            n_within_private += 1
        else:
            relation = "across"
            n_across += 1
        labeled.append({"a": a, "b": b, "distance": d, "relation": relation})
    labeled.sort(key=lambda r: r["distance"])

    near_duplicates = {
        "threshold": near_dup_threshold,
        "target_len": near_dup_target_len,
        "n_pairs_within_public": n_within_public,
        "n_pairs_within_private": n_within_private,
        "n_pairs_across_splits": n_across,
        "worst_pairs_cap": worst_k,
        "worst_pairs": labeled[:worst_k],
    }

    block: dict[str, Any] = {"schema_version": 1, "gate": gate, "near_duplicates": near_duplicates}

    if compute_realism and gate_effective and auditor._raw_refs and kept:
        by_tier: dict[str, list[TimeSeriesSample]] = {}
        for s in kept:
            tier = s.provenance.generator_params.get("tier", "unknown")
            by_tier.setdefault(tier, []).append(s)
        realism = {
            "note": "gap between generated and real reference distributions on "
                    "summary statistics; used only to tune realism, NEVER a "
                    "leakage signal (CLAUDE.md sec 4.4) -- render under 'how "
                    "realistic', never under 'how leak-free'.",
            "combined": realism_report(kept, auditor._raw_refs),
            "by_tier": {tier: realism_report(s, auditor._raw_refs) for tier, s in by_tier.items()},
        }
        block["realism"] = realism

    return block


def realism_report(samples: list[TimeSeriesSample], reference: list[np.ndarray]) -> dict[str, Any]:
    """Compare synthetic and real distributions on simple summary statistics.

    Reports the gap in mean, std, lag-1 autocorrelation, and spectral centroid
    between the generated set and a real target. Used to tune the realism tier,
    never as a leakage signal.
    """

    def feats(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        ac1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else 0.0
        spec = np.abs(np.fft.rfft(arr - arr.mean()))
        freqs = np.fft.rfftfreq(len(arr))
        centroid = float((freqs * spec).sum() / (spec.sum() + 1e-8))
        return np.array([arr.mean(), arr.std(), ac1, centroid])

    gen = np.stack([feats(s.values) for s in samples]).mean(0)
    ref = np.stack([feats(a) for a in reference]).mean(0)
    keys = ["mean", "std", "autocorr_lag1", "spectral_centroid"]
    return {k: {"generated": float(g), "reference": float(r), "gap": float(abs(g - r))} for k, g, r in zip(keys, gen, ref)}
