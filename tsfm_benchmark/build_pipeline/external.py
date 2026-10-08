"""The ``external_real`` slice: raw GIFT-Eval windows with no generator (ROADMAP sec 41, V3-A).

Every other sample in this benchmark is either generated (``synthetic``) or generated *from*
real data (``real_derived``). This module produces the third role: windows cut unchanged from
an external real-world benchmark, so a finding can be checked once on data that neither the
generators nor any dev-stage analysis ever touched. These windows are never z-scored or
otherwise transformed (models normalize themselves), carry no ground truth, and are sealed as
their own split so they cannot be mixed into dev or private by accident.

Source is Hugging Face ``Salesforce/GiftEval`` (Apache-2.0), whose repo stores each dataset as
an Arrow table of *whole* series (``target`` is ``[T]`` or, for multivariate sets, ``[n_var, T]``).
A window here is the final ``window_length`` points of a series (or variate), i.e. it lies inside
GIFT-Eval's own test region only when that region is at least as long; this is stated on the
corpus card rather than assumed. Selection is deterministic and never uses Python ``hash()``:
candidates are ranked by sha256 of ``seed:dataset:freq:item:variate:window``, and the number
drawn from each domain and each dataset inside it comes from a water-filling allocation, so a
thin dataset gives its unused share to the others instead of being silently over-drawn.

The leakage audit reuses ``LeakageAuditor`` unchanged, with the dev split (and optionally Monash)
as the reference set, and runs on every candidate window *before* the stratified draw (see
``ReferenceGate``) so the gate cannot silently re-weight domains. Datasets whose underlying data the dev real-derived sources already use
(electricity, ETT, solar, KDD Cup, M4 hourly) are simply not listed in the config, since a
dataset-level overlap would defeat the point of a held-out slice and the DTW gate only catches
near-copies of windows, not the same recording sampled elsewhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .audit import LeakageAuditor, _normalize, find_near_duplicates
from .schema import GroundTruth, Provenance, SourceRef, TimeSeriesSample
from .sources import _hash

DEFAULT_REPO = "Salesforce/GiftEval"
LICENSE = "apache-2.0"
GENERATOR = "gifteval_window"
TIER = "external_real"


def _rank_key(seed: int, *parts: object) -> str:
    return hashlib.sha256(f"{seed}:{':'.join(str(p) for p in parts)}".encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    """One eligible window: where it came from and its values."""

    dataset: str
    freq: str
    domain: str
    item_id: str
    variate: int
    window: int
    start: int
    values: np.ndarray


def default_opener(repo: str, dataset: str, freq: str | None) -> Any:
    """Open one GIFT-Eval dataset (and frequency) as a ``datasets.Dataset``.

    Downloads only that dataset's own Arrow file (not the repo) into the normal Hugging
    Face cache and memory-maps it, so the disk cost is one file per listed dataset.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    sub = f"{dataset}/{freq}" if freq else dataset
    path = hf_hub_download(repo, f"{sub}/data-00000-of-00001.arrow", repo_type="dataset")
    return Dataset.from_file(path)


def candidate_windows(rows: Iterable[dict[str, Any]], dataset: str, freq: str, domain: str,
                      window_length: int, max_windows: int) -> tuple[list[Candidate], dict[str, int]]:
    """Every non-overlapping tail window of every series/variate that is long enough and clean.

    Windows are cut backwards from the series end, so window 0 is the final ``window_length``
    points. A window with any non-finite value or no variation is dropped and counted, never
    repaired: ``counts`` returns how many series were too short and how many windows were
    non-finite or constant.
    """
    out: list[Candidate] = []
    counts = {"series": 0, "too_short": 0, "nonfinite": 0, "constant": 0}
    for row in rows:
        target = np.asarray(row["target"], dtype=float)
        variates = target[None, :] if target.ndim == 1 else target
        for v, series in enumerate(variates):
            counts["series"] += 1
            n_fit = len(series) // window_length
            if n_fit == 0:
                counts["too_short"] += 1
                continue
            for w in range(min(n_fit, max_windows)):
                end = len(series) - w * window_length
                vals = series[end - window_length:end]
                if not np.isfinite(vals).all():
                    counts["nonfinite"] += 1
                    continue
                if np.ptp(vals) < 1e-9:
                    counts["constant"] += 1
                    continue
                out.append(Candidate(dataset, freq, domain, str(row.get("item_id", "")), v, w,
                                     end - window_length, vals))
    return out, counts


def waterfill(capacity: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` as evenly as possible across keys without exceeding any capacity.

    Deterministic: leftover units after an even split go to keys in sorted order. If the
    capacities cannot cover ``total`` every key gets its full capacity.
    """
    alloc = {k: 0 for k in capacity}
    remaining = min(total, sum(capacity.values()))
    open_keys = sorted(k for k, c in capacity.items() if c > 0)
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        progressed = 0
        for i, k in enumerate(list(open_keys)):
            want = share + (1 if i < extra else 0)
            take = min(want, capacity[k] - alloc[k])
            alloc[k] += take
            progressed += take
        remaining -= progressed
        open_keys = [k for k in open_keys if alloc[k] < capacity[k]]
        if progressed == 0:
            break
    return alloc


def select_candidates(pools: dict[tuple[str, str], list[Candidate]], domains: dict[tuple[str, str], str],
                      n_total: int, seed: int) -> list[Candidate]:
    """Domain-stratified, sha256-ranked selection from per-dataset candidate pools.

    ``n_total`` is water-filled across domains, then each domain's share across its datasets,
    then each dataset's share is the first ``k`` of its candidates ordered by a sha256 rank
    of (seed, dataset, freq, item, variate, window). Same inputs always give the same windows.
    """
    by_domain: dict[str, dict[str, int]] = {}
    for key, cands in pools.items():
        by_domain.setdefault(domains[key], {})["/".join(key)] = len(cands)
    domain_alloc = waterfill({d: sum(c.values()) for d, c in by_domain.items()}, n_total)
    chosen: list[Candidate] = []
    for domain in sorted(by_domain):
        ds_alloc = waterfill(by_domain[domain], domain_alloc[domain])
        for name in sorted(ds_alloc):
            dataset, freq = name.split("/")
            cands = pools[(dataset, freq)]
            ranked = sorted(cands, key=lambda c: _rank_key(seed, c.dataset, c.freq, c.item_id, c.variate, c.window))
            chosen.extend(ranked[:ds_alloc[name]])
    return chosen


def to_sample(c: Candidate, window_length: int, seed: int, repo: str = DEFAULT_REPO) -> TimeSeriesSample:
    """Wrap one window as a role-tagged sample; the values are exactly the source's."""
    ref = SourceRef(corpus=f"hf/{repo}/{c.dataset}/{c.freq}", item_id=f"{c.item_id}:v{c.variate}:w{c.window}",
                    sha256=_hash(c.values), license=LICENSE)
    params = {"task_name": f"gifteval_{c.dataset}_{c.freq}", "tier": TIER, "epoch": 0, "dataset": c.dataset,
              "freq": c.freq, "domain": c.domain, "item_id": c.item_id, "variate": c.variate,
              "window": c.window, "window_start": c.start, "window_length": window_length}
    return TimeSeriesSample(
        values=c.values,
        ground_truth=GroundTruth(notes="raw external real window; no generative ground truth"),
        provenance=Provenance(generator=GENERATOR, generator_params=params,
                              seed=int(_rank_key(seed, c.dataset, c.freq, c.item_id, c.variate, c.window)[:8], 16),
                              source_refs=[ref]),
        role="external_real")


class ReferenceGate:
    """The unchanged ``LeakageAuditor`` gate applied to every external candidate before selection.

    Auditing the whole candidate pool, then stratifying over what passed, keeps the domain and
    dataset balance the config asks for; auditing the *selected* windows instead would let the
    gate silently re-weight domains (smooth series such as M4 daily/weekly sit within DTW 0.35 of
    some synthetic trend or Monash series far more often than noisy ones). The price is a
    selection effect that is recorded rather than hidden: ``rejected_by_dataset`` is the per-dataset
    count the gate removed, and the card should be read with that in mind.

    The gate tests for leakage, not for resemblance to our own generators. The DTW reference set
    is only the ``gate_roles`` dev samples (default ``real_derived``: the series that can contain
    the same recordings) plus whatever ``auditor`` already holds (the Monash leakage references).
    Synthetic dev series are excluded from the DTW gate because a smooth real window sitting
    near a synthetic trend is not a leak, and gating on them biased the slice toward non-smooth
    series (136 of 232 rejections in the first version). Against ALL reference samples,
    synthetic included, the gate still rejects an exact value-hash match and any near-duplicate
    (normalized distance below ``near_dup_threshold``).
    """

    def __init__(self, reference_samples: list[TimeSeriesSample], auditor: LeakageAuditor,
                 reference_corpus: str = "benchmark_v3/public_dev", near_dup_threshold: float = 0.05,
                 gate_roles: tuple[str, ...] = ("real_derived",)) -> None:
        self.gate_roles = tuple(gate_roles)
        self.n_dtw_references = 0
        for r in reference_samples:
            if r.role in self.gate_roles:
                auditor.add_reference(reference_corpus, r.sample_id, r.values)
                self.n_dtw_references += 1
        self.n_monash_references = len(auditor._refs) - self.n_dtw_references
        self._value_hashes = {_hash(r.values) for r in reference_samples}
        self._near_matrix = (np.stack([_normalize(r.values, 256) for r in reference_samples])
                             if reference_samples else None)
        self.n_exact_rejected = 0
        self.n_near_dup_rejected = 0
        self.auditor = auditor
        self.reference_samples = reference_samples
        self.near_dup_threshold = near_dup_threshold
        self.n_scored = 0
        self.rejected_distances: list[float] = []
        self.rejected_by_dataset: dict[str, int] = {}
        self.reports: dict[str, dict[str, Any]] = {}

    def __call__(self, sample: TimeSeriesSample) -> bool:
        report = self.auditor.audit(sample)
        self.n_scored += 1
        passed = bool(report.passed)
        reason_dist = float(report.nearest_distance)
        if passed and _hash(sample.values) in self._value_hashes:
            passed, reason_dist = False, 0.0
            self.n_exact_rejected += 1
        if passed and self._near_matrix is not None:
            q = _normalize(sample.values, 256)
            d = float(np.min(np.linalg.norm(self._near_matrix - q, axis=1)) / np.sqrt(256))
            if d < self.near_dup_threshold:
                passed, reason_dist = False, d
                self.n_near_dup_rejected += 1
        if not passed:
            ds = sample.provenance.generator_params.get("dataset", "unknown")
            self.rejected_by_dataset[ds] = self.rejected_by_dataset.get(ds, 0) + 1
            self.rejected_distances.append(reason_dist)
        else:
            self.reports[sample.sample_id] = sample.leakage_report
        return passed

    def block(self, kept: list[TimeSeriesSample]) -> dict[str, Any]:
        """Audit block for the manifest, including the near-duplicate pass over the selected windows.

        ``find_near_duplicates`` runs over the selected windows plus the reference split and only
        pairs with one window on each side are kept (a within-reference pair is not this slice's
        concern, and within-external pairs are reported separately).
        """
        ext_ids = {s.sample_id for s in kept}
        ref_ids = {r.sample_id for r in self.reference_samples}
        pairs = find_near_duplicates(kept + self.reference_samples, threshold=self.near_dup_threshold)
        cross = [(a, b, d) for a, b, d in pairs if (a in ext_ids and b in ref_ids) or (b in ext_ids and a in ref_ids)]
        within = [(a, b, d) for a, b, d in pairs if a in ext_ids and b in ext_ids]
        dists = [s.leakage_report["nearest_distance"] for s in kept if s.leakage_report]
        return {
            "references": sorted({c for c, _, _ in self.auditor._refs}), "reference_n_series": len(self.auditor._refs),
            "dtw_reference_sets": {"dev_roles": list(self.gate_roles), "n_dev_series": self.n_dtw_references,
                                   "n_leakage_reference_series": self.n_monash_references},
            "exact_and_near_duplicate_reference_set": {"scope": "all dev samples, every role",
                                                       "n_series": len(self.reference_samples)},
            "n_rejected_exact_hash": self.n_exact_rejected, "n_rejected_near_duplicate": self.n_near_dup_rejected,
            "gate_effective": len(self.auditor._refs) > 0, "metric": self.auditor.metric,
            "threshold": self.auditor.threshold,
            "stage": "every candidate window audited before stratified selection",
            "n_candidates_scored": self.n_scored, "n_rejected": len(self.rejected_distances),
            "rejected_by_dataset": dict(sorted(self.rejected_by_dataset.items())),
            "min_accepted_nearest_distance": float(min(dists)) if dists else None,
            "near_duplicate_threshold": self.near_dup_threshold,
            "n_near_duplicate_pairs_vs_reference": len(cross),
            "near_duplicate_pairs_vs_reference": [{"a": a, "b": b, "distance": d} for a, b, d in cross[:50]],
            "n_near_duplicate_pairs_within_external": len(within),
        }


def build_external_samples(cfg: dict[str, Any], opener: Callable[[str, str, str | None], Any] | None = None,
                           gate: Callable[[TimeSeriesSample], bool] | None = None
                           ) -> tuple[list[TimeSeriesSample], dict[str, Any]]:
    """Select the external windows named by an ``external_real`` config section.

    ``cfg`` keys: ``repo``, ``seed``, ``n_total``, ``window_length`` and ``datasets``, a list of
    ``{name, freq (optional), domain, max_windows_per_series (default 1)}``. ``opener`` lets a
    test substitute a local table for the Hugging Face download. ``gate``, when given, is applied
    to every candidate window (wrapped as a sample) before selection and only passing windows
    enter the pools. Returns the samples plus a ``selection`` record (per-dataset candidate
    counts, drop reasons, gate rejections, selected counts) for the manifest.
    """
    opener = opener or default_opener
    repo = cfg.get("repo", DEFAULT_REPO)
    seed = int(cfg["seed"])
    window_length = int(cfg["window_length"])
    pools: dict[tuple[str, str], list[Candidate]] = {}
    domains: dict[tuple[str, str], str] = {}
    per_dataset: dict[str, Any] = {}
    for entry in cfg["datasets"]:
        name, freq = entry["name"], entry.get("freq") or ""
        key = (name, freq)
        if key in pools:
            raise ValueError(f"external_real dataset {name}/{freq} listed twice")
        ds = opener(repo, name, entry.get("freq"))
        cands, counts = candidate_windows(ds, name, freq, entry["domain"], window_length,
                                          int(entry.get("max_windows_per_series", 1)))
        record = {"domain": entry["domain"], "windows_before_gate": len(cands), **counts}
        if gate is not None:
            cands = [c for c in cands if gate(to_sample(c, window_length, seed, repo))]
            record["gate_rejected"] = record["windows_before_gate"] - len(cands)
        record["candidates"] = len(cands)
        pools[key], domains[key] = cands, entry["domain"]
        per_dataset[f"{name}/{freq}" if freq else name] = record
    chosen = select_candidates(pools, domains, int(cfg["n_total"]), seed)
    samples = [to_sample(c, window_length, seed, repo) for c in chosen]
    reports = getattr(gate, "reports", {})
    for s in samples:
        s.leakage_report = reports.get(s.sample_id)
        key = (s.provenance.generator_params["dataset"], s.provenance.generator_params["freq"])
        label = "/".join(key) if key[1] else key[0]
        per_dataset[label]["selected"] = per_dataset[label].get("selected", 0) + 1
    selection = {"repo": repo, "seed": seed, "n_requested": int(cfg["n_total"]), "n_selected": len(samples),
                 "window_length": window_length, "per_dataset": per_dataset}
    return samples, selection
