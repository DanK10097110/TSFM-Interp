"""Benchmark loading and the context/target split every stage consumes.

Primary source is a sealed corpus produced by the companion `tsfm_benchmark`
package (verified via `load_sealed` when importable, raw `corpus.jsonl`
parsing otherwise). A small self-contained smoke generator exists so the
whole pipeline can be exercised without any external data or checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import log, sample_rows

_FAMILY_CANDIDATES = ("task", "task_name", "spec", "spec_name", "family")
_MIN_RESOLUTION_RATE = 0.9


@dataclass
class BenchmarkData:
    """In-memory benchmark: fixed-length series plus per-series metadata."""

    values: np.ndarray
    meta: pd.DataFrame
    context_len: int
    horizon: int
    # Populated by `_assemble` (`ROADMAP.md` sec 15 A6): which family_key was
    # requested/used, how many rows resolved, and the realized composition --
    # so a typo'd key or an unlabeled corpus is a recorded fact, not just an
    # INFO-level family count nobody checked.
    family_resolution: dict = field(default_factory=dict)
    # The sealed corpus's global sha256 digest (`ROADMAP.md` sec 15 A7) --
    # `load_sealed` already computes and verifies this; it was previously
    # discarded here (`samples, _ = load_sealed(...)`) rather than carried
    # into run provenance, so two runs against corpora that silently differ
    # (a regenerated private epoch, an edited config) had no recorded way to
    # tell apart. `None` for the smoke generator or an unverified jsonl load
    # (no manifest exists to digest).
    corpus_digest: Optional[str] = None

    @property
    def n(self) -> int:
        return self.values.shape[0]

    @property
    def n_families(self) -> int:
        return int(self.meta["family"].nunique())

    @property
    def families(self) -> np.ndarray:
        return self.meta["family"].to_numpy()

    def contexts(self) -> np.ndarray:
        """Model input windows, shape [N, context_len]."""
        return self.values[:, : self.context_len]

    def targets(self) -> np.ndarray:
        """Forecast ground truth, shape [N, horizon]."""
        return self.values[:, self.context_len : self.context_len + self.horizon]


def load_benchmark(cfg: DataConfig, seed: int = 0) -> BenchmarkData:
    """Load the configured benchmark source into a BenchmarkData bundle."""
    digest = None
    if cfg.source == "smoke":
        rows = _smoke_rows(cfg, seed)
    elif cfg.source in ("sealed", "jsonl"):
        rows, digest = _load_corpus_rows(Path(cfg.path), verify=cfg.source == "sealed")
    else:
        raise ValueError(f"unknown data.source '{cfg.source}'")
    return _assemble(rows, cfg, seed, corpus_digest=digest)


def _assemble(rows: list, cfg: DataConfig, seed: int = 0,
             corpus_digest: Optional[str] = None) -> BenchmarkData:
    """Filter to usable lengths, crop to context+horizon, and build metadata.

    `tsfm_benchmark` writes corpora grouped by task/generator, so a plain
    `kept[:max_series]` head slice silently selects a family-skewed prefix
    instead of a representative sample (`ROADMAP.md` sec 15 A4) -- the cap is
    applied via a stratified `sample_rows` instead, computed *before* the cap
    so family membership can be used as strata.
    """
    need = cfg.context_len + cfg.horizon
    kept, skipped = [], 0
    for r in rows:
        v = np.asarray(r["values"], dtype=np.float32)
        if v.ndim != 1 or len(v) < need or not np.all(np.isfinite(v)):
            skipped += 1
            continue
        kept.append((v[:need], r))
    if skipped:
        log.warning("data: skipped %d series shorter than context+horizon or non-finite", skipped)
    if not kept:
        raise ValueError("no usable series after length filtering")

    resolution_rate = _check_family_key_resolution(kept, cfg.family_key)
    families_all = np.array([_family_of(r, cfg.family_key) for _, r in kept])
    archetypes_all = np.array([_archetype_of(r) for _, r in kept], dtype=object)
    generators_all = np.array([_generator_of(r) for _, r in kept], dtype=object)
    roles_all = (np.array([r.get("role") or "unknown" for _, r in kept], dtype=object)
                 if any(r.get("role") for _, r in kept) else None)
    if cfg.max_series is not None and cfg.max_series < len(kept):
        idx = sample_rows(len(kept), cfg.max_series, seed, strata=families_all)
        kept = [kept[i] for i in idx]
        families_all = families_all[idx]
        archetypes_all = archetypes_all[idx]
        generators_all = generators_all[idx]
        if roles_all is not None:
            roles_all = roles_all[idx]
    values = np.stack([v for v, _ in kept])
    meta = pd.DataFrame({
        "series_id": [r.get("sample_id", f"s{i}") for i, (_, r) in enumerate(kept)],
        "family": families_all.tolist(),
        "tier": [_tier_of(r) for _, r in kept],
        # `archetype` (ROADMAP.md sec 15 A9): random_parametric's per-sample
        # sampled archetype, `None` for tiers whose generator can't express
        # one (real-derived, smoke) -- a fallback, not a crash, since most of
        # this pipeline's own corpora predate this column. `generator` is the
        # raw provenance generator name, finer than `family` when family_key
        # groups several generators under one task label.
        "archetype": archetypes_all.tolist(),
        "generator": generators_all.tolist(),
    })
    if roles_all is not None:
        meta["role"] = roles_all.tolist()
    n_families = int(meta["family"].nunique())
    log.info("data: %d series, %d families, len=%d (%d context + %d horizon)",
             len(values), n_families, need, cfg.context_len, cfg.horizon)
    family_resolution = {
        "key_requested": cfg.family_key,
        "key_used": cfg.family_key,
        "resolution_rate": resolution_rate,
        "n_unknown": int((families_all == "unknown").sum()),
        "n_families": n_families,
        "composition": {str(k): int(v) for k, v in
                        meta["family"].value_counts().sort_index().items()},
    }
    return BenchmarkData(values, meta, cfg.context_len, cfg.horizon, family_resolution,
                        corpus_digest)


def _resolve_family_key(row: dict, family_key: str) -> Optional[str]:
    """Resolve one explicit dotted key against a row, with no fallback chain.

    Distinct from `_family_of`'s "auto" behavior on purpose (`ROADMAP.md` sec
    15 A6): an explicit key that doesn't resolve for a row must be counted as
    a miss here so the *aggregate* resolution rate is meaningful, rather than
    silently blending in whatever the fallback chain would have produced.
    """
    cur: object = row
    for part in family_key.split("."):
        cur = cur.get(part, {}) if isinstance(cur, dict) else {}
    return cur if isinstance(cur, str) and cur else None


def _check_family_key_resolution(kept: list, family_key: str) -> Optional[float]:
    """Raise if an explicit `family_key` resolves for too few rows to trust (sec 15 A6).

    A typo'd or wrong key would otherwise fall through `_family_of`'s
    candidate/provenance chain per row, silently regrouping every per-family
    statistic downstream (L0's Holm-corrected strengths, L1's per-family CKA,
    attention's family-conditioned profiles, clustering labels/AMI) under a
    grouping the user never asked for -- exactly the "silently wrong slice"
    `CLAUDE.md` sec 2.5 forbids. `family_key: auto` has no such failure mode
    (it's already the fallback chain) so this is a no-op for it.
    """
    if family_key == "auto" or not kept:
        return None
    hits = np.array([_resolve_family_key(r, family_key) is not None for _, r in kept])
    rate = float(hits.mean())
    if rate >= _MIN_RESOLUTION_RATE:
        return rate
    candidate_rates = {k: float(np.mean([_resolve_family_key(r, k) is not None for _, r in kept]))
                       for k in _FAMILY_CANDIDATES if k != family_key}
    working = sorted((k for k, r in candidate_rates.items() if r >= _MIN_RESOLUTION_RATE),
                     key=candidate_rates.get, reverse=True)
    suggestion = (f"keys that resolve for >={_MIN_RESOLUTION_RATE:.0%} of rows instead: {working}"
                 if working else
                 f"no candidate key in {_FAMILY_CANDIDATES} resolves better either "
                 f"(rates: {candidate_rates}); pass family_key: auto to use the "
                 f"built-in task/spec/provenance fallback chain")
    raise ValueError(
        f"data.family_key={family_key!r} resolved for only {rate:.0%} of {len(kept)} rows "
        f"(need >={_MIN_RESOLUTION_RATE:.0%}) -- refusing to silently regroup every "
        f"per-family statistic downstream under whatever the fallback chain happens to "
        f"produce for the rest (ROADMAP.md sec 15 A6, CLAUDE.md sec 2.5). {suggestion}.")


def _family_of(row: dict, family_key: str) -> str:
    """Resolve the family label for one sample, honoring an explicit key path."""
    if family_key != "auto":
        cur = _resolve_family_key(row, family_key)
        if cur is not None:
            return cur
    for key in _FAMILY_CANDIDATES:
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    prov = row.get("provenance") or {}
    return str(prov.get("generator", "unknown"))


def _tier_of(row: dict) -> str:
    """`tier` is a top-level key on smoke rows but lives at
    `provenance.generator_params.tier` on real `TimeSeriesSample`s
    (`build_pipeline/builder.py` writes it there, not as a `sample.tier`
    attribute -- `TimeSeriesSample` has no such field). A bare
    `row.get("tier", "unknown")` therefore silently mis-tagged every real
    sealed-corpus series as `"unknown"` (`ROADMAP.md` sec 15 A10, found
    while fixing that item's ground-truth tier column -- confirmed live
    against `benchmark_medium/public_dev`, not assumed).
    """
    if isinstance(row.get("tier"), str) and row["tier"]:
        return row["tier"]
    params = (row.get("provenance") or {}).get("generator_params") or {}
    val = params.get("tier")
    return val if isinstance(val, str) and val else "unknown"


def _archetype_of(row: dict) -> Optional[str]:
    """`random_parametric`'s sampled archetype, or `None` where the tier can't express one."""
    params = (row.get("provenance") or {}).get("generator_params") or {}
    val = params.get("archetype")
    return val if isinstance(val, str) and val else None


def _generator_of(row: dict) -> str:
    """Raw provenance generator name (finer than `family` when `family_key` groups by task)."""
    return str((row.get("provenance") or {}).get("generator", "unknown"))


def _load_corpus_rows(path: Path, verify: bool) -> tuple:
    """Load samples from a tsfm_benchmark directory, verified when the package is available.

    Returns `(rows, corpus_digest)` -- the sealed manifest's global sha256
    digest was already computed by `load_sealed`'s own integrity check and
    previously discarded here (`ROADMAP.md` sec 15 A7); carrying it out lets
    run provenance record exactly which corpus contents a run actually used.
    """
    if verify:
        try:
            from tsfm_benchmark import load_sealed
            samples, manifest = load_sealed(str(path), verify=True)
            log.info("data: sealed corpus verified via tsfm_benchmark (%d samples)", len(samples))
            return [_sample_to_row(s) for s in samples], manifest.get("global_digest")
        except ImportError:
            log.warning("tsfm_benchmark not installed; reading corpus.jsonl without seal verification")
        except Exception as e:
            raise RuntimeError(f"sealed corpus failed verification: {e}") from e
    jsonl = path / "corpus.jsonl" if path.is_dir() else path
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    log.info("data: loaded %d rows from %s", len(rows), jsonl)
    return rows, None


def _sample_to_row(sample: object) -> dict:
    """Normalize a tsfm_benchmark sample object into the plain-dict row schema.

    ``role`` (ROADMAP sec 41: synthetic / real_derived / external_real) is copied only when the
    sample carries one, so rows from corpora built without roles are unchanged.
    """
    row = {"values": np.asarray(sample.values, dtype=np.float32)}
    for key in ("sample_id", "tier", *_FAMILY_CANDIDATES):
        if hasattr(sample, key):
            row[key] = getattr(sample, key)
    role = getattr(sample, "role", None)
    if role is not None:
        row["role"] = role
    prov = getattr(sample, "provenance", None)
    if prov is not None:
        row["provenance"] = {"generator": getattr(prov, "generator", "unknown"),
                             "generator_params": getattr(prov, "generator_params", {})}
    return row


def _smoke_rows(cfg: DataConfig, seed: int) -> list:
    """Generate a tiny labeled taxonomy so the pipeline runs with no external inputs."""
    rng = np.random.default_rng(seed)
    n, T = cfg.smoke_series_per_family, cfg.context_len + cfg.horizon
    t = np.arange(T, dtype=np.float32)
    rows = []

    def add(family: str, batch: np.ndarray) -> None:
        for i, v in enumerate(batch):
            rows.append({"values": v.astype(np.float32), "family": family,
                         "sample_id": f"{family}_{i}", "tier": "synthetic"})

    slope = rng.uniform(-0.05, 0.05, (n, 1))
    add("trend", slope * t + rng.normal(0, 0.3, (n, T)))
    p_lo = rng.uniform(T / 6, T / 3, (n, 1))
    add("seasonal_lf", np.sin(2 * np.pi * t / p_lo) * rng.uniform(1, 2, (n, 1)) + rng.normal(0, 0.2, (n, T)))
    p_hi = rng.uniform(4, 10, (n, 1))
    add("seasonal_hf", np.sin(2 * np.pi * t / p_hi) * rng.uniform(1, 2, (n, 1)) + rng.normal(0, 0.2, (n, T)))
    ar = np.zeros((n, T), dtype=np.float32)
    eps = rng.normal(0, 1, (n, T))
    for k in range(1, T):
        ar[:, k] = 0.85 * ar[:, k - 1] + eps[:, k]
    add("ar_noise", ar)
    shift_pos = rng.integers(T // 4, 3 * T // 4, n)
    base = rng.normal(0, 0.3, (n, T))
    for i in range(n):
        base[i, shift_pos[i]:] += rng.choice([-1, 1]) * rng.uniform(2, 4)
    add("level_shift", base)
    spiky = rng.normal(0, 0.3, (n, T))
    for i in range(n):
        idx = rng.choice(T, 4, replace=False)
        spiky[i, idx] += rng.choice([-1, 1], 4) * rng.uniform(4, 7, 4)
    add("spiky", spiky)
    return rows
