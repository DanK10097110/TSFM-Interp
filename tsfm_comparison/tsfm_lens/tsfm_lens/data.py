"""Benchmark loading and the context/target split every stage consumes.

Primary source is a sealed corpus produced by the companion `tsfm_benchmark`
package (verified via `load_sealed` when importable, raw `corpus.jsonl`
parsing otherwise). A small self-contained smoke generator exists so the
whole pipeline can be exercised without any external data or checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import log

_FAMILY_CANDIDATES = ("task", "task_name", "spec", "spec_name", "family")


@dataclass
class BenchmarkData:
    """In-memory benchmark: fixed-length series plus per-series metadata."""

    values: np.ndarray
    meta: pd.DataFrame
    context_len: int
    horizon: int

    @property
    def n(self) -> int:
        return self.values.shape[0]

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
    if cfg.source == "smoke":
        rows = _smoke_rows(cfg, seed)
    elif cfg.source in ("sealed", "jsonl"):
        rows = _load_corpus_rows(Path(cfg.path), verify=cfg.source == "sealed")
    else:
        raise ValueError(f"unknown data.source '{cfg.source}'")
    return _assemble(rows, cfg)


def _assemble(rows: list, cfg: DataConfig) -> BenchmarkData:
    """Filter to usable lengths, crop to context+horizon, and build metadata."""
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
    if cfg.max_series is not None:
        kept = kept[: cfg.max_series]
    values = np.stack([v for v, _ in kept])
    meta = pd.DataFrame({
        "series_id": [r.get("sample_id", f"s{i}") for i, (_, r) in enumerate(kept)],
        "family": [_family_of(r, cfg.family_key) for _, r in kept],
        "tier": [r.get("tier", "unknown") for _, r in kept],
    })
    log.info("data: %d series, %d families, len=%d (%d context + %d horizon)",
             len(values), meta["family"].nunique(), need, cfg.context_len, cfg.horizon)
    return BenchmarkData(values, meta, cfg.context_len, cfg.horizon)


def _family_of(row: dict, family_key: str) -> str:
    """Resolve the family label for one sample, honoring an explicit key path."""
    if family_key != "auto":
        cur: object = row
        for part in family_key.split("."):
            cur = cur.get(part, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, str) and cur:
            return cur
    for key in _FAMILY_CANDIDATES:
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    prov = row.get("provenance") or {}
    return str(prov.get("generator", "unknown"))


def _load_corpus_rows(path: Path, verify: bool) -> list:
    """Load samples from a tsfm_benchmark directory, verified when the package is available."""
    if verify:
        try:
            from tsfm_benchmark import load_sealed
            samples, _ = load_sealed(str(path), verify=True)
            log.info("data: sealed corpus verified via tsfm_benchmark (%d samples)", len(samples))
            return [_sample_to_row(s) for s in samples]
        except ImportError:
            log.warning("tsfm_benchmark not installed; reading corpus.jsonl without seal verification")
        except Exception as e:
            raise RuntimeError(f"sealed corpus failed verification: {e}") from e
    jsonl = path / "corpus.jsonl" if path.is_dir() else path
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    log.info("data: loaded %d rows from %s", len(rows), jsonl)
    return rows


def _sample_to_row(sample: object) -> dict:
    """Normalize a tsfm_benchmark sample object into the plain-dict row schema."""
    row = {"values": np.asarray(sample.values, dtype=np.float32)}
    for key in ("sample_id", "tier", *_FAMILY_CANDIDATES):
        if hasattr(sample, key):
            row[key] = getattr(sample, key)
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
