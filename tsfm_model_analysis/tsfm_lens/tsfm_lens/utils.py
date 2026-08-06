"""Shared utilities for seeding, device handling, and small artifact IO."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import logging
import platform
import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("tsfm_lens")

_PROVENANCE_PACKAGES = ("torch", "numpy", "pandas", "zarr", "sklearn",
                       "transformers", "timesfm", "chronos")


def setup_logging(level: str = "INFO") -> None:
    """Configure a single stream handler for the pipeline logger."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(level)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to cpu with a warning."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map a config dtype string to a torch dtype, degrading safely on cpu."""
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dt = table.get(name, torch.float32)
    if device.type == "cpu" and dt is torch.float16:
        return torch.bfloat16
    return dt


def _git(args: list) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _pkg_version(module_name: str) -> Optional[str]:
    """A package's version, preferring its own `__version__` but falling
    back to installed-distribution metadata for packages that expose
    neither (confirmed live: `timesfm` has no `__version__` attribute at
    all, `ROADMAP.md` sec 15 A7)."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(mod, "__version__", None)
    if version is not None:
        return version
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version(module_name)
    except Exception:
        return None


def _hf_revision(repo_id: str) -> Optional[str]:
    """The locally cached commit hash for a checkpoint, if `huggingface_hub` can find it.

    Deliberately reads the *local cache*, not the Hub API -- this reflects
    the exact snapshot a run actually used (works offline, matches what was
    really loaded) rather than "whatever the latest revision on the Hub is
    right now," which could silently disagree with the cached weights.
    """
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified)
                return revisions[-1].commit_hash if revisions else None
    except Exception:
        return None
    return None


def _nvidia_driver_version() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def run_provenance(cfg=None, corpus_digest: Optional[str] = None) -> dict:
    """Environment/version/git/checkpoint snapshot for a run's manifest (`ROADMAP.md` sec 15 A7).

    Best-effort throughout: a package that isn't installed (e.g. no
    TimesFM/Chronos deps in a mock-only environment) or a git/nvidia-smi
    call that fails (not a git repo, no GPU, no network) is recorded as
    `None` rather than raising -- this is diagnostic metadata, not a
    correctness dependency (`CLAUDE.md` sec 2.5's "unsupported capability"
    branch, applied to environment introspection). No hostnames or absolute
    user paths (invariant 11): only versions, hashes, and device *names*.
    """
    device = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        device["device_names"] = [torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count())]
        device["cuda_version"] = torch.version.cuda
        device["driver_version"] = _nvidia_driver_version()

    models = None
    config_hash = None
    if cfg is not None:
        models = [{"name": m.name, "adapter": m.adapter, "checkpoint": m.checkpoint,
                  "hf_revision": _hf_revision(m.checkpoint) if m.checkpoint else None}
                 for m in cfg.models]
        from .manifest import _stable_json
        config_hash = hashlib.sha256(
            _stable_json(dataclasses.asdict(cfg)).encode("utf-8")).hexdigest()[:16]

    from . import __version__ as _tsfm_lens_version
    return {
        "tsfm_lens_version": _tsfm_lens_version,
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "python_version": platform.python_version(),
        "packages": {pkg: _pkg_version(pkg) for pkg in _PROVENANCE_PACKAGES},
        "device": device,
        "models": models,
        "corpus_digest": corpus_digest,
        "config_hash": config_hash,
    }


def save_json(path: Path, obj: object) -> None:
    """Write an object as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: object) -> object:
    """Serialize numpy scalars and arrays inside JSON artifacts."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def relative_depths(n_layers: int) -> np.ndarray:
    """Relative depth in [0, 1] for each layer index, used to compare models of different depth."""
    if n_layers == 1:
        return np.array([0.5])
    return np.arange(n_layers) / (n_layers - 1)


def batch_slices(n: int, batch_size: int):
    """Yield (start, end) index pairs covering range(n) in batches."""
    for s in range(0, n, batch_size):
        yield s, min(s + batch_size, n)


def _stratified_alloc(counts: np.ndarray, k: int) -> np.ndarray:
    """Per-stratum sample counts summing to `min(k, sum(counts))`.

    Proportional allocation with a floor of 1 per nonempty stratum (when
    `k` is at least the number of strata), remainder assigned by largest
    fractional remainder, never exceeding a stratum's own count. Used by
    `sample_rows` so a capped sample's family composition doesn't drift
    with the cap size (`ROADMAP.md` sec 15 A4).
    """
    counts = np.asarray(counts, dtype=np.int64)
    n_strata, n = len(counts), int(counts.sum())
    k = min(k, n)
    if n_strata == 0 or k == 0:
        return np.zeros(n_strata, dtype=np.int64)
    if k <= n_strata:
        alloc = np.zeros(n_strata, dtype=np.int64)
        order = np.argsort(-counts)
        alloc[order[:k]] = 1
        return alloc
    raw = counts * (k / n)
    alloc = np.minimum(np.maximum(1, np.floor(raw).astype(np.int64)), counts)
    frac = raw - np.floor(raw)
    diff = k - int(alloc.sum())
    if diff > 0:
        order = np.argsort(-frac)
        i = 0
        while diff > 0 and i < 1000 * n_strata:
            j = order[i % n_strata]
            if alloc[j] < counts[j]:
                alloc[j] += 1
                diff -= 1
            i += 1
    elif diff < 0:
        order = np.argsort(frac)
        i = 0
        while diff < 0 and i < 1000 * n_strata:
            j = order[i % n_strata]
            if alloc[j] > 1:
                alloc[j] -= 1
                diff += 1
            i += 1
    return alloc


def sample_rows(n: int, k: int, seed: int, strata: Optional[np.ndarray] = None,
                composition_tolerance: float = 0.15) -> np.ndarray:
    """Deterministic row sample of size `min(k, n)`, always sorted.

    Plain `rng.choice(..., replace=False)` when `strata` is `None` -- the
    pattern already used at every already-random call site in this repo.
    Stratified-without-replacement when `strata` (one label per row, length
    `n`) is given, so a capped sample's family composition stays close to
    the population's regardless of `k`, instead of silently depending on
    row order (`ROADMAP.md` sec 15 A4 -- head-slice caps like `kept[:n]`
    select a family-skewed prefix from a corpus written grouped by task).
    Deterministic in `(n, k, seed, strata)`: the same call always returns
    the same rows. Logs the realized family composition, and warns if any
    stratum's realized share deviates from its population share by more
    than `composition_tolerance` (only possible when `k < n_strata`, the
    one case stratification itself can't fully fix).
    """
    k = min(k, n)
    rng = np.random.default_rng(seed)
    if strata is None:
        return np.arange(n) if k >= n else np.sort(rng.choice(n, size=k, replace=False))
    strata = np.asarray(strata)
    labels, inv = np.unique(strata, return_inverse=True)
    counts = np.bincount(inv, minlength=len(labels))
    alloc = _stratified_alloc(counts, k)
    chosen = []
    for i in range(len(labels)):
        pool = np.where(inv == i)[0]
        take = min(int(alloc[i]), len(pool))
        if take > 0:
            chosen.append(pool[rng.choice(len(pool), size=take, replace=False)])
    result = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=np.int64)
    pop_share = counts / max(1, n)
    real_share = alloc / max(1, alloc.sum())
    log.info("sample_rows: n=%d k=%d realized composition %s (population %s)",
             n, len(result),
             {str(l): int(c) for l, c in zip(labels, alloc)},
             {str(l): int(c) for l, c in zip(labels, counts)})
    skew = np.abs(pop_share - real_share)
    if np.any(skew > composition_tolerance):
        skewed = {str(labels[i]): (float(pop_share[i]), float(real_share[i]))
                  for i in np.where(skew > composition_tolerance)[0]}
        log.warning("sample_rows: realized composition deviates from the population by "
                   "more than %.0f%% for %s (population_share, realized_share) -- likely "
                   "because k=%d is smaller than the number of strata (%d); consider a "
                   "larger sample if this stage's per-family numbers matter",
                   composition_tolerance * 100, skewed, k, len(labels))
    return result


def capped_take(requested: int, **limits: int) -> dict:
    """Resolve a configured sample-size request against one or more hard
    limits, warning and naming which limit(s) actually bound the result
    (`ROADMAP.md` sec 15 A16).

    Several stages must fit one forward pass, so they silently clamp their
    configured cap to `adapter.cfg.batch_size` (`lens.py`, `l3_perturbation.
    py`, `sae/eval.py`) with no warning and no record -- meaning the
    *realized* sample size can differ from the configured one, and differ
    per model (batch sizes differ), making a cross-run or cross-model
    comparison of that stage's numbers silently untrustworthy unless
    someone manually checks every batch_size matched (`ROADMAP.md` §5.3
    already hit this once). Call as e.g. `capped_take(cfg.lens.max_series,
    n_available=data.n, batch_size=adapter.cfg.batch_size)`; fold the
    returned dict into the stage's own artifact meta (the same
    `n_requested`/`n_realized` fields A4 already established, now with
    `limited_by` naming *why* they differ when they do).
    """
    take = requested
    for limit in limits.values():
        take = min(take, limit)
    limited_by = sorted(name for name, limit in limits.items()
                        if limit == take and limit < requested)
    if limited_by:
        log.warning("sample cap reduced: requested %d, realized %d (limited by %s)",
                   requested, take, ", ".join(limited_by))
    return {"n_requested": int(requested), "n_realized": int(take),
           "limited_by": limited_by or None}
