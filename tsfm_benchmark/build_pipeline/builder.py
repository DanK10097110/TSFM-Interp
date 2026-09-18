"""Assembly layer.

Turns a declarative config into a benchmark while enforcing the invariants that
make the result trustworthy: nothing enters either split without passing the
leakage audit, the public dev split and the held-out private test split draw
from disjoint seed ranges so they share a distribution but never an instance,
and the private split is sealed (see ``seal``) as a stable held-out corpus.

Contamination over time is handled by epochs. Each epoch uses a fresh,
non-overlapping seed range, so ``regenerate_private`` produces a brand-new
held-out corpus when an older one is suspected of having leaked into training
data, LiveBench-style, while every sealed epoch stays reproducible from its
(private) seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .audit import LeakageAuditor, compose_audit_block, find_near_duplicates
from .registry import CORRUPTIONS, GENERATORS
from .schema import TimeSeriesSample
from . import seal as seal_mod
from .sources import load_sources

_EPOCH_STRIDE = 100_000_000


def _resolve_param(value: Any, rng: np.random.Generator) -> Any:
    """Resolve a corruption param that may be a ``[low, high]`` range.

    A two-number list/tuple is sampled once per call (uniform, or integer when
    both bounds are ints); anything else passes through unchanged. This lets a
    fixed corruption list vary its strength per sample instead of applying the
    exact same magnitude to every series in a task.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        lo, hi = value
        if isinstance(lo, int) and isinstance(hi, int):
            return lo if hi <= lo else int(rng.integers(lo, hi + 1))
        return float(lo) if hi <= lo else float(rng.uniform(lo, hi))
    return value


def _sample_corruption_chain(pool: list[dict[str, Any]], n_range: tuple[int, int] | list[int] | None, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Draw a random subset (and order) of a corruption pool for one sample.

    Exists so a task's ``count`` repeats each see a different combination of
    corruptions, not the same fixed chain applied to every series.
    """
    lo, hi = n_range if n_range else (1, len(pool))
    hi = min(hi, len(pool))
    lo = min(lo, hi)
    k = lo if hi <= lo else int(rng.integers(lo, hi + 1))
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in rng.permutation(idx)]


def _sample_source_subset(pool: list[Any], size_range: tuple[int, int] | list[int], rng: np.random.Generator) -> list[Any]:
    """Draw a random subset of a loaded source pool for one sample.

    A real-derived task's whole source pool is typically loaded once (see
    ``BenchmarkBuilder._load_cached_sources``) and may span many domains when
    it comes from ``bootstrap_catalog``. Recombining a different random
    handful of that pool per sample -- rather than feeding every sample the
    same fixed pool -- means the ``count`` repeats span many different
    domain combinations instead of one fixed (if reweighted) blend.
    """
    lo, hi = size_range
    hi = min(hi, len(pool))
    lo = min(lo, hi)
    k = lo if hi <= lo else int(rng.integers(lo, hi + 1))
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("numpy", "scipy", "dtaidistance"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def _stable_offset(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % 9973


@dataclass
class TaskSpec:
    """One generator plus a corruption chain, repeated n times.

    ``corruptions`` is a fixed, ordered chain applied to every repeat (its
    param values may still be ``[low, high]`` ranges resolved per sample).
    ``corruption_pool`` + ``n_corruptions`` instead draw a random subset and
    order from a larger pool independently for each repeat; set it to vary
    which corruptions apply per sample, not just their magnitude. Only one of
    the two is used: ``corruption_pool`` takes precedence when set.

    ``source_sample_size``, for a real-derived generator, draws that many
    sources at random from the loaded pool for *each* repeat instead of
    handing every repeat the whole pool. Most useful when ``source_config``
    bootstraps across many domains (see ``sources.bootstrap_catalog``): each
    repeat then combines a different random handful of domains rather than
    the same fixed blend reweighted.
    """

    name: str
    generator: str
    count: int
    generator_params: dict[str, Any] = field(default_factory=dict)
    corruptions: list[dict[str, Any]] = field(default_factory=list)
    corruption_pool: list[dict[str, Any]] = field(default_factory=list)
    n_corruptions: tuple[int, int] | list[int] | None = None
    tier: str = "synthetic"
    source_config: dict[str, Any] = field(default_factory=dict)
    source_sample_size: tuple[int, int] | list[int] | None = None


@dataclass
class BuildResult:
    public_dev: list[TimeSeriesSample]
    private_test: list[TimeSeriesSample]
    epoch: int
    rejected: list[tuple[str, float]]
    duplicates: list[tuple[str, str, float]]
    audit: dict[str, Any] = field(default_factory=dict)


class BenchmarkBuilder:
    """Builds, audits, splits, and seals a benchmark from a list of task specs."""

    def __init__(self, auditor: LeakageAuditor | None = None, private_seed_offset: int = 1_000_000) -> None:
        self.auditor = auditor or LeakageAuditor()
        self.private_seed_offset = private_seed_offset
        self._source_pool_cache: dict[str, list[Any]] = {}

    def _load_cached_sources(self, source_config: dict[str, Any]) -> list[Any]:
        """Load a task's source pool once and reuse it across every repeat.

        ``_make_one`` runs once per repeat (and once per split), so calling
        ``load_sources`` directly from here would re-hit the network on every
        single generated sample -- fine for a handful of series, ruinous for
        a task with a count in the thousands. The pool itself only needs
        loading once per distinct config; per-sample variation instead comes
        from ``source_sample_size`` drawing a different subset of the cached
        pool for each repeat.
        """
        key = json.dumps(source_config, sort_keys=True, default=str)
        if key not in self._source_pool_cache:
            self._source_pool_cache[key] = load_sources(source_config)
        return self._source_pool_cache[key]

    def _make_one(self, spec: TaskSpec, seed: int, epoch: int) -> TimeSeriesSample:
        gen = GENERATORS.get(spec.generator)
        generator_params = dict(spec.generator_params)
        rng = np.random.default_rng(seed)

        if spec.generator in {"mixture", "block_bootstrap", "sequential_par"} and not generator_params.get("sources") and spec.source_config:
            pool = self._load_cached_sources(spec.source_config)
            sources = _sample_source_subset(pool, spec.source_sample_size, rng) if spec.source_sample_size and pool else pool
            if spec.generator == "block_bootstrap" and sources:
                generator_params["source"] = sources[0]
            elif spec.generator == "sequential_par" and sources:
                generator_params["training"] = sources
            else:
                generator_params["sources"] = sources

        sample = gen(seed=seed, **generator_params)

        chain = _sample_corruption_chain(spec.corruption_pool, spec.n_corruptions, rng) if spec.corruption_pool else spec.corruptions
        for step in chain:
            corrupt = CORRUPTIONS.get(step["op"])
            params = {k: _resolve_param(v, rng) for k, v in step.items() if k != "op"}
            sample = corrupt(sample, seed=seed, **params)

        sample.provenance.generator_params["epoch"] = epoch
        # Recorded so downstream analysis (e.g. benchmark_validation) can tell
        # apart two tasks that share a generator -- generator_params alone
        # won't distinguish e.g. two 'mixture' tasks bootstrapping different
        # Monash domain samples, but their task names differ.
        sample.provenance.generator_params["task_name"] = spec.name
        sample.provenance.generator_params["tier"] = spec.tier
        sample.provenance.library_versions = _library_versions()
        sample.sample_id = sample.content_hash()[:16]
        return sample

    def _build_split(self, specs: list[TaskSpec], base_seed: int, epoch: int) -> tuple[list[TimeSeriesSample], list[tuple[str, float]]]:
        kept: list[TimeSeriesSample] = []
        rejected: list[tuple[str, float]] = []
        for spec in specs:
            for k in range(spec.count):
                seed = base_seed + _stable_offset(spec.name) + k
                sample = self._make_one(spec, seed=seed, epoch=epoch)
                report = self.auditor.audit(sample)
                if report.passed:
                    kept.append(sample)
                else:
                    rejected.append((sample.sample_id, report.nearest_distance))
        return kept, rejected

    def _epoch_seeds(self, seed: int, epoch: int) -> tuple[int, int]:
        public_base = seed + epoch * _EPOCH_STRIDE
        return public_base, public_base + self.private_seed_offset

    def build(self, specs: list[TaskSpec], seed: int = 0, epoch: int = 0) -> BuildResult:
        """Produce a public dev split and a held-out private test split for an epoch.

        The two splits use disjoint seed ranges, both pass the leakage gate, and
        the union is checked for near-duplicates as a final cross-split guard.
        """
        public_base, private_base = self._epoch_seeds(seed, epoch)
        public, rej_pub = self._build_split(specs, public_base, epoch)
        private, rej_priv = self._build_split(specs, private_base, epoch)
        dups = find_near_duplicates(public + private)
        rejected = rej_pub + rej_priv
        split_of = {s.sample_id: "public" for s in public}
        split_of.update({s.sample_id: "private" for s in private})
        audit_block = compose_audit_block(self.auditor, public + private, rejected, dups, split_of)
        return BuildResult(public_dev=public, private_test=private, epoch=epoch, rejected=rejected, duplicates=dups, audit=audit_block)

    def build_and_seal(self, specs: list[TaskSpec], public_dir: str, private_dir: str, seed: int = 0, epoch: int = 0) -> BuildResult:
        """Build an epoch and seal both splits to disk; private_dir is held out.

        Both manifests carry the same ``extra.audit`` block (ROADMAP.md sec 34
        item B1): the gate and near-duplicate check both ran over the whole
        public+private union, so the verdict is a property of the epoch, not
        of one split. ``extra.audit`` is metadata only -- ``_global_digest``
        hashes sample content alone (seal.py), so adding it here cannot move
        either split's digest or fail an existing ``load_sealed(verify=True)``
        (T-B1.1 pins this).
        """
        result = self.build(specs, seed=seed, epoch=epoch)
        seal_mod.seal_corpus(result.public_dev, public_dir, epoch, "public", extra={"audit": result.audit})
        seal_mod.seal_corpus(
            result.private_test, private_dir, epoch, "private",
            extra={"audit": result.audit, "note": "held-out; do not publish"})
        return result

    def regenerate_private(self, specs: list[TaskSpec], private_dir: str, seed: int = 0, epoch: int = 1) -> list[TimeSeriesSample]:
        """Generate and seal a fresh held-out corpus for a new epoch.

        Use when a previously published or used corpus is suspected of having
        leaked into training data. The new epoch shares no seeds with prior ones.
        """
        _, private_base = self._epoch_seeds(seed, epoch)
        private, _ = self._build_split(specs, private_base, epoch)
        seal_mod.seal_corpus(private, private_dir, epoch, "private", extra={"note": "regenerated held-out corpus"})
        return private
