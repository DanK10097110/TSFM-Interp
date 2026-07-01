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
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .audit import LeakageAuditor, find_near_duplicates
from .registry import CORRUPTIONS, GENERATORS
from .schema import TimeSeriesSample
from . import seal as seal_mod
from .sources import load_sources

_EPOCH_STRIDE = 100_000_000


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
    """One generator plus an optional ordered corruption chain, repeated n times."""

    name: str
    generator: str
    count: int
    generator_params: dict[str, Any] = field(default_factory=dict)
    corruptions: list[dict[str, Any]] = field(default_factory=list)
    tier: str = "synthetic"
    source_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    public_dev: list[TimeSeriesSample]
    private_test: list[TimeSeriesSample]
    epoch: int
    rejected: list[tuple[str, float]]
    duplicates: list[tuple[str, str, float]]


class BenchmarkBuilder:
    """Builds, audits, splits, and seals a benchmark from a list of task specs."""

    def __init__(self, auditor: LeakageAuditor | None = None, private_seed_offset: int = 1_000_000) -> None:
        self.auditor = auditor or LeakageAuditor()
        self.private_seed_offset = private_seed_offset

    def _make_one(self, spec: TaskSpec, seed: int, epoch: int) -> TimeSeriesSample:
        gen = GENERATORS.get(spec.generator)
        generator_params = dict(spec.generator_params)

        if spec.generator in {"mixture", "block_bootstrap", "sequential_par"} and not generator_params.get("sources") and spec.source_config:
            sources = load_sources(spec.source_config)
            if spec.generator == "block_bootstrap" and sources:
                generator_params["source"] = sources[0]
            elif spec.generator == "sequential_par" and sources:
                generator_params["training"] = [values for _, values in sources]
            else:
                generator_params["sources"] = sources

        sample = gen(seed=seed, **generator_params)
        for step in spec.corruptions:
            corrupt = CORRUPTIONS.get(step["op"])
            params = {k: v for k, v in step.items() if k != "op"}
            sample = corrupt(sample, seed=seed, **params)
        sample.provenance.generator_params["epoch"] = epoch
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
        return BuildResult(public_dev=public, private_test=private, epoch=epoch, rejected=rej_pub + rej_priv, duplicates=dups)

    def build_and_seal(self, specs: list[TaskSpec], public_dir: str, private_dir: str, seed: int = 0, epoch: int = 0) -> BuildResult:
        """Build an epoch and seal both splits to disk; private_dir is held out."""
        result = self.build(specs, seed=seed, epoch=epoch)
        seal_mod.seal_corpus(result.public_dev, public_dir, epoch, "public")
        seal_mod.seal_corpus(result.private_test, private_dir, epoch, "private", extra={"note": "held-out; do not publish"})
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
