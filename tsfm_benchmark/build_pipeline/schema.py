"""Canonical in-memory representation for every series in the benchmark.

Every sample carries three things that travel together for its whole life:
the raw values, a complete machine-readable provenance trail, and the
ground-truth structural annotations that interpretability evaluation scores
against. Nothing in the pipeline is allowed to produce a series without these.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

ROLES = ("synthetic", "real_derived", "external_real")
"""The data roles a sample can carry (ROADMAP sec 41). Exactly one per sample, opt-in: a
sample whose ``role`` is ``None`` serializes and hashes exactly as it did before roles existed."""

REAL_DERIVED_GENERATORS = frozenset({"mixture", "block_bootstrap", "sequential_par"})
SYNTHETIC_GENERATORS = frozenset({"parametric", "random_parametric"})
_TIER_ROLES = {"synthetic": "synthetic", "realism_stress": "real_derived", "real_derived": "real_derived"}


def derive_role(tier: str, generator: str) -> str:
    """The role implied by a task's leakage tier, cross-checked against its generator.

    ``synthetic`` tiers map to ``synthetic`` and ``realism_stress`` / ``real_derived`` tiers to
    ``real_derived``. A tier that contradicts its generator (a real-derived generator tagged
    ``synthetic``, or a parametric generator tagged ``realism_stress``) raises rather than
    picking one, because a wrong role would mislabel a series as ground-truth-bearing or not.
    ``external_real`` is reserved for raw external windows and is never derived from a task.
    """
    if tier not in _TIER_ROLES:
        raise ValueError(f"cannot derive a role from tier '{tier}'; set `role` on the task explicitly")
    role = _TIER_ROLES[tier]
    if generator in REAL_DERIVED_GENERATORS and role != "real_derived":
        raise ValueError(f"generator '{generator}' consumes real data but tier '{tier}' implies role '{role}'")
    if generator in SYNTHETIC_GENERATORS and role != "synthetic":
        raise ValueError(f"generator '{generator}' touches no real data but tier '{tier}' implies role '{role}'")
    return role


@dataclass
class GroundTruth:
    """Known generative structure used as the interpretability target.

    For purely synthetic series these are exact. For real-derived series they
    record only what we can guarantee (for example mixture weights), and any
    field we cannot vouch for is left empty rather than guessed.
    """

    components: dict[str, list[float]] = field(default_factory=dict)
    changepoints: list[int] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    source_weights: dict[str, float] = field(default_factory=dict)
    generative_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ProvenanceStep:
    """A single ordered transformation applied to a series."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceRef:
    """Reference to a real-data item that influenced a derived series.

    Stores a content hash and license string rather than the raw source so the
    benchmark can be audited and redistributed without leaking the original.
    """

    corpus: str
    item_id: str
    sha256: str
    license: str = "unknown"


@dataclass
class Provenance:
    """Full reproducibility record for one sample."""

    generator: str
    generator_params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    source_refs: list[SourceRef] = field(default_factory=list)
    transforms: list[ProvenanceStep] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    library_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeriesSample:
    """One benchmark series with everything needed to score and audit it."""

    values: np.ndarray
    ground_truth: GroundTruth
    provenance: Provenance
    sample_id: str = ""
    timestamps: list[str] | None = None
    leakage_report: dict[str, Any] | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        if self.role is not None and self.role not in ROLES:
            raise ValueError(f"unknown role '{self.role}'; expected one of {ROLES}")
        self.values = np.asarray(self.values, dtype=float)
        if not self.sample_id:
            self.sample_id = self.content_hash()[:16]

    def content_hash(self) -> str:
        """Reproducible hash over values and identity-bearing provenance.

        Excludes wall-clock and environment metadata (creation time, library
        versions) so the same seed and parameters always yield the same id. A
        ``role`` is bound into the hash only when set, so a role-free sample's
        hash is unchanged and a role edited after sealing fails verification.
        """
        rounded = np.round(self.values, 6).tobytes()
        identity = {
            "generator": self.provenance.generator,
            "generator_params": self.provenance.generator_params,
            "seed": self.provenance.seed,
            "source_refs": [asdict(r) for r in self.provenance.source_refs],
            "transforms": [asdict(t) for t in self.provenance.transforms],
        }
        if self.role is not None:
            identity["role"] = self.role
        prov = json.dumps(identity, sort_keys=True, default=str).encode()
        return hashlib.sha256(rounded + prov).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["values"] = self.values.tolist()
        if d.get("role") is None:
            d.pop("role", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
