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

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if not self.sample_id:
            self.sample_id = self.content_hash()[:16]

    def content_hash(self) -> str:
        """Reproducible hash over values and identity-bearing provenance.

        Excludes wall-clock and environment metadata (creation time, library
        versions) so the same seed and parameters always yield the same id.
        """
        rounded = np.round(self.values, 6).tobytes()
        identity = {
            "generator": self.provenance.generator,
            "generator_params": self.provenance.generator_params,
            "seed": self.provenance.seed,
            "source_refs": [asdict(r) for r in self.provenance.source_refs],
            "transforms": [asdict(t) for t in self.provenance.transforms],
        }
        prov = json.dumps(identity, sort_keys=True, default=str).encode()
        return hashlib.sha256(rounded + prov).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["values"] = self.values.tolist()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
