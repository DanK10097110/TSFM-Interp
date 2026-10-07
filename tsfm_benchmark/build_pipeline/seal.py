"""Materialising and sealing corpora to disk.

The public dev split is meant to be released. The private test split is a
genuinely held-out corpus: it is written to a separate directory that must be
access-controlled and never published, and it is sealed with a manifest that
hashes every sample plus a global digest, so any later tampering or accidental
mixing with the public split is detectable on load.

Sealing and the regeneration protocol are complementary. A sealed corpus is a
stable, versioned artifact for longitudinal comparison; regenerating a new epoch
(see builder) produces a fresh sealed corpus when an old one is suspected of
contamination. This module cannot enforce filesystem permissions; it records the
intent and guarantees integrity, and the private directory must be locked down
by the surrounding deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .generators import BIT_EXACT
from .schema import GroundTruth, Provenance, ProvenanceStep, SourceRef, TimeSeriesSample
import numpy as np


def _global_digest(sample_hashes: list[str]) -> str:
    joined = "".join(sorted(sample_hashes)).encode()
    return hashlib.sha256(joined).hexdigest()


def _determinism_summary(samples: list[TimeSeriesSample]) -> dict[str, Any]:
    """Per-generator `{bit_exact, count}` breakdown for this corpus (sec 15 A8).

    A sealed corpus is a mix of generators with genuinely different
    reproducibility guarantees (`generators.BIT_EXACT`); this is a
    correctness claim a manifest should carry explicitly, not a docs
    footnote a consumer has to already know to go looking for.
    """
    by_gen: dict[str, dict[str, Any]] = {}
    for s in samples:
        gen = s.provenance.generator
        entry = by_gen.setdefault(gen, {"bit_exact": BIT_EXACT.get(gen), "count": 0})
        entry["count"] += 1
    return by_gen


def _role_counts(samples: list[TimeSeriesSample]) -> dict[str, int]:
    """Count of samples per data role, empty when no sample carries one (the legacy case)."""
    counts: dict[str, int] = {}
    for s in samples:
        if s.role is not None:
            counts[s.role] = counts.get(s.role, 0) + 1
    return counts


def seal_corpus(samples: list[TimeSeriesSample], directory: str, epoch: int, visibility: str, extra: dict[str, Any] | None = None) -> str:
    """Write a corpus plus an integrity manifest and return the manifest path.

    ``visibility`` is 'public', 'private' or 'external_real'; a private corpus is the held-out
    test set and the directory it lands in must not be published.
    """
    os.makedirs(directory, exist_ok=True)
    sample_hashes = []
    with open(os.path.join(directory, "corpus.jsonl"), "w") as fh:
        for s in samples:
            sample_hashes.append(s.content_hash())
            fh.write(s.to_json() + "\n")

    manifest = {
        "visibility": visibility,
        "epoch": epoch,
        "n_samples": len(samples),
        "sample_hashes": sample_hashes,
        "global_digest": _global_digest(sample_hashes),
        "determinism": _determinism_summary(samples),
        "extra": extra or {},
    }
    roles = _role_counts(samples)
    if roles:
        manifest["roles"] = roles
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def _sample_from_dict(d: dict[str, Any]) -> TimeSeriesSample:
    gt = GroundTruth(**d["ground_truth"])
    prov_d = dict(d["provenance"])
    prov_d["source_refs"] = [SourceRef(**r) for r in prov_d.get("source_refs", [])]
    prov_d["transforms"] = [ProvenanceStep(**t) for t in prov_d.get("transforms", [])]
    prov = Provenance(**prov_d)
    return TimeSeriesSample(
        values=np.asarray(d["values"], dtype=float),
        ground_truth=gt,
        provenance=prov,
        sample_id=d["sample_id"],
        timestamps=d.get("timestamps"),
        leakage_report=d.get("leakage_report"),
        role=d.get("role"),
    )


def load_sealed(directory: str, verify: bool = True) -> tuple[list[TimeSeriesSample], dict[str, Any]]:
    """Load a sealed corpus, verifying per-sample and global hashes by default.

    Raises if any sample hash or the global digest disagrees with the manifest,
    which catches tampering, truncation, or a public/private mix-up.
    """
    with open(os.path.join(directory, "manifest.json")) as fh:
        manifest = json.load(fh)

    samples: list[TimeSeriesSample] = []
    with open(os.path.join(directory, "corpus.jsonl")) as fh:
        for line in fh:
            if line.strip():
                samples.append(_sample_from_dict(json.loads(line)))

    if verify:
        recomputed = [s.content_hash() for s in samples]
        if recomputed != manifest["sample_hashes"]:
            raise ValueError("sample hash mismatch; sealed corpus integrity check failed")
        if _global_digest(recomputed) != manifest["global_digest"]:
            raise ValueError("global digest mismatch; sealed corpus integrity check failed")

    return samples, manifest
