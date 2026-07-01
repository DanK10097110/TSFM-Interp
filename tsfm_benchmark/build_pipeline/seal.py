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

from .schema import GroundTruth, Provenance, ProvenanceStep, SourceRef, TimeSeriesSample
import numpy as np


def _global_digest(sample_hashes: list[str]) -> str:
    joined = "".join(sorted(sample_hashes)).encode()
    return hashlib.sha256(joined).hexdigest()


def seal_corpus(samples: list[TimeSeriesSample], directory: str, epoch: int, visibility: str, extra: dict[str, Any] | None = None) -> str:
    """Write a corpus plus an integrity manifest and return the manifest path.

    ``visibility`` is 'public' or 'private'; a private corpus is the held-out
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
        "extra": extra or {},
    }
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
