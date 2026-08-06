"""Regression tests for the sealed-corpus determinism contract
(`ROADMAP.md` sec 15 A8, fix item 5): a manifest should state which of its
samples' generators are bit-exact reproducible from their seed and which
are best-effort, rather than leaving that as an unstated assumption.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_benchmark.build_pipeline.generators import BIT_EXACT
from tsfm_benchmark.build_pipeline.schema import GroundTruth, Provenance, TimeSeriesSample
from tsfm_benchmark.build_pipeline.seal import load_sealed, seal_corpus


def _sample(generator: str, sample_id: str) -> TimeSeriesSample:
    return TimeSeriesSample(
        values=np.arange(16, dtype=float), ground_truth=GroundTruth(),
        provenance=Provenance(generator=generator, seed=0), sample_id=sample_id)


def test_bit_exact_registry_covers_every_generator_used_by_examples():
    assert BIT_EXACT["parametric"] is True
    assert BIT_EXACT["random_parametric"] is True
    assert BIT_EXACT["mixture"] is True
    assert BIT_EXACT["block_bootstrap"] is True
    assert BIT_EXACT["sequential_par"] is False, (
        "PARSynthesizer exposes no seed argument (CLAUDE.md sec 11.11); must "
        "be flagged best-effort, not silently assumed bit-exact")


def test_sealed_manifest_records_per_generator_determinism():
    samples = [_sample("parametric", "p0"), _sample("parametric", "p1"),
              _sample("sequential_par", "s0")]
    out = tempfile.mkdtemp()
    seal_corpus(samples, out, epoch=0, visibility="public")
    _, manifest = load_sealed(out, verify=True)

    det = manifest["determinism"]
    assert det["parametric"] == {"bit_exact": True, "count": 2}
    assert det["sequential_par"] == {"bit_exact": False, "count": 1}


def test_unknown_generator_records_none_not_a_crash():
    """A generator not in BIT_EXACT (e.g. a future addition not yet
    classified) must degrade to an explicit `bit_exact: None`, not crash the
    seal step or silently claim a guarantee that was never verified."""
    samples = [_sample("some_future_generator", "f0")]
    out = tempfile.mkdtemp()
    seal_corpus(samples, out, epoch=0, visibility="public")
    _, manifest = load_sealed(out, verify=True)
    assert manifest["determinism"]["some_future_generator"] == {
        "bit_exact": None, "count": 1}


if __name__ == "__main__":
    test_bit_exact_registry_covers_every_generator_used_by_examples()
    test_sealed_manifest_records_per_generator_determinism()
    test_unknown_generator_records_none_not_a_crash()
    print("seal determinism tests passed")
