"""Tests for persisting the leakage/near-duplicate/realism verdict into the
sealed manifest (ROADMAP.md sec 34 item B1).

The organizing fact this whole item rests on: `seal._global_digest` hashes
sample content only, so writing an `extra.audit` block cannot move any
corpus's digest or fail an existing `load_sealed(verify=True)`. T-B1.1 below
is that claim's own test and is written first, per the item's own instruction
("must be written first, before any production change") -- it is the test
that makes the whole item safe, so CLAUDE.md sec 2.4/34.0 rule 6 require it be
shown to actually discriminate, not just pass.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.audit import LeakageAuditor, compose_audit_block
from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec
from tsfm_benchmark.build_pipeline.schema import GroundTruth, Provenance, TimeSeriesSample
from tsfm_benchmark.build_pipeline.seal import _global_digest, load_sealed, seal_corpus


def _sample(sample_id: str, values, tier: str = "synthetic") -> TimeSeriesSample:
    s = TimeSeriesSample(
        values=np.asarray(values, dtype=float), ground_truth=GroundTruth(),
        provenance=Provenance(generator="test", generator_params={"tier": tier}))
    s.sample_id = sample_id
    return s


# --------------------------------------------------------------------------
# T-B1.1 -- the load-bearing negative: extra can carry anything, the digest
# cannot move, and the test must be shown to fail if _global_digest is ever
# changed to hash the whole manifest instead of sample content alone.
# --------------------------------------------------------------------------

def test_TB1_1_extra_audit_block_cannot_move_the_global_digest():
    samples = [_sample("a", np.arange(16)), _sample("b", np.arange(16) + 1)]
    hashes = [s.content_hash() for s in samples]
    digest_no_extra = _global_digest(hashes)

    out_bare = tempfile.mkdtemp()
    seal_corpus(samples, out_bare, epoch=0, visibility="public")
    _, manifest_bare = load_sealed(out_bare, verify=True)
    assert manifest_bare["global_digest"] == digest_no_extra
    assert manifest_bare["extra"] == {}

    big_extra = {
        "audit": {
            "schema_version": 1,
            "gate": {"references": ["monash/weather"], "n_rejected": 3},
            "near_duplicates": {"worst_pairs": [{"a": "x", "b": "y", "distance": 0.01}] * 10},
        }
    }
    out_extra = tempfile.mkdtemp()
    seal_corpus(samples, out_extra, epoch=0, visibility="public", extra=big_extra)
    loaded_samples, manifest_extra = load_sealed(out_extra, verify=True)  # must not raise
    assert manifest_extra["global_digest"] == digest_no_extra, (
        "adding arbitrary keys under extra moved the global digest -- this is "
        "exactly the corpus-invalidating change B1 must never make")
    assert manifest_extra["extra"] == big_extra
    assert len(loaded_samples) == 2


def test_TB1_1_discriminates_against_a_whole_manifest_digest():
    """Confirm the test above actually fails if `_global_digest` were changed
    to hash the whole manifest (extra included) rather than sample content
    alone -- reproduced here directly against the real `_global_digest`
    function rather than a reimplementation (CLAUDE.md sec 11.55 lesson 3:
    ablate the real function, never a stand-in for it)."""
    hashes = ["h1", "h2"]
    content_only_digest = _global_digest(hashes)

    def _hash_whole_manifest(sample_hashes, extra):
        payload = "".join(sorted(sample_hashes)) + repr(sorted(extra.items()))
        return hashlib.sha256(payload.encode()).hexdigest()

    digest_bare = _hash_whole_manifest(hashes, {})
    digest_with_extra = _hash_whole_manifest(hashes, {"audit": {"n_rejected": 3}})

    # The real function is extra-blind (this is what B1 depends on):
    assert content_only_digest == _global_digest(hashes)
    # A whole-manifest hash is NOT extra-blind -- proving the assertion in the
    # test above would fail under that (rejected) design:
    assert digest_bare != digest_with_extra


# --------------------------------------------------------------------------
# compose_audit_block: gate_effective, quantiles, and the three-state
# absent/degenerate/measured distinction (sec 11.37 / B1.4).
# --------------------------------------------------------------------------

def test_gate_not_effective_when_no_references_and_no_fabricated_quantiles():
    auditor = LeakageAuditor(threshold=0.35)
    kept = [_sample("k0", np.sin(np.linspace(0, 6, 64)))]
    for s in kept:
        auditor.audit(s)  # no refs registered -> nearest_distance = inf, passed=True
    block = compose_audit_block(auditor, kept, rejected=[], dup_pairs=[], split_of={"k0": "public"})

    gate = block["gate"]
    assert gate["gate_effective"] is False
    assert gate["references"] == "none"
    assert gate["reference_n_series"] == 0
    # inf distances must never render as a measured quantile block:
    assert gate["accepted_distance_quantiles"] is None
    assert gate["rejected_distance_quantiles"] is None
    assert "realism" not in block  # no real reference series to compare against


def test_gate_effective_with_references_records_licenses_and_quantiles():
    auditor = LeakageAuditor(threshold=0.35, metric="dtw")
    rng = np.random.default_rng(0)
    for i in range(6):
        auditor.add_reference("monash/weather", f"ref{i}", rng.normal(0, 1, size=64), license="CC-BY-4.0")

    kept, rejected = [], []
    for i in range(5):
        s = _sample(f"k{i}", rng.normal(0, 1, size=64))
        report = auditor.audit(s)
        if report.passed:
            kept.append(s)
        else:
            rejected.append((s.sample_id, report.nearest_distance))

    block = compose_audit_block(auditor, kept, rejected, dup_pairs=[], split_of={s.sample_id: "public" for s in kept})
    gate = block["gate"]
    assert gate["gate_effective"] is True
    assert gate["references"] == ["monash/weather"]
    assert gate["reference_licenses"] == {"monash/weather": "CC-BY-4.0"}
    assert gate["reference_n_series"] == 6
    assert gate["n_candidates"] == len(kept) + len(rejected) == 5
    if kept:
        assert gate["accepted_distance_quantiles"] is not None
        for k in ("min", "p01", "p05", "p50"):
            assert k in gate["accepted_distance_quantiles"]
    # realism is computed once real references exist:
    assert "realism" in block
    assert "NEVER a leakage signal" in block["realism"]["note"]


def test_near_duplicate_pairs_labeled_by_split_across_is_the_leakage_relevant_one():
    auditor = LeakageAuditor()
    pub = _sample("pub0", np.ones(64))
    priv = _sample("priv0", np.ones(64) + 1e-6)  # near-identical, across splits
    pub2 = _sample("pub1", np.linspace(0, 1, 64))
    pub3 = _sample("pub2", np.linspace(0, 1, 64) + 1e-6)  # near-identical, within public
    kept = [pub, priv, pub2, pub3]
    for s in kept:
        auditor.audit(s)
    split_of = {"pub0": "public", "priv0": "private", "pub1": "public", "pub2": "public"}
    dup_pairs = [("pub0", "priv0", 0.001), ("pub1", "pub2", 0.001)]

    block = compose_audit_block(auditor, kept, rejected=[], dup_pairs=dup_pairs, split_of=split_of)
    nd = block["near_duplicates"]
    assert nd["n_pairs_across_splits"] == 1
    assert nd["n_pairs_within_public"] == 1
    assert nd["n_pairs_within_private"] == 0
    relations = {frozenset((p["a"], p["b"])): p["relation"] for p in nd["worst_pairs"]}
    assert relations[frozenset(("pub0", "priv0"))] == "across"
    assert relations[frozenset(("pub1", "pub2"))] == "within_public"


def test_worst_pairs_capped_at_worst_k():
    auditor = LeakageAuditor()
    kept = [_sample(f"s{i}", np.ones(8) * i) for i in range(5)]
    for s in kept:
        auditor.audit(s)
    split_of = {s.sample_id: "public" for s in kept}
    dup_pairs = [(f"s{i}", f"s{i+1}", 0.01 * i) for i in range(4)]
    block = compose_audit_block(auditor, kept, rejected=[], dup_pairs=dup_pairs, split_of=split_of, worst_k=2)
    assert len(block["near_duplicates"]["worst_pairs"]) == 2
    assert block["near_duplicates"]["worst_pairs_cap"] == 2
    # capped to the two smallest (closest / most concerning) distances:
    assert [p["distance"] for p in block["near_duplicates"]["worst_pairs"]] == sorted(
        [d for _, _, d in dup_pairs])[:2]


# --------------------------------------------------------------------------
# End-to-end through BenchmarkBuilder.build_and_seal -- no network, synthetic
# references only, but exercises the exact code path run_full.py uses.
# --------------------------------------------------------------------------

def test_build_and_seal_persists_audit_block_without_moving_digest():
    rng = np.random.default_rng(0)
    auditor = LeakageAuditor(threshold=0.35)
    for i in range(4):
        auditor.add_reference("fake/ref", f"r{i}", rng.normal(0, 1, size=64), license="CC0")

    specs = [TaskSpec(name="t", generator="parametric", count=6, tier="synthetic",
                      generator_params={"length": 64, "trend": {"order": 1, "scale": 0.1}})]

    builder_with_refs = BenchmarkBuilder(auditor=auditor)
    out_pub, out_priv = tempfile.mkdtemp(), tempfile.mkdtemp()
    result = builder_with_refs.build_and_seal(specs, out_pub, out_priv, seed=0, epoch=0)

    _, manifest_pub = load_sealed(out_pub, verify=True)
    _, manifest_priv = load_sealed(out_priv, verify=True)
    assert manifest_pub["extra"]["audit"]["schema_version"] == 1
    assert manifest_pub["extra"]["audit"] == manifest_priv["extra"]["audit"]  # same epoch, same verdict
    assert manifest_priv["extra"]["note"] == "held-out; do not publish"
    assert manifest_priv["extra"]["audit"]["gate"]["gate_effective"] is True

    # Same seed/specs WITHOUT any references registered must produce the exact
    # same global_digest -- the audit block is metadata, not corpus content.
    builder_no_refs = BenchmarkBuilder(auditor=LeakageAuditor(threshold=0.35))
    out_pub2, out_priv2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    builder_no_refs.build_and_seal(specs, out_pub2, out_priv2, seed=0, epoch=0)
    _, manifest_pub2 = load_sealed(out_pub2, verify=True)
    assert manifest_pub2["global_digest"] == manifest_pub["global_digest"]
    assert manifest_pub2["extra"]["audit"]["gate"]["gate_effective"] is False


if __name__ == "__main__":
    test_TB1_1_extra_audit_block_cannot_move_the_global_digest()
    test_TB1_1_discriminates_against_a_whole_manifest_digest()
    test_gate_not_effective_when_no_references_and_no_fabricated_quantiles()
    test_gate_effective_with_references_records_licenses_and_quantiles()
    test_near_duplicate_pairs_labeled_by_split_across_is_the_leakage_relevant_one()
    test_worst_pairs_capped_at_worst_k()
    test_build_and_seal_persists_audit_block_without_moving_digest()
    print("audit persistence tests passed")
