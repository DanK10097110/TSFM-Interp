"""Per-sample data roles (ROADMAP sec 41, V3-A): derivation, opt-in byte-identity, sealing, the card.

The load-bearing properties: a role is derived from tier AND checked against the generator (a
tier that contradicts its generator raises, it is not resolved by guessing); a corpus built
without roles is byte-identical to one built before roles existed; a role is bound into the
sample hash only when set, so editing it after sealing fails verification; and the card is
computed from the sealed splits and refuses a split with a missing or inconsistent role.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec
from tsfm_benchmark.build_pipeline.data_roles import build_roles_card, render_markdown, role_violations
from tsfm_benchmark.build_pipeline.schema import GroundTruth, Provenance, SourceRef, TimeSeriesSample, derive_role
from tsfm_benchmark.build_pipeline.seal import load_sealed, seal_corpus


def _sample(generator="parametric", role=None, tier="synthetic", n=32, seed=0):
    values = np.sin(np.arange(n) / 3.0) + seed
    return TimeSeriesSample(
        values=values, ground_truth=GroundTruth(),
        provenance=Provenance(generator=generator, seed=seed, generator_params={"tier": tier, "task_name": "t"}),
        role=role)


def _legacy_hash(s: TimeSeriesSample) -> str:
    """The content hash exactly as it was computed before the role field existed."""
    identity = {"generator": s.provenance.generator, "generator_params": s.provenance.generator_params,
                "seed": s.provenance.seed, "source_refs": [asdict(r) for r in s.provenance.source_refs],
                "transforms": [asdict(t) for t in s.provenance.transforms]}
    prov = json.dumps(identity, sort_keys=True, default=str).encode()
    return hashlib.sha256(np.round(s.values, 6).tobytes() + prov).hexdigest()


def test_derive_role_known_answers():
    assert derive_role("synthetic", "parametric") == "synthetic"
    assert derive_role("synthetic", "random_parametric") == "synthetic"
    assert derive_role("realism_stress", "mixture") == "real_derived"
    assert derive_role("realism_stress", "block_bootstrap") == "real_derived"
    assert derive_role("real_derived", "sequential_par") == "real_derived"


@pytest.mark.parametrize("tier,generator", [
    ("synthetic", "mixture"),
    ("synthetic", "block_bootstrap"),
    ("realism_stress", "parametric"),
    ("realism_stress", "random_parametric"),
    ("external_real", "parametric"),
    ("not_a_tier", "parametric"),
])
def test_derive_role_refuses_contradictions_and_unknown_tiers(tier, generator):
    with pytest.raises(ValueError):
        derive_role(tier, generator)


def test_unknown_role_is_refused_at_construction():
    with pytest.raises(ValueError):
        _sample(role="real")


def test_roleless_sample_serializes_and_hashes_as_before_roles_existed():
    s = _sample()
    assert "role" not in s.to_dict()
    assert '"role"' not in s.to_json()
    assert s.content_hash() == _legacy_hash(s)


def test_role_is_bound_into_the_hash_only_when_set():
    plain, tagged = _sample(), _sample(role="synthetic")
    assert tagged.content_hash() != plain.content_hash()
    assert _sample(role="synthetic").content_hash() == tagged.content_hash()
    assert _sample(role="real_derived").content_hash() != tagged.content_hash()
    assert tagged.to_dict()["role"] == "synthetic"


def _mixture_sources():
    rng = np.random.default_rng(0)
    return [(SourceRef(corpus="hf/fake/src", item_id=str(i), sha256=str(i)), rng.normal(size=80).cumsum()) for i in range(3)]


def _specs():
    return [
        TaskSpec(name="syn", generator="parametric", count=2, tier="synthetic", generator_params={"length": 64}),
        TaskSpec(name="mix", generator="mixture", count=2, tier="realism_stress",
                 generator_params={"mode": "weighted_sum", "sources": _mixture_sources()}),
    ]


def test_builder_assigns_roles_from_tier_and_generator():
    b = BenchmarkBuilder(assign_roles=True)
    got = {spec.name: b._make_one(spec, seed=5, epoch=0).role for spec in _specs()}
    assert got == {"syn": "synthetic", "mix": "real_derived"}


def test_builder_without_assign_roles_is_byte_identical_to_legacy():
    legacy = BenchmarkBuilder()
    for spec in _specs():
        s = legacy._make_one(spec, seed=5, epoch=0)
        assert s.role is None and "role" not in s.to_dict()
        assert s.content_hash() == _legacy_hash(s)


def test_builder_refuses_a_contradicting_tier_and_a_generated_external_role():
    b = BenchmarkBuilder(assign_roles=True)
    bad_tier = TaskSpec(name="x", generator="mixture", count=1, tier="synthetic",
                        generator_params={"mode": "weighted_sum", "sources": _mixture_sources()})
    with pytest.raises(ValueError):
        b._make_one(bad_tier, seed=1, epoch=0)
    external = TaskSpec(name="y", generator="parametric", count=1, tier="synthetic", role="external_real",
                        generator_params={"length": 64})
    with pytest.raises(ValueError):
        b._make_one(external, seed=1, epoch=0)


def test_role_survives_sealing_and_roleless_manifest_has_no_roles_key(tmp_path):
    seal_corpus([_sample(role="synthetic"), _sample(seed=1, role="real_derived", generator="mixture", tier="realism_stress")],
                str(tmp_path / "a"), 0, "public")
    samples, manifest = load_sealed(str(tmp_path / "a"), verify=True)
    assert [s.role for s in samples] == ["synthetic", "real_derived"]
    assert manifest["roles"] == {"synthetic": 1, "real_derived": 1}
    seal_corpus([_sample()], str(tmp_path / "b"), 0, "public")
    _, legacy = load_sealed(str(tmp_path / "b"), verify=True)
    assert "roles" not in legacy


def test_editing_a_role_after_sealing_fails_verification(tmp_path):
    seal_corpus([_sample(role="synthetic")], str(tmp_path), 0, "public")
    path = tmp_path / "corpus.jsonl"
    path.write_text(path.read_text().replace('"role": "synthetic"', '"role": "real_derived"'))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sealed(str(tmp_path), verify=True)


def test_role_violations_flags_each_inconsistency():
    ok = [_sample(role="synthetic"), _sample(seed=1, generator="mixture", role="real_derived", tier="realism_stress"),
          _sample(seed=2, generator="gifteval_window", role="external_real", tier="external_real")]
    assert role_violations(ok) == []
    bad = [_sample(seed=3), _sample(seed=4, generator="mixture", role="synthetic"),
           _sample(seed=5, generator="parametric", role="real_derived"),
           _sample(seed=6, generator="parametric", role="external_real"),
           _sample(seed=7, generator="gifteval_window", role="synthetic")]
    assert len(role_violations(bad)) == 5


def _seal_split(tmp_path, name, samples):
    d = tmp_path / name
    seal_corpus(samples, str(d), 0, "public")
    return str(d)


def test_roles_card_counts_come_from_the_sealed_splits(tmp_path):
    dev = _seal_split(tmp_path, "dev", [
        _sample(seed=0, role="synthetic"), _sample(seed=1, role="synthetic"),
        TimeSeriesSample(values=np.arange(40.0), ground_truth=GroundTruth(), role="real_derived",
                         provenance=Provenance(generator="mixture", seed=2, generator_params={"tier": "realism_stress", "task_name": "m"},
                                               source_refs=[SourceRef(corpus="hf/x/a", item_id="1", sha256="a"),
                                                            SourceRef(corpus="hf/x/a", item_id="2", sha256="b")]))])
    ext = _seal_split(tmp_path, "ext", [_sample(seed=3, generator="gifteval_window", role="external_real", tier="external_real", n=48)])
    card = build_roles_card({"dev": dev, "external_real": ext})
    assert {r: b["count"] for r, b in card["roles"].items()} == {"synthetic": 2, "real_derived": 1, "external_real": 1}
    assert card["roles"]["real_derived"]["sources"] == {"hf/x/a": 1}
    assert card["splits"]["dev"]["by_role"]["synthetic"]["length_min"] == 32
    assert card["splits"]["external_real"]["by_role"]["external_real"]["length_max"] == 48
    assert card["roles"]["external_real"]["never_used_for"].startswith("discovery")
    assert card["roles"]["synthetic"]["used_for"].startswith("causal discovery")
    md = render_markdown(card)
    for role in ("synthetic", "real_derived", "external_real", "sae_augment", "leakage_reference"):
        assert f"`{role}`" in md
    assert "Never used for" in md


def test_roles_card_refuses_a_split_with_a_missing_role(tmp_path):
    dev = _seal_split(tmp_path, "dev", [_sample(role="synthetic"), _sample(seed=1)])
    with pytest.raises(ValueError, match="role violations"):
        build_roles_card({"dev": dev})
