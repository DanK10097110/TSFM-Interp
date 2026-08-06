"""Regression tests for the hypothesis registry (`ROADMAP.md` sec 15 A15):
`confirm`'s pre-registration discipline (`CLAUDE.md` §6.7) was previously
enforced by convention alone. `hypotheses.json` pins exactly which dev
claims get tested and the exact content hash of the artifact each was
derived from; `confirm` refuses to run if any of those hashes have since
changed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.analysis.hypotheses import build_registry, check_registry_freshness, run_register
from tsfm_lens.config import config_from_dict
from tsfm_lens.utils import save_json


def _cfg_with_l0(out: str) -> object:
    cfg = config_from_dict(build_config(out))
    (cfg.run_dir() / "l0").mkdir(parents=True)
    save_json(cfg.run_dir() / "l0" / "summary.json", {
        "strengths": {"patchy": ["trend"], "steppy": ["spiky"]},
        "mase_ratio": {"trend": 0.7, "spiky": 1.4},
        "per_archetype": {"strengths": {"patchy": ["trend_dominant"]}},
    })
    return cfg


def test_build_registry_reads_l0_strengths_with_artifact_hash():
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    reg = build_registry(cfg)
    l0 = [h for h in reg["hypotheses"] if h["stage"] == "l0"]
    assert {(h["family"], h["favored"]) for h in l0} == {("trend", "patchy"), ("spiky", "steppy")}
    assert all(h["replicable"] for h in l0)
    assert all(len(h["artifact_sha256"]) == 64 for h in l0)


def test_build_registry_marks_archetype_claims_not_replicable():
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    reg = build_registry(cfg)
    arch = [h for h in reg["hypotheses"] if h["stage"] == "l0_archetype"]
    assert len(arch) == 1 and arch[0]["archetype"] == "trend_dominant"
    assert arch[0]["replicable"] is False
    assert "not_replicable_reason" in arch[0]


def test_build_registry_empty_when_no_l0_artifact():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    reg = build_registry(cfg)
    assert reg["hypotheses"] == []


def test_run_register_writes_hypotheses_json():
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    run_register(cfg)
    path = cfg.run_dir() / "hypotheses.json"
    assert path.exists()
    from tsfm_lens.utils import load_json
    reg = load_json(path)
    assert len(reg["hypotheses"]) >= 2


def test_check_registry_freshness_passes_when_unchanged():
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    reg = build_registry(cfg)
    check_registry_freshness(cfg, reg)  # must not raise


def test_check_registry_freshness_raises_naming_drifted_artifact_after_mutation():
    """The exact test this item's fix plan names (item 5): mutate the dev
    artifact after registration, assert confirm's freshness check refuses
    and names the drifted file."""
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    reg = build_registry(cfg)
    # A second look at dev, e.g. re-running l0 with a different seed/config,
    # changes the artifact's content -- exactly the "repeated peeking" this
    # discipline exists to catch.
    save_json(cfg.run_dir() / "l0" / "summary.json", {
        "strengths": {"patchy": ["trend", "spiky"]},  # a claim added after registration
        "mase_ratio": {"trend": 0.7, "spiky": 1.4},
    })
    try:
        check_registry_freshness(cfg, reg)
        raise AssertionError("expected RuntimeError on artifact drift")
    except RuntimeError as e:
        assert "l0/summary.json" in str(e)
        assert "hash changed" in str(e)


def test_check_registry_freshness_raises_when_artifact_deleted():
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    reg = build_registry(cfg)
    (cfg.run_dir() / "l0" / "summary.json").unlink()
    try:
        check_registry_freshness(cfg, reg)
        raise AssertionError("expected RuntimeError on missing artifact")
    except RuntimeError as e:
        assert "no longer exists" in str(e)


def test_confirm_refuses_without_a_registry():
    """`confirm` must not silently re-derive claims from `l0/summary.json`
    if `hypotheses.json` was never written (e.g. `--stages confirm` skipping
    `register`) -- it should refuse loudly instead."""
    from tsfm_lens.analysis.confirm import run_confirm
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    try:
        run_confirm(cfg, hub=None)
        raise AssertionError("expected RuntimeError for missing hypotheses.json")
    except RuntimeError as e:
        assert "hypotheses.json" in str(e)


def test_confirm_refuses_when_registry_is_stale():
    from tsfm_lens.analysis.confirm import run_confirm
    cfg = _cfg_with_l0(tempfile.mkdtemp())
    run_register(cfg)
    save_json(cfg.run_dir() / "l0" / "summary.json", {"strengths": {}, "mase_ratio": {}})
    try:
        run_confirm(cfg, hub=None)
        raise AssertionError("expected RuntimeError for stale registry")
    except RuntimeError as e:
        assert "hash changed" in str(e) or "l0/summary.json" in str(e)


if __name__ == "__main__":
    test_build_registry_reads_l0_strengths_with_artifact_hash()
    test_build_registry_marks_archetype_claims_not_replicable()
    test_build_registry_empty_when_no_l0_artifact()
    test_run_register_writes_hypotheses_json()
    test_check_registry_freshness_passes_when_unchanged()
    test_check_registry_freshness_raises_naming_drifted_artifact_after_mutation()
    test_check_registry_freshness_raises_when_artifact_deleted()
    test_confirm_refuses_without_a_registry()
    test_confirm_refuses_when_registry_is_stale()
    print("hypothesis registry tests passed")
