"""`sae.ablation_null` now defaults to `profile_matched`; the fingerprint stays honest.

The field is `omit_at_value="mean_magnitude"`: the key is omitted at the LEGACY
value (so a config that pins `mean_magnitude` fingerprints exactly as before the
default moved) and present otherwise (so a default run never matches an old
lenient-null concepts artifact). Only the `concepts` stage reads it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.config import SAEConfig, load_config  # noqa: E402
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys  # noqa: E402
from tsfm_lens.pipeline import _stages  # noqa: E402
from tsfm_lens.sae import concept_stage  # noqa: E402
from tsfm_lens.sae.ablation_run import check_ablation_null_modes  # noqa: E402

# Every stage's fingerprint for configs/known_answer.yaml as computed by the code
# BEFORE the default moved (legacy null, key omitted). Recorded from a pristine
# checkout of the parent commit.
OLD_FINGERPRINTS = {'attention': '9b435c76be0c0c23', 'budget': '88b093a6a8f7ac57', 'cluster': 'd11a3835c323cdd4', 'concepts': '294c6f262b0f44cd', 'confirm': '6f9769f4de374b47', 'corpus': '2d2bf22c410839b7', 'exemplars': '4108c4b9b4fb14bd', 'extract': '0ffdfe2233b18c4f', 'frontend': '1cf6a10e23b7368a', 'internals': '31a18964cc40ee6b', 'l0': 'a61ed1f12c11eb4a', 'l1': '603f901e0101bf7a', 'l2': '3f541c74d766ebb5', 'l3': '62885e04897e0eb7', 'layer_screen': '394d92e8c66b1d0e', 'lens': 'b096efeb45fb388c', 'register': '5745122ab2b37170', 'report': '6df0ba9473d0cb01', 'sae': '46b4fb722ff6e0f0'}


def _fingerprints(cfg) -> dict:
    cur = {}
    for s in _stages():
        cur[s.name] = fingerprint_stage(resolve_config_keys(cfg, s.config_keys),
                                        {d: cur[d] for d in s.deps})
    return cur


def _cfg(mode=None):
    cfg = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    if mode is not None:
        cfg.sae.ablation_null = mode
    return cfg


def test_default_is_profile_matched():
    assert SAEConfig().ablation_null == "profile_matched"


def test_legacy_run_is_stale_for_concepts_only_under_the_new_default():
    """(a) A run fingerprinted with the legacy null, read under the new default,
    differs for `concepts` and for no other stage."""
    new = _fingerprints(_cfg())
    changed = sorted(k for k in new if new[k] != OLD_FINGERPRINTS[k])
    assert changed == ["concepts"]


def test_explicit_mean_magnitude_matches_the_old_fingerprints_exactly():
    """(b) Pinning the legacy null reproduces every old fingerprint byte for byte."""
    assert _fingerprints(_cfg("mean_magnitude")) == OLD_FINGERPRINTS


def test_explicit_profile_matched_equals_the_new_default():
    """(c)"""
    assert _fingerprints(_cfg("profile_matched")) == _fingerprints(_cfg())


def test_cov_mode_is_distinct_from_the_default():
    assert _fingerprints(_cfg("profile_matched_cov"))["concepts"] != _fingerprints(_cfg())["concepts"]


def _write_ablation(run_dir: Path, **extra):
    d = run_dir / "sae" / "M"
    d.mkdir(parents=True, exist_ok=True)
    art = {"model": "M", "layer": "blocks.0", "withheld": False, **extra}
    (d / "blocks_0_ablation.json").write_text(json.dumps(art), encoding="utf-8")
    (run_dir / "sae" / "meta.json").write_text(json.dumps({"M/blocks.0": {}}), encoding="utf-8")


def test_check_refuses_a_mode_mismatched_artifact_and_reads_missing_key_as_legacy(tmp_path):
    cfg = _cfg()
    _write_ablation(tmp_path)
    with pytest.raises(RuntimeError, match="mean_magnitude"):
        check_ablation_null_modes(cfg, tmp_path)
    cfg.sae.ablation_null = "mean_magnitude"
    check_ablation_null_modes(cfg, tmp_path)
    _write_ablation(tmp_path, ablation_null="profile_matched")
    with pytest.raises(RuntimeError, match="profile_matched"):
        check_ablation_null_modes(cfg, tmp_path)
    _write_ablation(tmp_path, withheld=True)
    check_ablation_null_modes(cfg, tmp_path)


def test_concepts_stage_refuses_a_mode_mismatched_on_disk_artifact(tmp_path, monkeypatch):
    """(d) The stage itself stops before clustering on a stale-mode artifact."""
    cfg = _cfg()
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = "r"
    cfg.run.keep_models_loaded = True
    run_dir = cfg.run_dir()
    _write_ablation(run_dir)
    monkeypatch.setattr(concept_stage, "run_ablation_all", lambda *a, **k: [])

    def boom(*a, **k):
        raise AssertionError("clustering reached despite a mode-mismatched artifact")
    monkeypatch.setattr(concept_stage, "run_concepts", boom)
    with pytest.raises(RuntimeError, match="different null"):
        concept_stage.run_concept_stage(cfg, None, None, None, "cpu")
