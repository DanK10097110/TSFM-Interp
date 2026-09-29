"""Capability-matrix generator tests (ROADMAP.md §10's "auto-generate the
capability matrix from the registry" checklist item).

All synthetic -- no checkpoint download or GPU needed to build or test this.
The module was additionally cross-checked by hand against the real,
already-completed `runs/medium_run_chronos_base` (see ROADMAP.md's new
Findings entry for this iteration): it reproduced `CLAUDE.md` §6.2's
hand-maintained capability table exactly, including the one genuinely
subtle case this module exists to automate -- TimesFM's `mlp_info`
*declares* an override but *verified* nothing in that real run, matching
§6.2's documented "usually finds nothing" note. That cross-check is not
repeated here as a pytest test since it depends on a specific run
directory's continued presence, not something this suite should require.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.models.base import ModelAdapter
from tsfm_lens.models.capability_matrix import (
    build_capability_matrix,
    declared_capabilities,
    render_capability_matrix_markdown,
    verified_capabilities_from_run,
)
from tsfm_lens.models.mock import MockPatchAdapter
from tsfm_lens.utils import save_json


def test_declared_capabilities_matches_mock_overrides():
    """mock_patch overrides attention_info/mlp_info/attention_patterns but
    not cross_attention_patterns (it's a decoder-only-shaped mock)."""
    caps = declared_capabilities(MockPatchAdapter)
    assert caps["attention_info_declared"] is True
    assert caps["mlp_info_declared"] is True
    assert caps["attention_patterns_declared"] is True
    assert caps["cross_attention_patterns_declared"] is False
    print("declared_capabilities mock-override test passed")


def test_declared_capabilities_base_class_has_none():
    """A stand-in with no overrides at all must show every capability as not declared."""
    class _BareAdapter(MockPatchAdapter):
        attention_info = ModelAdapter.attention_info
        mlp_info = ModelAdapter.mlp_info
        attention_patterns = ModelAdapter.attention_patterns
        cross_attention_patterns = ModelAdapter.cross_attention_patterns

    caps = declared_capabilities(_BareAdapter)
    assert not any(caps.values()), caps
    print("declared_capabilities bare-adapter test passed")


def test_verified_capabilities_from_run_parses_fixture(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    attn_dir = out / "attention"
    attn_dir.mkdir()
    save_json(attn_dir / "meta.json", {"ModelA": {}, "ModelB": {}})
    np.savez(attn_dir / "arrays.npz",
             head_delta_ModelA=np.zeros(3), lag_profile_ModelA=np.zeros(3),
             head_delta_ModelB=np.zeros(3))

    verified_a = verified_capabilities_from_run(out, "ModelA")
    assert verified_a == {
        "attention_info_verified": True, "mlp_info_verified": False,
        "attention_patterns_verified": True, "cross_attention_patterns_verified": False,
    }, verified_a

    verified_b = verified_capabilities_from_run(out, "ModelB")
    assert verified_b["attention_info_verified"] is True
    assert verified_b["attention_patterns_verified"] is False

    assert verified_capabilities_from_run(out, "ModelC") is None, (
        "a model name absent from meta.json must return None, not all-False")
    print("verified_capabilities_from_run fixture test passed")


def test_verified_capabilities_from_run_returns_none_when_stage_missing(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    assert verified_capabilities_from_run(out, "AnyModel") is None
    print("verified_capabilities_from_run missing-stage test passed")


def test_build_capability_matrix_distinguishes_unverified_from_verified_false(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    attn_dir = out / "attention"
    attn_dir.mkdir()
    save_json(attn_dir / "meta.json", {"Alpha": {}})
    np.savez(attn_dir / "arrays.npz", head_delta_Alpha=np.zeros(2))

    registry = {"mock_patch": MockPatchAdapter}
    df = build_capability_matrix(registry, verify={"mock_patch": (out, "Alpha")})
    row = df.iloc[0]
    assert bool(row["attention_info_verified"]) is True
    assert bool(row["mlp_info_verified"]) is False  # ran, found nothing -- not the same as unchecked
    assert not pd.isna(row["attention_patterns_verified"])
    assert bool(row["attention_patterns_verified"]) is False

    df_unverified = build_capability_matrix(registry)
    assert pd.isna(df_unverified.iloc[0]["attention_info_verified"]), (
        "with no `verify` entry, the column must be NA (never checked), not False")
    print("build_capability_matrix verified-vs-unverified test passed")


def test_render_capability_matrix_markdown_is_a_table():
    df = build_capability_matrix({"mock_patch": MockPatchAdapter})
    md = render_capability_matrix_markdown(df)
    assert md.count("|") > 10
    assert "mock_patch" in md
    print("render_capability_matrix_markdown smoke test passed")


def test_render_capability_matrix_markdown_renders_verified_true_with_no_na_rows(tmp_path=None):
    """Regression: when every row is verified (no NA anywhere), pandas may
    store the column as numpy-bool dtype rather than `object`, so a naive
    `verified is True` check in the renderer silently mis-renders a truly
    verified capability as unverified. This registry+run combo has exactly
    one row and one fully-verified capability, so the column has no NA."""
    out = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    attn_dir = out / "attention"
    attn_dir.mkdir()
    save_json(attn_dir / "meta.json", {"Alpha": {}})
    np.savez(attn_dir / "arrays.npz", head_delta_Alpha=np.zeros(2),
             mlp_delta_Alpha=np.zeros(2), lag_profile_Alpha=np.zeros(2),
             cross_profile_Alpha=np.zeros(2))

    registry = {"mock_patch": MockPatchAdapter}
    df = build_capability_matrix(registry, verify={"mock_patch": (out, "Alpha")})
    assert not df["attention_info_verified"].isna().any()
    md = render_capability_matrix_markdown(df)
    assert "✅ verified" in md, md
    assert "🔶 declared, unverified" not in md, md
    print("render_capability_matrix_markdown all-verified-no-NA regression test passed")


if __name__ == "__main__":
    test_declared_capabilities_matches_mock_overrides()
    test_declared_capabilities_base_class_has_none()
    test_verified_capabilities_from_run_parses_fixture()
    test_verified_capabilities_from_run_returns_none_when_stage_missing()
    test_build_capability_matrix_distinguishes_unverified_from_verified_false()
    test_render_capability_matrix_markdown_is_a_table()
    test_render_capability_matrix_markdown_renders_verified_true_with_no_na_rows()
