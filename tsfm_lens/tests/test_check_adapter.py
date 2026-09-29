"""Tests for `models/adapter_check.py` (`ROADMAP.md` sec 34.6 Item E4).

T-E4.1: a deliberately-broken adapter's checklist names the actual defect in
its remediation, not a generic "something failed" message. T-E4.2: a tier-2
adapter's missing `attention_patterns` renders `not_applicable`, never
`failed` -- the capability was never claimed at that tier, so its absence is
a fact about the adapter, not a defect (`CLAUDE.md` sec 6.2).

All synthetic, CPU-only, no checkpoint download -- built on the existing
`mock_patch` adapter the same way `models/conformance.py::make_test_adapter`
already does for conformance's own tests.
"""

from __future__ import annotations

import numpy as np
import torch

from tsfm_lens.config import DataConfig, ModelConfig, PipelineConfig, RunConfig
from tsfm_lens.models.adapter_check import (_discover_spans_row, capability_rows,
                                            overall_status, run_adapter_checklist)
from tsfm_lens.models.base import ModelAdapter
from tsfm_lens.models.mock import MockPatchAdapter, MockStepAdapter


class _TierTwoAdapter(MockPatchAdapter):
    """`mock_patch`, minus its attention-capability overrides -- tier 2, not 3.

    Reassigning to the exact base-class function object (not merely
    returning None from a new override) is what makes
    `ModelAdapter._overrides` read these as un-overridden, which is the
    real mechanism `capability_tier()` uses -- a fresh `def ...: return
    None` override would still count as "overridden" and stay tier 3.
    """
    attention_info = ModelAdapter.attention_info
    attention_patterns = ModelAdapter.attention_patterns


class _BrokenSpansAdapter(MockPatchAdapter):
    """`mock_patch`, with `token_time_spans` broken to have non-increasing starts."""

    def token_time_spans(self) -> np.ndarray:
        spans = super().token_time_spans()
        return spans[::-1].copy()


def _build(adapter_cls, context_len: int = 128, horizon: int = 8) -> ModelAdapter:
    mcfg = ModelConfig(name="under_test", adapter="mock_patch")
    data_cfg = DataConfig(context_len=context_len, horizon=horizon)
    adapter = adapter_cls(mcfg, data_cfg, torch.device("cpu"), torch.float32)
    adapter.ensure_loaded()
    return adapter




def test_broken_token_time_spans_names_the_actual_defect_in_remediation():
    """T-E4.1: the conformance row's remediation is specific, not generic."""
    adapter = _build(_BrokenSpansAdapter)
    rows = run_adapter_checklist_from_adapter(adapter)
    conformance = next(r for r in rows if r["name"] == "conformance")
    assert conformance["status"] == "fail"
    assert "starts are not increasing" in conformance["detail"]
    assert conformance["remediation"]

    # Load-bearing negative: the pre-fix adapter (unmodified mock_patch) must
    # NOT trip this same check, so the failure above is really about the
    # planted defect and not some incidental property of every mock.
    healthy = _build(MockPatchAdapter)
    healthy_rows = run_adapter_checklist_from_adapter(healthy)
    healthy_conformance = next(r for r in healthy_rows if r["name"] == "conformance")
    assert healthy_conformance["status"] == "pass"


def test_tier_two_adapters_missing_attention_patterns_is_not_applicable_not_failed():
    """T-E4.2: absence of a tier-3-only capability at tier 2 is a fact, not a defect."""
    adapter = _build(_TierTwoAdapter)
    assert adapter.capability_tier() == 2
    rows = capability_rows(adapter)
    row = next(r for r in rows if r["name"] == "attention_patterns")
    assert row["status"] == "not_applicable"
    assert row["status"] != "fail"

    # Load-bearing negative: a tier-3 adapter (mock_patch itself) DOES declare
    # attention_patterns, so the same capability must NOT read not_applicable
    # there -- the row's meaning depends on what was actually overridden, not
    # on a hardcoded name.
    tier3 = _build(MockPatchAdapter)
    assert tier3.capability_tier() == 3
    tier3_rows = capability_rows(tier3)
    tier3_row = next(r for r in tier3_rows if r["name"] == "attention_patterns")
    assert tier3_row["status"] == "pass"


def test_tier_zero_adapter_reports_every_capability_row_not_applicable():
    """A black-box adapter cannot even call prepare() -- nothing here is attemptable."""
    from tsfm_lens.models.mock import MockBlackBoxAdapter

    adapter = _build(MockBlackBoxAdapter)
    assert adapter.capability_tier() == 0
    rows = capability_rows(adapter)
    assert rows  # never an empty list
    assert all(r["status"] == "not_applicable" for r in rows)


def test_span_discovery_stride_adapts_to_a_per_timestep_tokenizer():
    """A fixed stride wide enough for patch=32 models silently starves a
    patch=1 (per-timestep) model's probe -- measured live against real
    `mock_encdec` before this was fixed: a flat stride of 4 left 96 of 128
    width-1 tokens completely unprobed, collapsing declared-vs-measured IoU
    to 0.062 against an adapter whose declared spans were correct the whole
    time. `_discover_spans_stride` must derive a stride from the model's OWN
    token width rather than use one constant for every architecture."""
    from tsfm_lens.models.adapter_check import _discover_spans_stride

    per_timestep = _build(MockStepAdapter)
    assert per_timestep.token_time_spans().shape[0] == per_timestep.data_cfg.context_len
    assert _discover_spans_stride(per_timestep) == 1

    row = _discover_spans_row(per_timestep)
    assert row["status"] == "pass"
    assert "mean IoU 1.000" in row["detail"] or float(row["detail"].split("IoU ")[1].split(" ")[0]) >= 0.9

    # Load-bearing negative: a coarse-patch model must NOT be forced down to
    # stride 1 -- the whole point of the cap is to keep this checklist fast
    # on architectures that do not need finer probing.
    coarse = _build(MockPatchAdapter)
    assert _discover_spans_stride(coarse) > 1


def test_overall_status_is_fail_iff_any_row_failed():
    passing = [{"name": "a", "status": "pass", "detail": "", "remediation": ""}]
    warning = passing + [{"name": "b", "status": "warn", "detail": "", "remediation": ""}]
    failing = warning + [{"name": "c", "status": "fail", "detail": "", "remediation": ""}]
    assert overall_status(passing) == "pass"
    assert overall_status(warning) == "warn"
    assert overall_status(failing) == "fail"


def test_construct_failure_for_an_unknown_model_name_names_the_config_problem():
    """The whole checklist degrades to one row when the adapter cannot even build."""
    cfg = PipelineConfig(run=RunConfig(device="cpu"),
                         data=DataConfig(context_len=128, horizon=8),
                         models=[ModelConfig(name="nonexistent",
                                             adapter="not_a_real_adapter_name")])
    rows = run_adapter_checklist(cfg, "nonexistent")
    assert len(rows) == 1
    assert rows[0]["name"] == "construct"
    assert rows[0]["status"] == "fail"
    assert "not_a_real_adapter_name" in rows[0]["detail"] or "unknown adapter" in rows[0]["detail"]


def run_adapter_checklist_from_adapter(adapter: ModelAdapter) -> list:
    """Test helper: run the post-construction half of the checklist directly
    against an already-built adapter, bypassing `Context`/`build_adapter` --
    the fixtures above are subclasses that were never registered, so they
    cannot be resolved by name through the real registry."""
    from tsfm_lens.models.adapter_check import (_check_alignment_row,
                                                 _discover_spans_row,
                                                 capability_rows as _cap_rows,
                                                 _row)
    from tsfm_lens.models.conformance import check_adapter_conformance

    rows = [_row("construct", "pass", f"tier {adapter.capability_tier()}")]
    try:
        report = check_adapter_conformance(adapter, 32)
        rows.append(_row("conformance", "pass", str(report)))
    except Exception as exc:
        rows.append(_row("conformance", "fail", f"{type(exc).__name__}: {exc}",
                         remediation="fix the failing assertion named above"))
    rows.append(_check_alignment_row(adapter, 32))
    rows.append(_discover_spans_row(adapter))
    rows.extend(_cap_rows(adapter))
    return rows
