"""Computational coverage: how much of a model a run actually observed.

`report/coverage.json` has always been *section* coverage -- which report
sections rendered. This file covers the other kind (ROADMAP.md §18 F4): what
fraction of each model's own forward computation the capture surface saw. That
is the single most important caveat on every Chronos claim in the repo
(`CLAUDE.md` §12 items 1-2), and until F4 it existed only as prose.

Two properties carry the item and are what these tests are really for:

- the three fractions measure three *different* losses and must not be
  collapsed. `block_fraction` sees capture-stride loss only; an encoder-only
  regex reads 1.0 there while `param_fraction` correctly reads ~0.5. A test
  that only checked "coverage is reported" would pass on a version that
  silently reported the flattering number.
- the depth-claim qualifier is a mechanical rule in the findings builder, not
  author discipline (invariant 8's lesson that discipline-only mechanisms
  decay), so it is tested through the findings list rather than by reading it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.model_budget import capture_coverage
from tsfm_lens.report.report import _qualify_depth_claims
from tsfm_lens.utils import save_json


class _FakeCfg:
    def __init__(self, stride):
        self.capture_layer_stride = stride


class _FakeAdapter:
    """Just the three surfaces `capture_coverage` reads."""

    def __init__(self, name, blocks, stride=1):
        self.name = name
        self._blocks = blocks
        self.cfg = _FakeCfg(stride)

    def all_layer_names(self):
        return list(self._blocks)

    def layer_names(self):
        return list(self._blocks)[:: max(1, self.cfg.capture_layer_stride)]


def _census(per_block, total):
    return {"total": total, "per_block": per_block, "n_blocks": len(per_block)}


def _forward(per_block_flops, total_flops):
    return {"flops": total_flops, "blocks": {"per_block": per_block_flops}}


def test_encoder_only_regex_is_invisible_to_block_fraction_but_not_to_the_others():
    """The Chronos shape: the regex matches every encoder block and nothing
    else, so blocks-captured is a perfect 1.0 while half the model is unseen.
    This is exactly why the headline is a FLOPs fraction, not a block count."""
    blocks = [f"encoder.block.{i}" for i in range(4)]
    cov = capture_coverage(
        _FakeAdapter("Chronos-like", blocks),
        _census({b: 25 for b in blocks}, total=200),      # decoder is the other 100
        _forward({b: 25.0 for b in blocks}, total_flops=100.0),
        predict={"flops": 400.0})                          # encoder + sampled decoder

    assert cov["block_fraction"] == 1.0, "stride loss only -- sees nothing here"
    assert cov["param_fraction"] == pytest.approx(0.5)
    assert cov["flops_fraction_of_capture_pass"] == pytest.approx(1.0)
    assert cov["flops_fraction_of_forecast"] == pytest.approx(0.25)
    # The headline prefers the full-forecast denominator when it exists, since
    # the capture-pass one cannot see a decoder it never runs.
    assert cov["headline_flops_fraction"] == pytest.approx(0.25)
    assert cov["depth_claims_qualified"] is True
    assert any("outside the" in s for s in cov["uncaptured_surfaces"])


def test_capture_stride_loss_is_named_even_when_the_regex_covers_everything():
    blocks = [f"blk.{i}" for i in range(20)]
    cov = capture_coverage(
        _FakeAdapter("TimesFM-like", blocks, stride=2),
        _census({b: 10 for b in blocks}, total=200),
        _forward({b: 5.0 for b in blocks}, total_flops=100.0),
        predict={"flops": 100.0})

    assert cov["captured_blocks"] == 10 and cov["regex_matched_blocks"] == 20
    assert cov["block_fraction"] == pytest.approx(0.5)
    assert cov["param_fraction"] == pytest.approx(0.5)
    assert any("capture_layer_stride=2" in s for s in cov["uncaptured_surfaces"])


def test_a_fully_observed_model_is_not_qualified():
    blocks = [f"blk.{i}" for i in range(4)]
    cov = capture_coverage(
        _FakeAdapter("Whole", blocks),
        _census({b: 24 for b in blocks}, total=100),
        _forward({b: 24.0 for b in blocks}, total_flops=100.0),
        predict={"flops": 100.0})
    assert cov["depth_claims_qualified"] is False
    assert cov["uncaptured_surfaces"] == []


def test_the_upper_bound_rescues_the_model_the_item_exists_for():
    """The first live run's actual failure (ROADMAP.md §18 F4 Findings):
    Chronos-T5-Base's per-block FLOPs don't resolve by name, so every direct
    fraction was None for the one model whose coverage asymmetry this whole
    item was built to measure -- while its record already held both halves of
    a usable ratio. The bound gates correctly and says it is a bound."""
    blocks = [f"encoder.block.{i}" for i in range(12)]
    cov = capture_coverage(
        _FakeAdapter("Chronos-T5-Base", blocks),
        _census({b: 7_000_000 for b in blocks}, total=201_374_976),
        {"flops": 774_755_352_576.0, "blocks": None},   # counter resolved 0 of 12
        predict={"flops": 5_398_283_452_416.0})

    assert cov["flops_fraction_of_forecast"] is None, "still unmeasured, not invented"
    assert cov["capture_pass_fraction_of_forecast"] == pytest.approx(0.1435, abs=1e-3)
    assert cov["headline_flops_fraction"] == pytest.approx(0.1435, abs=1e-3)
    assert cov["headline_is_upper_bound"] is True
    assert cov["depth_claims_qualified"] is True
    assert any("runs unobserved" in s for s in cov["uncaptured_surfaces"])


def test_the_regex_surface_does_not_double_count_stride_loss():
    """The other live-run bug: TimesFM reported "132.9M parameters (57%) lie
    outside the blocks the regex matched" when 98.4M of that was stride loss,
    already reported by the very next surface line. The regex line's
    denominator is the *matched* blocks, not the captured ones."""
    blocks = [f"blk.{i}" for i in range(20)]
    cov = capture_coverage(
        _FakeAdapter("TimesFM", blocks, stride=2),
        # 20 blocks x 9.84M = body 196.7M of a 231.3M total: 34.6M (15%) is
        # genuinely outside the matched blocks; the stride drops another 98.4M.
        _census({b: 9_835_760 for b in blocks}, total=231_289_280),
        _forward({b: 2_527_068_160.0 for b in blocks}, total_flops=59_391_344_640.0),
        predict={"flops": 59_391_344_640.0})

    regex_line = next(s for s in cov["uncaptured_surfaces"] if "layer regex matched" in s)
    assert "34.6M parameters (15%)" in regex_line
    assert "132.9M" not in regex_line, "that number is regex loss plus stride loss"
    assert any("10 of 20 matched blocks are skipped" in s
               for s in cov["uncaptured_surfaces"])
    # The headline is still a genuine measurement here, not a bound.
    assert cov["headline_is_upper_bound"] is False
    assert cov["headline_flops_fraction"] == pytest.approx(0.4255, abs=1e-3)


def test_an_upper_bound_qualifier_reads_as_a_bound(tmp_path):
    save_json(tmp_path / "budget" / "model_budget.json", {"models": {"Chronos-T5-Base": {
        "coverage": {"headline_flops_fraction": 0.1435,
                     "headline_is_upper_bound": True,
                     "depth_claims_qualified": True,
                     "uncaptured_surfaces": []}}}})
    out = _qualify_depth_claims(tmp_path, [
        "Lens — Chronos-T5-Base crystallizes at relative depth 0.73."])
    assert "at least ~86% of Chronos-T5-Base's forward computation is unobserved" in out[0]


def test_a_finding_naming_both_models_states_both_fractions_once(tmp_path):
    """L1's peak-pair finding is the most depth-located claim in the report and
    it names both models, whose coverage differs. Both numbers must appear, the
    shared preamble only once, and neither model's surface inventory inline --
    two full inventories in one finding is how a true qualifier gets skimmed."""
    save_json(tmp_path / "budget" / "model_budget.json", {"models": {
        "TimesFM": {"coverage": {"headline_flops_fraction": 0.4255,
                                 "depth_claims_qualified": True,
                                 "uncaptured_surfaces": ["stride drops 10 of 20"]}},
        "Chronos-T5-Base": {"coverage": {"headline_flops_fraction": 0.1435,
                                         "headline_is_upper_bound": True,
                                         "depth_claims_qualified": True,
                                         "uncaptured_surfaces": ["decoder unobserved"]}}}})
    out = _qualify_depth_claims(tmp_path, [
        "L1 — peak similarity CKA=0.38 at TimesFM L4 ↔ Chronos-T5-Base L10 "
        "(relative depths 0.22 / 0.91)."])

    assert out[0].count("within the captured surface only") == 1
    assert "~57% of TimesFM's forward computation is unobserved" in out[0]
    assert "at least ~86% of Chronos-T5-Base's forward computation is unobserved" in out[0]
    assert "stride drops 10 of 20" not in out[0] and "decoder unobserved" not in out[0]


def test_missing_measurements_are_none_not_zero():
    """An unmeasured coverage and a low coverage are opposite claims -- the
    same tri-state discipline §18 F6 applies to deltas. A `0.0` here would
    read as "we observed nothing", which is a much stronger statement than
    "the FLOP counter was unavailable"."""
    blocks = ["blk.0", "blk.1"]
    cov = capture_coverage(
        _FakeAdapter("NoCounter", blocks),
        _census({b: 50 for b in blocks}, total=100),
        {"flops": None, "blocks": None},
        predict={"error": "boom"})
    assert cov["captured_flops"] is None
    assert cov["flops_fraction_of_capture_pass"] is None
    assert cov["flops_fraction_of_forecast"] is None
    assert cov["headline_flops_fraction"] is None
    assert cov["depth_claims_qualified"] is False, "unmeasured must not qualify"
    assert cov["param_fraction"] == pytest.approx(1.0)


def _budget_with_coverage(run_dir: Path, frac: float, surfaces: list) -> Path:
    save_json(run_dir / "budget" / "model_budget.json", {"models": {"Chronos-T5-Base": {
        "coverage": {"headline_flops_fraction": frac,
                     "depth_claims_qualified": frac < 0.9,
                     "uncaptured_surfaces": surfaces}}}})
    return run_dir


def test_depth_located_findings_about_an_under_captured_model_are_qualified(tmp_path):
    _budget_with_coverage(tmp_path, 0.52, ["the decoder is not captured"])
    out = _qualify_depth_claims(tmp_path, [
        "Lens — Chronos-T5-Base crystallizes at relative depth 0.8.",
        "L1 — peak CKA at Chronos-T5-Base block 10.",
    ])
    assert all("within the captured surface only" in f for f in out)
    assert "~48% of Chronos-T5-Base's forward computation is unobserved" in out[0]
    assert "the decoder is not captured" in out[0]


def test_non_depth_findings_and_other_models_are_left_alone(tmp_path):
    """Deliberately conservative on both sides: a qualifier appended to a
    behavioral claim would be false, and rewriting a claim could change what a
    recorded number means (§2.1)."""
    _budget_with_coverage(tmp_path, 0.52, [])
    out = _qualify_depth_claims(tmp_path, [
        "L0 — Chronos-T5-Base wins on the seasonal family (MASE ratio 0.81).",
        "Lens — TimesFM crystallizes at relative depth 0.4.",
    ])
    assert out == [
        "L0 — Chronos-T5-Base wins on the seasonal family (MASE ratio 0.81).",
        "Lens — TimesFM crystallizes at relative depth 0.4.",
    ]


def test_no_budget_artifact_leaves_every_finding_untouched(tmp_path):
    findings = ["Lens — Chronos-T5-Base crystallizes at relative depth 0.8."]
    assert _qualify_depth_claims(tmp_path, list(findings)) == findings


def test_a_well_captured_model_adds_no_qualifier(tmp_path):
    _budget_with_coverage(tmp_path, 0.98, [])
    findings = ["Lens — Chronos-T5-Base crystallizes at relative depth 0.8."]
    assert _qualify_depth_claims(tmp_path, list(findings)) == findings
