"""Plain-language sentences whose clauses must agree with their own numbers.

Each helper was split out of a report section after the rendered
`runs/concept_atlas_v2` report printed a self-contradicting sentence
(ROADMAP.md sec 37.11c follow-up). Planted numbers with a known correct
reading, plus the boundary case that used to contradict itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import (  # noqa: E402
    _calibration_plain, _frontend_scale_plain, _lens_plain)


def test_poor_calibration_never_claims_nominal_coverage():
    s = _calibration_plain("M", gap=0.30, empirical=0.33, nominal=0.80)
    assert "poorly calibrated" in s
    assert "about as often as claimed" not in s
    assert "less often than claimed" in s and "33% vs 80%" in s


def test_overcoverage_direction_is_stated():
    s = _calibration_plain("M", gap=0.10, empirical=0.95, nominal=0.80)
    assert "somewhat calibrated" in s and "more often than claimed" in s


def test_well_calibrated_keeps_the_nominal_sentence():
    s = _calibration_plain("M", gap=0.01, empirical=0.79, nominal=0.80)
    assert "well calibrated" in s and "about as often as claimed" in s


def test_lens_depth_at_top_of_stack_does_not_claim_later_refinement():
    s = _lens_plain("M", 1.0)
    assert "only refines" not in s and "last layers" in s


def test_lens_early_depth_keeps_the_refinement_sentence():
    s = _lens_plain("M", 0.27)
    assert "27%" in s and "only refines" in s


def test_lens_none_depth():
    assert "never fully settles" in _lens_plain("M", None)


def test_frontend_exact_equivariance_is_not_called_imperfect():
    s = _frontend_scale_plain("M", 1.965e-06, 0.001)
    assert "doesn't perfectly reproduce" not in s and "essentially exactly" in s


def test_frontend_real_residual_keeps_the_mismatch_sentence():
    s = _frontend_scale_plain("M", 0.0332, 0.001)
    assert "doesn't perfectly reproduce" in s and "0.033" in s
