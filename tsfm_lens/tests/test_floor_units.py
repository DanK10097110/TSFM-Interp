"""Every ΔMASE rendered in its own model's repeat-run noise-floor units.

`tests/test_noise_floor.py` covers *measuring* the floor (ROADMAP.md §15 A13).
This file covers what §18 F6 asks for on top of that: that a delta is never
shown, and a finding never emitted, without the reader being told how large it
is relative to calling the same model twice on the same inputs.

The load-bearing design point under test is the **tri-state**: a delta whose
floor was never measured and a delta that sits *below* its floor read almost
identically if each call site writes its own sentence, and those are opposite
claims -- one is "we don't know", the other is "we know, and it's noise". So
`interpretable` is `None`/`False`/`True`, never a bool, and the report's
suppression gate keys on `is False` specifically rather than on falsiness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.stats import format_floor_units, in_floor_units
from tsfm_lens.report import report as R
from tsfm_lens.utils import save_json

_SAMPLING = {"deterministic": False, "mase_abs_delta_mean": 0.10,
             "mase_abs_delta_p95": 0.4, "mase_abs_delta_max": 0.9}
_DETERMINISTIC = {"deterministic": True, "mase_abs_delta_mean": 0.0,
                  "mase_abs_delta_p95": 0.0, "mase_abs_delta_max": 0.0}


def test_ratio_and_verdict_against_a_sampling_model():
    below = in_floor_units(0.05, _SAMPLING)
    assert below["ratio"] == 0.5 and below["interpretable"] is False

    marginal = in_floor_units(0.15, _SAMPLING)
    assert marginal["interpretable"] is False, "1.5x is inside the 2x band"

    clear = in_floor_units(0.5, _SAMPLING)
    assert clear["ratio"] == 5.0 and clear["interpretable"] is True

    # Sign must not change the verdict: a model that got *better* under an
    # intervention is just as noise-limited as one that got worse.
    assert in_floor_units(-0.5, _SAMPLING)["interpretable"] is True
    assert in_floor_units(-0.05, _SAMPLING)["interpretable"] is False


def test_deterministic_model_makes_any_nonzero_delta_real_signal():
    fu = in_floor_units(0.001, _DETERMINISTIC)
    assert fu["deterministic"] is True and fu["interpretable"] is True
    assert fu["ratio"] == float("inf")
    assert in_floor_units(0.0, _DETERMINISTIC)["interpretable"] is False


def test_unmeasured_floor_is_none_not_false():
    """The whole point of the tri-state -- "not checked" must never be
    renderable as "checked and fine", in either direction."""
    for missing in (None, {}):
        fu = in_floor_units(0.5, missing)
        assert fu["interpretable"] is None
        assert fu["ratio"] is None
        assert "floor" in fu["reason"] or "no delta" in fu["reason"]
    assert in_floor_units(None, _SAMPLING)["interpretable"] is None


def test_rendered_phrases_name_the_floor_and_the_verdict():
    assert format_floor_units(in_floor_units(None, _SAMPLING)) == ""

    below = format_floor_units(in_floor_units(0.05, _SAMPLING))
    assert "0.5×" in below and "0.100" in below
    assert "below its own repeat-run noise floor" in below

    marginal = format_floor_units(in_floor_units(0.15, _SAMPLING))
    assert "not distinguishable from repeat-run noise" in marginal

    clear = format_floor_units(in_floor_units(0.5, _SAMPLING))
    assert "5.0×" in clear
    assert "noise" not in clear, "a clear delta must not read as caveated"

    assert "deterministic" in format_floor_units(in_floor_units(0.5, _DETERMINISTIC))
    unmeasured = format_floor_units(in_floor_units(0.5, None))
    assert "no repeat-run floor measured" in unmeasured and "A13" in unmeasured


def _attention_fixture(tmp_path: Path, delta: float, floor: dict | None) -> Path:
    save_json(tmp_path / "attention" / "meta.json", {"M": {
        "patterns": {},
        "ablation": {"head_blocks": ["blk.0"],
                     "top_heads": [{"layer": "blk.0", "head": 1,
                                    "delta_mase": delta}]}}})
    np.savez(tmp_path / "attention" / "arrays.npz",
             head_delta_M=np.array([[0.0, delta]], dtype=np.float32))
    if floor is not None:
        save_json(tmp_path / "l0" / "noise_floor.json", {"M": floor})
    return tmp_path


def _render_attention(run_dir: Path):
    R._FLOOR_AUDIT.update(checked=0, below_floor=0, unmeasured=0, suppressed=[])
    findings: list = []
    html = R._sec_attention(run_dir, {"M": "#000"}, findings)
    return html, findings


def test_a_head_ranking_below_its_floor_emits_no_finding(tmp_path):
    """§18 F6's hard rule: an uninterpretable delta may not become a finding.
    A ranking of heads that all sit inside the noise is a ranking of noise."""
    html, findings = _render_attention(_attention_fixture(tmp_path, 0.05, _SAMPLING))
    assert not any("most load-bearing head" in f.text for f in findings)
    # ...and the suppression is stated in the body, not only in the audit --
    # a silently missing finding is exactly the failure mode invariant 8 names.
    assert "no ranking finding is" in html and "0.5×" in html
    assert R._FLOOR_AUDIT["suppressed"] == ["Attention head ranking (M)"]
    assert R._FLOOR_AUDIT["below_floor"] == 1


def test_a_head_ranking_above_its_floor_keeps_its_finding_and_the_ratio(tmp_path):
    html, findings = _render_attention(_attention_fixture(tmp_path, 0.9, _SAMPLING))
    hit = next(f for f in findings if "most load-bearing head" in f.text)
    assert "9.0×" in hit.text
    assert R._FLOOR_AUDIT["suppressed"] == [] and R._FLOOR_AUDIT["below_floor"] == 0
    assert "no ranking finding is" not in html


def test_an_unmeasured_floor_neither_suppresses_nor_pretends(tmp_path):
    """No `l0/noise_floor.json` at all: the finding survives (suppressing it
    would be asserting noise we never measured) but says so in words."""
    html, findings = _render_attention(_attention_fixture(tmp_path, 0.05, None))
    hit = next(f for f in findings if "most load-bearing head" in f.text)
    assert "no repeat-run floor measured" in hit.text
    assert R._FLOOR_AUDIT["unmeasured"] == 1 and R._FLOOR_AUDIT["below_floor"] == 0
    assert R._FLOOR_AUDIT["suppressed"] == []


def test_audit_counters_reset_between_runs(tmp_path):
    """`_FLOOR_AUDIT` is module state and `run_report` is called repeatedly in
    one process by the test suite itself -- a leaked count would make the
    audit finding report a number from someone else's run."""
    run = _attention_fixture(tmp_path, 0.05, _SAMPLING)
    _render_attention(run)
    _render_attention(run)
    assert R._FLOOR_AUDIT["checked"] == 1
    assert R._FLOOR_AUDIT["suppressed"] == ["Attention head ranking (M)"]
