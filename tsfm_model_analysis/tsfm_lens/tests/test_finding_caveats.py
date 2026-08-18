"""`Finding.caveat` is generated, not authored (`ROADMAP.md` sec 21 J1).

The whole point of J1's three-register claim contract is that `caveat` is a
pure function of a finding's own structured fields (`evidence_class`,
`registered`, `cleared_noise_floor`) plus already-written run artifacts (the
fairness card, the coverage qualifiers) -- never hand-typed prose at a
`findings.append` call site, since `CLAUDE.md` invariant 8's own lesson is
that hand-written caveats go stale exactly when the numbers beside them
change and the caveat doesn't. These tests prove that property directly:
two findings differing in exactly one structured field must produce
differing caveats, and a finding with no applicable clause must not
manufacture one out of nothing (sec 2.5 -- "loud" is not "everywhere").
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import Finding, _compose_caveats
from tsfm_lens.utils import save_json


def _finding(**overrides) -> Finding:
    base = dict(claim_id="test.0", stage="l1", evidence_class="geometric",
               text="L1 — peak similarity CKA=0.50.", plain="placeholder",
               registered=False)
    base.update(overrides)
    return Finding(**base)


def test_caveat_differs_between_registered_and_exploratory(tmp_path):
    """Two findings identical except `registered` must get different caveats,
    and the difference must say what `registered` actually means."""
    exploratory = _finding(registered=False)
    confirmatory = _finding(registered=True)
    out = _compose_caveats([exploratory, confirmatory], tmp_path)
    assert out[0].caveat != out[1].caveat
    assert "exploratory" in out[0].caveat.lower()
    assert "pre-registered" in out[1].caveat.lower()
    assert "exploratory" not in out[1].caveat.lower()


def test_caveat_differs_by_cleared_noise_floor(tmp_path):
    """`cleared_noise_floor` True / False / None (unmeasured) must each read
    as a distinct, correctly-directed claim -- `None` is "not checked", not
    "checked and failed", and must not be conflated with `False`."""
    cleared = _finding(cleared_noise_floor=True,
                       text="Attention — m: most load-bearing head L4·h2 (ΔMASE +0.30x floor).")
    not_cleared = _finding(cleared_noise_floor=False,
                           text="Attention — m: most load-bearing head L4·h2 (ΔMASE +0.30x floor).")
    unmeasured = _finding(cleared_noise_floor=None,
                          text="Attention — m: most load-bearing head L4·h2 (ΔMASE +0.10).")
    out = _compose_caveats([cleared, not_cleared, unmeasured], tmp_path)
    assert "exceeds" in out[0].caveat.lower()
    assert "at or below" in out[1].caveat.lower()
    assert "no repeat-run noise floor was checked" in out[2].caveat.lower()
    assert out[0].caveat != out[1].caveat != out[2].caveat


def test_no_noise_floor_clause_for_a_finding_with_no_delta_and_no_floor_check(tmp_path):
    """A finding that never had a noise-floor concept in the first place (no
    `ΔMASE`/`MASE ratio` in its text, `cleared_noise_floor` never set) must
    not get a fabricated noise-floor sentence -- CKA, AMI and cluster labels
    have no noise floor to report the absence of."""
    f = _finding(evidence_class="geometric",
                text="L1 — peak similarity CKA=0.50 at model_a L4 ↔ model_b L6.")
    out = _compose_caveats([f], tmp_path)
    assert "noise floor" not in out[0].caveat.lower()


def test_evidence_class_clause_is_present_and_matches_the_field():
    """Every evidence class gets its own, distinct one-clause description,
    and the clause must actually appear verbatim for that class."""
    from tsfm_lens.report.report import _EVIDENCE_CLASS_CAVEATS
    findings = [_finding(claim_id=f"test.{i}", evidence_class=ec)
               for i, ec in enumerate(_EVIDENCE_CLASS_CAVEATS)]
    out = _compose_caveats(findings, Path("/nonexistent/run/dir"))
    for f, ec in zip(out, _EVIDENCE_CLASS_CAVEATS):
        assert _EVIDENCE_CLASS_CAVEATS[ec] in f.caveat
    # Every evidence class must produce a caveat distinct from every other.
    assert len({f.caveat for f in out}) == len(out)


def test_depth_located_finding_gets_coverage_qualifier_only_when_the_model_is_undercovered(tmp_path):
    """Reuses the exact artifact `_qualify_depth_claims` already reads
    (`budget/model_budget.json`'s `coverage.depth_claims_qualified`) so the
    caveat's coverage clause can never disagree with the one already
    mechanically appended to `.text`."""
    budget_dir = tmp_path / "budget"
    budget_dir.mkdir()
    save_json(budget_dir / "model_budget.json", {"models": {
        "chronos_like": {"coverage": {
            "depth_claims_qualified": True,
            "headline_flops_fraction": 0.14,
            "headline_is_upper_bound": False,
            "uncaptured_surfaces": ["decoder"],
        }},
        "timesfm_like": {"coverage": {"depth_claims_qualified": False}},
    }})
    depth_finding = _finding(
        stage="lens", evidence_class="descriptive",
        text="Lens — chronos_like: forecast crystallizes at 0.80 of depth.")
    non_depth_finding = _finding(
        stage="sae", evidence_class="descriptive",
        text="SAE — chronos_like: feature 12 best matches has_trend.")
    fully_covered = _finding(
        stage="lens", evidence_class="descriptive",
        text="Lens — timesfm_like: forecast crystallizes at 0.80 of depth.")
    out = _compose_caveats([depth_finding, non_depth_finding, fully_covered], tmp_path)
    assert "captured surface only" in out[0].caveat.lower()
    assert "captured surface only" not in out[1].caveat.lower()
    assert "captured surface only" not in out[2].caveat.lower()


def test_behavioral_finding_gets_size_asymmetry_clause_from_fairness_card(tmp_path):
    """`evidence_class == "behavioral"` findings (accuracy comparisons) get
    the parameter/FLOPs asymmetry clause when the fairness card measured
    one; a geometric finding does not, since the fairness card itself scopes
    that row to "all quality claims" only."""
    fairness_dir = tmp_path / "fairness"
    fairness_dir.mkdir()
    save_json(fairness_dir / "card.json", {"rows": [
        {"Axis": "Parameters", "Asymmetry": "2.10×", "Qualifies": "all quality claims"},
        {"Axis": "FLOPs per forward (per series)", "Asymmetry": "13.00×",
         "Qualifies": "all quality claims"},
    ]})
    behavioral = _finding(evidence_class="behavioral",
                          text="L0 — model_a is significantly stronger on: seasonal.")
    geometric = _finding(evidence_class="geometric",
                         text="L1 — peak similarity CKA=0.50.")
    out = _compose_caveats([behavioral, geometric], tmp_path)
    assert "not matched in size or compute" in out[0].caveat
    assert "not matched in size or compute" not in out[1].caveat


def test_compose_caveats_is_idempotent_shaped_and_never_empty(tmp_path):
    """Every finding must end up with a non-empty caveat -- the function
    must never silently skip a finding."""
    findings = [_finding(claim_id=f"test.{i}", registered=bool(i % 2)) for i in range(5)]
    out = _compose_caveats(findings, tmp_path)
    assert len(out) == len(findings)
    assert all(f.caveat for f in out)
