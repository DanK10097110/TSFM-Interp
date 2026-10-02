"""The power reading of a K2 causal-concept claim (`confirm.causal_claim_reading`).

The causal test is adaptive: only a claim at the p floor is redrawn from
`causal_n_null` to `causal_max_null` draws, and a claim's MDE is computed from
its OWN draws. A claim that was not at the floor therefore reports
`unsatisfiable_correction` although the procedure can reach Holm alpha at the
ceiling, and must not be labelled untestable. The decoy is the same entry with
`n_max == n_null`, which IS untestable. Each assertion names the planted
regression that breaks it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis.confirm import causal_claim_reading  # noqa: E402
from tsfm_lens.report.report import _sec_confirm  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402

UNSAT = {"mde": None, "reason": "unsatisfiable_correction"}


def _t(confirmed=False, dev=0.04, mde=None, n_null=200):
    return {"status": "tested", "confirmed": confirmed, "dev_effect": dev,
            "n_null": n_null, "p": 0.65, "mde": mde if mde is not None else dict(UNSAT)}


def test_not_at_floor_claim_is_not_labelled_untestable():
    """Plant: drop the `n_max > n_used` branch -> 'untestable'."""
    r = causal_claim_reading(_t(), m=31, alpha=0.05, n_max=1000)
    assert "untestable" not in r and "not detected" in r


def test_genuinely_unsatisfiable_is_untestable():
    """Plant: make the branch ignore `m/(n_max+1) <= alpha` -> not 'untestable'."""
    r = causal_claim_reading(_t(), m=31, alpha=0.05, n_max=200)
    assert "untestable" in r
    redraw_cannot_reach = causal_claim_reading(_t(), m=31, alpha=0.05, n_max=300)
    assert "untestable" in redraw_cannot_reach


def test_confirmed_below_mde_has_no_non_confirmation_sentence():
    """Plant: remove the confirmed branch -> 'a non-confirmation reads as underpowered'."""
    r = causal_claim_reading(_t(confirmed=True, dev=0.04, mde={"mde": 0.07}, n_null=1000),
                             m=31, alpha=0.05, n_max=1000)
    assert "non-confirmation" not in r and r.startswith("confirmed")


def test_confirmed_above_mde_is_confirmed_with_power():
    """Plant: confirmed branch always returns the below-MDE wording."""
    r = causal_claim_reading(_t(confirmed=True, dev=0.1, mde={"mde": 0.07}, n_null=1000),
                             m=31, alpha=0.05, n_max=1000)
    assert r == "confirmed with power"


def test_not_confirmed_above_mde_reads_absent_and_below_reads_underpowered():
    """Plant: swap the `dev >= mde` comparison."""
    above = causal_claim_reading(_t(dev=0.1, mde={"mde": 0.07}, n_null=1000), 31, 0.05, 1000)
    below = causal_claim_reading(_t(dev=0.04, mde={"mde": 0.07}, n_null=1000), 31, 0.05, 1000)
    assert "absent" in above and "underpowered" not in above
    assert "underpowered" in below


def test_report_recomputes_reading_from_old_style_entry(tmp_path):
    """Plant: render `non_replication_reading` as stored -> the stale 'untestable' shows."""
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True)
    stale = ("power not computable (unsatisfiable_correction): read this claim as "
             "untestable, not absent")
    entry = dict(_t(), id="c", model="M", layer="L.0", feature=3, channel="level", sign=1,
                 dev_effect_over_null_p95=1.0, private_effect=0.01, p_method="exact",
                 p_holm=1.0, verdict="not confirmed", non_replication_reading=stale)
    base = {"status": "tested", "n_registered": 1, "n_tested": 31, "n_confirmed": 0,
            "n_not_testable": 0}
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_private_series": 40, "alpha": 0.05, "tests": [], "n_registered": 1,
        "n_replicable": 0, "registry_sha256": "abc",
        "concept_replication": {"status": "tested", "causal": dict(
            base, tests=[entry], p_method="exact permutation p ...; adaptive redraw to "
                                          "1000 at the floor")}})
    html = _sec_confirm(run_dir, [], 0)
    assert "Causal feature claims" in html
    assert "untestable" not in html and "not detected" in html
