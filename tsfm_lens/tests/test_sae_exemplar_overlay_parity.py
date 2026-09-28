"""The two series columns of the feature table must examine the same series.

The correlational column ("Top-activating series") drew a hardcoded 4
thumbnails; the causal column beside it ("Forecast with · without") drew a
hardcoded 3 overlays. Both select from the same descending-activation
ranking, so on `runs/full_report_run_4model` the overlays were literally the
first three of the four thumbnails -- and the fourth read as a series the
causal panel had silently declined to examine.

Neither number is a constant: the overlay count is `--keep-forecasts`, a
property of how the ablation pass was RUN. So the fix derives the exemplar
count from the artifact rather than replacing one hardcoded number with
another, and the caption STATES the relationship after measuring it instead
of asserting the two columns line up.

The load-bearing negatives: a larger `--keep-forecasts` must not be
truncated back to 3, and a run whose two rankings genuinely disagree must
say so rather than inherit the agreeing run's sentence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report import report  # noqa: E402
from tsfm_lens.report.sae_features import ablation_cell  # noqa: E402


def _forecast(sid):
    return {"series_id": sid, "activation": 1.0,
            "context": [0.0, 1.0, 2.0], "target": [3.0, 4.0],
            "with_feature": [3.1, 4.1], "without_feature": [3.5, 4.5]}


def _entry(sids):
    return {"scorable": True, "feature": 7,
            "forecasts": [_forecast(s) for s in sids]}


def _card(fid, sids):
    return {"feature": fid,
            "exemplars": [{"series_id": s, "activation": 1.0} for s in sids],
            "structural_field": "n_seasonalities", "structural_rho": 0.5,
            "structural_n": 10, "provenance_field": None, "provenance_rho": None}


# ---------------------------------------------------------------------------
# the overlay column no longer truncates
# ---------------------------------------------------------------------------

def test_every_kept_forecast_is_drawn_when_no_cap_is_given():
    """The regression: `--keep-forecasts 5` used to render 3."""
    html = ablation_cell(_entry(["a", "b", "c", "d", "e"]))
    assert html.count("<figure") == 5


def test_an_explicit_cap_is_still_honoured():
    html = ablation_cell(_entry(["a", "b", "c", "d", "e"]), 2)
    assert html.count("<figure") == 2


def test_a_feature_with_no_kept_forecasts_says_so_rather_than_rendering_empty():
    html = ablation_cell({"scorable": True, "feature": 1, "forecasts": []})
    assert "no forecasts kept" in html


# ---------------------------------------------------------------------------
# the caption is measured, not asserted
# ---------------------------------------------------------------------------

def test_identical_orderings_are_stated_as_one_series_seen_twice():
    cards = [_card(7, ["a", "b", "c"])]
    clause = report._overlay_series_clause(cards, {7: _entry(["a", "b", "c"])})
    assert "same series in the same order" in clause


def test_disagreeing_orderings_are_not_given_the_agreeing_sentence():
    """The load-bearing negative -- the claim must follow the data."""
    cards = [_card(7, ["a", "b", "c"])]
    clause = report._overlay_series_clause(cards, {7: _entry(["x", "y", "z"])})
    assert "same series in the same order" not in clause
    assert "different series" in clause


def test_partial_agreement_reports_its_own_count():
    cards = [_card(7, ["a", "b"]), _card(8, ["a", "b"])]
    ablations = {7: _entry(["a", "b"]), 8: _entry(["b", "a"])}
    clause = report._overlay_series_clause(cards, ablations)
    assert "1 of 2" in clause


def test_a_target_with_no_ablation_pass_makes_no_claim_at_all():
    assert report._overlay_series_clause([_card(7, ["a"])], {}) == ""
