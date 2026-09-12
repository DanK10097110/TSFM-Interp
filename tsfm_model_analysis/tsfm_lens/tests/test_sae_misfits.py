"""ROADMAP.md sec 30.8: `sae/misfits.py::misfit_rows`.

All synthetic, with hand-picked cosine values so each test's expected
verdict is computable by hand rather than trusted from the implementation
(`CLAUDE.md` sec 2.4). The centroid in every fixture points purely along
`trend`, so a candidate's cosine to the centroid is exactly its own
trend-axis component after unit-normalizing -- no dependence on centroid
magnitude, which keeps the "same candidate, different concept" comparisons
in `test_threshold_is_relative_to_concept_cohesion` honest.
"""

from tsfm_lens.sae.concepts import CHANNELS
from tsfm_lens.sae.misfits import misfit_rows

TREND = CHANNELS[0]
SEASONAL = CHANNELS[1]


def _candidate(feature: int, trend: float, seasonal: float = 0.0) -> dict:
    channels = {TREND: {"signed_effect": trend, "null_p95": 1.0, "clears_null": abs(trend) >= 1.0}}
    if seasonal:
        channels[SEASONAL] = {"signed_effect": seasonal, "null_p95": 1.0,
                              "clears_null": abs(seasonal) >= 1.0}
    return {"feature": feature, "scorable": True, "n_channels_clearing": 1,
           "channels": channels}


def _concept(concept_id: int, features: list, within_cosine_mean, centroid_trend: float = 5.0) -> dict:
    return {"concept": concept_id, "features": features, "n_members": len(features),
           "n_members_clearing": len(features),
           "centroid_null_units": {ch: (centroid_trend if ch == TREND else 0.0) for ch in CHANNELS},
           "profile": [{"channel": TREND, "signed_null_units": centroid_trend,
                       "n_members_clearing": len(features)}],
           "within_cosine_mean": within_cosine_mean}


def _target(concepts: list) -> dict:
    return {"model": "m", "layer": "L", "concepts": concepts, "channel_columns": list(CHANNELS)}


# A candidate whose own vector has cosine exactly 0.69 to a pure-trend
# centroid: trend component 0.69, seasonal component sqrt(1 - 0.69**2), so
# the vector has unit norm and cosine == 0.69 exactly regardless of the
# centroid's magnitude.
_MEMBER_COSINE = 0.69
_SEASONAL_COMPONENT = (1.0 - _MEMBER_COSINE ** 2) ** 0.5


def test_threshold_is_relative_to_concept_cohesion():
    ablation_art = {"candidates": [
        _candidate(1, trend=5.0),
        _candidate(2, trend=_MEMBER_COSINE, seasonal=_SEASONAL_COMPONENT),
    ]}

    # Tight concept: mean within-concept cosine 1.0 -> threshold 0.70.
    # 0.69 < 0.70, so feature 2 IS a misfit here.
    tight = _target([_concept(0, [1, 2], within_cosine_mean=1.0)])
    tight_rows = misfit_rows(tight, ablation_art, min_cosine_gap=0.3)
    tight_features = {r["feature"] for r in tight_rows}
    assert 2 in tight_features, "a member below a TIGHT concept's own bar must be flagged"

    # Loose concept: mean within-concept cosine 0.50 -> threshold 0.20.
    # The SAME candidate (cosine 0.69) clears 0.20, so it is NOT a misfit.
    loose = _target([_concept(0, [1, 2], within_cosine_mean=0.50)])
    loose_rows = misfit_rows(loose, ablation_art, min_cosine_gap=0.3)
    loose_features = {r["feature"] for r in loose_rows}
    assert 2 not in loose_features, "the identical member must clear a LOOSE concept's bar"

    # Negative control this test is built to catch: a global constant
    # threshold (independent of the concept's own cohesion) cannot produce
    # this disagreement -- the same absolute cosine (0.69) would either
    # clear both concepts or neither. Confirm the two verdicts really do
    # differ, which is only possible because the bar itself moved.
    assert (2 in tight_features) != (2 in loose_features)


def test_singleton_concept_yields_no_misfit():
    # A candidate whose own vector is maximally far from the concept
    # "centroid" (orthogonal: pure seasonal vs. pure trend) would be an
    # obvious misfit under any positive gap -- except a singleton concept
    # (n_members == 1) has no "own cohesion" to be far from, so it must
    # never be scored at all.
    #
    # `within_cosine_mean=nan` would already suppress this via the separate
    # NaN guard (that IS what `concepts.py::_within_cosine_mean` returns for
    # a singleton), so a `nan` fixture alone cannot tell apart "the n_members
    # guard fired" from "the NaN guard happened to fire too". Use a
    # non-`nan` mean instead -- a stale or hand-built artifact carrying a
    # singleton with a real number in `within_cosine_mean` -- so only the
    # explicit `n_members < 2` check can be what suppresses it.
    ablation_art = {"candidates": [_candidate(1, trend=0.0, seasonal=5.0)]}
    singleton = _target([_concept(0, [1], within_cosine_mean=1.0)])
    rows = misfit_rows(singleton, ablation_art, min_cosine_gap=0.01)
    assert rows == []


def test_row_carries_both_own_and_concept_channels():
    ablation_art = {"candidates": [
        _candidate(1, trend=5.0),
        _candidate(2, trend=_MEMBER_COSINE, seasonal=_SEASONAL_COMPONENT * 5.0),
    ]}
    tight = _target([_concept(0, [1, 2], within_cosine_mean=1.0)])
    rows = misfit_rows(tight, ablation_art, min_cosine_gap=0.3)
    assert len(rows) == 1
    row = rows[0]

    assert "own_top_channels" in row and "concept_channels" in row
    own_channels = {c["channel"] for c in row["own_top_channels"]}
    concept_channels = {c["channel"] for c in row["concept_channels"]}

    # The whole point of the section: the member's own dominant channel
    # (seasonal, scaled up here to dominate) diverges from what its concept
    # is named/profiled on (trend) -- showing only one side would reproduce
    # the naming failure sec 30.1 measurement 2 records.
    assert SEASONAL in own_channels
    assert TREND in concept_channels
    assert own_channels != concept_channels
