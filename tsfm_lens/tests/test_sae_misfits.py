"""ROADMAP.md sec 30.8 / sec 32.13b, Item L: `sae/misfits.py::misfit_rows`.

All synthetic, with hand-picked cosine values so each test's expected
verdict is computable by hand rather than trusted from the implementation
(`CLAUDE.md` sec 2.4). The centroid in every fixture points purely along
`trend`, so a candidate's cosine to the centroid is exactly its own
trend-axis component after unit-normalizing -- no dependence on centroid
magnitude, which keeps the "same candidate, different concept" comparisons
in `test_threshold_is_relative_to_concept_cohesion` honest.

Item L moved the bar `misfit_rows` compares against from `within_cosine_mean`
(mean PAIRWISE cosine between members) to `centroid_cosine_mean` (mean
member-to-CENTROID cosine) -- the same quantity that gets scored, so the
threshold and the score are finally like-for-like (sec 32.13b: the two used
to disagree by +0.248 on average, which is why the pre-fix detector produced
zero misfits site-wide). `_concept()` below sets `centroid_cosine_mean`
directly, exactly as its pre-fix version set `within_cosine_mean` directly --
a hand-picked bar value, independent of whatever the fixture's own candidate
vectors would compute, since these tests are about threshold BEHAVIOR (does a
looser/tighter bar change the verdict) rather than about re-deriving
`concepts.py::_centroid_cosine_mean`'s own arithmetic (covered separately in
`test_sae_concepts.py`).
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


def _concept(concept_id: int, features: list, centroid_cosine_mean, centroid_trend: float = 5.0) -> dict:
    return {"concept": concept_id, "features": features, "n_members": len(features),
           "n_members_clearing": len(features),
           "centroid_null_units": {ch: (centroid_trend if ch == TREND else 0.0) for ch in CHANNELS},
           "profile": [{"channel": TREND, "signed_null_units": centroid_trend,
                       "n_members_clearing": len(features)}],
           "centroid_cosine_mean": centroid_cosine_mean}


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

    # Tight concept: mean member-to-centroid cosine 1.0 -> threshold 0.70.
    # 0.69 < 0.70, so feature 2 IS a misfit here.
    tight = _target([_concept(0, [1, 2], centroid_cosine_mean=1.0)])
    tight_rows = misfit_rows(tight, ablation_art, min_cosine_gap=0.3)
    tight_features = {r["feature"] for r in tight_rows}
    assert 2 in tight_features, "a member below a TIGHT concept's own bar must be flagged"

    # Loose concept: mean member-to-centroid cosine 0.50 -> threshold 0.20.
    # The SAME candidate (cosine 0.69) clears 0.20, so it is NOT a misfit.
    loose = _target([_concept(0, [1, 2], centroid_cosine_mean=0.50)])
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
    # `centroid_cosine_mean=nan` would already suppress this via the separate
    # NaN guard (that IS what `concepts.py::_centroid_cosine_mean` returns for
    # a singleton), so a `nan` fixture alone cannot tell apart "the n_members
    # guard fired" from "the NaN guard happened to fire too". Use a
    # non-`nan` mean instead -- a stale or hand-built artifact carrying a
    # singleton with a real number in `centroid_cosine_mean` -- so only the
    # explicit `n_members < 2` check can be what suppresses it.
    ablation_art = {"candidates": [_candidate(1, trend=0.0, seasonal=5.0)]}
    singleton = _target([_concept(0, [1], centroid_cosine_mean=1.0)])
    rows = misfit_rows(singleton, ablation_art, min_cosine_gap=0.01)
    assert rows == []


def test_row_carries_both_own_and_concept_channels():
    ablation_art = {"candidates": [
        _candidate(1, trend=5.0),
        _candidate(2, trend=_MEMBER_COSINE, seasonal=_SEASONAL_COMPONENT * 5.0),
    ]}
    tight = _target([_concept(0, [1, 2], centroid_cosine_mean=1.0)])
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


def _concept_from_real_centroid(concept_id: int, features: list, rows_matrix) -> tuple:
    """Build a concept record whose `centroid_null_units` and
    `centroid_cosine_mean` are BOTH derived from the same real member matrix
    `misfits.py` would score against -- unlike `_concept()` above (which
    hand-sets a bar independent of any real vector, for pure threshold-
    behavior tests), the two new tests below need the bar and the scored
    centroid to agree, or a planted member could be flagged/cleared for the
    wrong reason. Returns `(concept_dict, centroid_cosine_mean)`."""
    import numpy as np
    from tsfm_lens.sae.concepts import _centroid_cosine_mean

    centroid = rows_matrix.mean(axis=0)
    ccm = _centroid_cosine_mean(rows_matrix, centroid)
    centroid_null_units = {ch: float(centroid[i]) if i < rows_matrix.shape[1] else 0.0
                           for i, ch in enumerate(CHANNELS)}
    concept = {"concept": concept_id, "features": features, "n_members": len(features),
              "n_members_clearing": len(features),
              "centroid_null_units": centroid_null_units,
              "profile": [{"channel": TREND, "signed_null_units": centroid_null_units[TREND],
                          "n_members_clearing": len(features)}],
              "centroid_cosine_mean": ccm}
    return concept, ccm


def test_planted_orthogonal_member_is_flagged_in_a_tight_concept():
    """ROADMAP.md Item L's load-bearing negative, first half: a concept whose
    members are all tight around the centroid EXCEPT one planted at 90 degrees
    must flag exactly that one member -- and must do so under the REAL
    `centroid_cosine_mean` `concepts.py` would compute, not a hand-set bar,
    so this test also pins that the fix's own arithmetic (not just its
    threshold plumbing, already covered above) discriminates a real plant."""
    import numpy as np

    # Three tight members (pure trend) plus one planted at exactly 90 degrees
    # (pure seasonal) to the trend axis -- NOT to the centroid itself, since
    # the plant pulls the centroid off pure-trend too; the assertions below
    # only rely on the fixed function's own arithmetic, not on this being
    # exactly 90 degrees from the (moved) centroid.
    tight_a = _candidate(1, trend=5.0)
    tight_b = _candidate(2, trend=4.5)
    tight_c = _candidate(3, trend=6.0)
    orthogonal = _candidate(4, trend=0.0, seasonal=5.0)
    ablation_art = {"candidates": [tight_a, tight_b, tight_c, orthogonal]}

    rows_matrix = np.array([[5.0, 0.0], [4.5, 0.0], [6.0, 0.0], [0.0, 5.0]])
    concept, ccm = _concept_from_real_centroid(0, [1, 2, 3, 4], rows_matrix)
    assert 0.5 < ccm < 1.0, "sanity: the plant should pull the mean well below a tight 1.0"
    target = _target([concept])

    rows = misfit_rows(target, ablation_art, min_cosine_gap=0.3)
    flagged = {r["feature"] for r in rows}
    assert flagged == {4}, f"expected only the planted orthogonal member flagged, got {flagged}"


def test_uniformly_tight_concept_flags_nobody():
    """ROADMAP.md Item L's load-bearing negative, second half: a concept
    whose members are ALL close to the centroid (no plant) must produce zero
    misfits at every one of the three swept gaps (0.2/0.3/0.4) -- pairing
    with the test above in the same file is the point (a one-sided test
    passes against a threshold moved to zero, per Item L's acceptance
    clause)."""
    import numpy as np

    members = [_candidate(i, trend=t) for i, t in enumerate([4.8, 5.0, 5.2, 4.9], start=1)]
    ablation_art = {"candidates": members}
    rows_matrix = np.array([[4.8, 0.0], [5.0, 0.0], [5.2, 0.0], [4.9, 0.0]])
    concept, ccm = _concept_from_real_centroid(0, [1, 2, 3, 4], rows_matrix)
    assert ccm == 1.0, "every member points exactly along the centroid's own direction"
    target = _target([concept])

    for gap in (0.2, 0.3, 0.4):
        rows = misfit_rows(target, ablation_art, min_cosine_gap=gap)
        assert rows == [], f"a uniformly tight concept must flag nobody at gap={gap}, got {rows}"


def test_missing_centroid_cosine_mean_skips_rather_than_reusing_the_old_statistic():
    """A concept record written before Item L (or any hand-built fixture)
    carries `within_cosine_mean` but not `centroid_cosine_mean`. The fixed
    `misfit_rows` must skip such a concept entirely rather than falling back
    to `within_cosine_mean` -- silently reapplying the very statistic this
    fix's own docstring names as mismatched would reintroduce sec 32.13b's
    bug under a different name (`CLAUDE.md` sec 2.5: "not yet measured"
    degrades to no rows, never to the wrong rows)."""
    ablation_art = {"candidates": [
        _candidate(1, trend=5.0),
        _candidate(2, trend=0.0, seasonal=5.0),  # orthogonal -- would obviously flag
    ]}
    legacy = {"concept": 0, "features": [1, 2], "n_members": 2,
             "n_members_clearing": 2,
             "centroid_null_units": {ch: (5.0 if ch == TREND else 0.0) for ch in CHANNELS},
             "profile": [{"channel": TREND, "signed_null_units": 5.0, "n_members_clearing": 2}],
             "within_cosine_mean": 1.0}  # old field present, new field ABSENT
    target = _target([legacy])
    rows = misfit_rows(target, ablation_art, min_cosine_gap=0.01)
    assert rows == [], "a concept with no centroid_cosine_mean must be skipped, not scored"


def test_row_persists_which_statistic_set_the_bar():
    """Item L2: a row must print the rule it was judged by. Confirms the row
    carries `bar_statistic` naming the field the threshold came from, and that
    `centroid_cosine_mean` (not the old `concept_mean_cosine` name) is the
    value actually used to build `threshold`."""
    ablation_art = {"candidates": [
        _candidate(1, trend=5.0),
        _candidate(2, trend=_MEMBER_COSINE, seasonal=_SEASONAL_COMPONENT),
    ]}
    tight = _target([_concept(0, [1, 2], centroid_cosine_mean=1.0)])
    rows = misfit_rows(tight, ablation_art, min_cosine_gap=0.3)
    assert len(rows) == 1
    row = rows[0]
    assert row["bar_statistic"] == "centroid_cosine_mean"
    assert row["centroid_cosine_mean"] == 1.0
    assert row["threshold"] == row["centroid_cosine_mean"] - row["min_cosine_gap"]
