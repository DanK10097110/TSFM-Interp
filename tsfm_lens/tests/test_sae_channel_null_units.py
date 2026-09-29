"""The feature x channel heatmap's cells are in NULL UNITS, not raw channel
units (ROADMAP.md sec 28.19, user-requested 2026-09-11).

`report/sae_roles.py::feature_channel_matrix` returned raw `signed_mean`
values while its own docstring, the colorbar title, the hovertemplate and the
figure caption all said null-normalized. The nine response channels are not
commensurable raw -- measured on `runs/full_report_run_4model`, a `trend`
effect's median magnitude is 0.00056 against `seasonal`'s 0.72, ~1300x -- so
one shared RdBu scale was pinned by the largest channel and 67-78% of three
of the four models' filled cells rendered within 5% of white, indistinguishable
from a cell that was never measured.

These tests are synthetic with planted answers. The load-bearing property is
that the per-channel nulls in every fixture DIFFER: a fixture whose nulls are
all equal cannot distinguish a normalized matrix from a raw one scaled by a
constant, so every assertion here would pass against the defect (sec 11.53's
"a plant that changes nothing is indistinguishable from a guard that works").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sae_roles import (  # noqa: E402
    feature_channel_matrix, model_feature_channel_matrix)
from tsfm_lens.sae.response import CHANNELS  # noqa: E402

CH = list(CHANNELS)
# Two channels whose nulls differ by 2500x -- the real spread, not a token one.
TINY, BIG = "trend", "seasonal"
P95 = {ch: {TINY: 0.0002, BIG: 0.5}.get(ch, 0.05) for ch in CH}


def _cand(feature: int, signed: dict, clearing: list) -> dict:
    def channels(sign: int) -> dict:
        return {ch: {"available": True,
                     "signed_mean": sign * float(signed.get(ch, 0.0)),
                     "null_p95": P95[ch],
                     "clears_null": ch in clearing}
                for ch in CH}
    return {"feature": feature, "rules": ["variance"],
            "up": {"channels": channels(1)},
            "down": {"channels": channels(-1)},
            "clearing_channels": sorted(clearing)}


def test_cells_are_divided_by_that_channels_own_null_p95():
    """The plant: a `trend` effect 1000x SMALLER in raw units than a
    `seasonal` one, but 5x LARGER relative to its own null. Raw units order
    them one way, null units the other, so this assertion cannot be satisfied
    by the pre-fix code under any constant rescaling."""
    raw_trend, raw_seasonal = 0.002, 2.0          # raw: seasonal is 1000x
    c = _cand(1, {TINY: raw_trend, BIG: raw_seasonal}, [TINY, BIG])
    _, cols, values, _ = feature_channel_matrix([c], CH)
    t = values[0, cols.index(TINY)]
    s = values[0, cols.index(BIG)]

    assert t == 10.0, t                            # 0.002 / 0.0002
    assert s == 4.0, s                             # 2.0   / 0.5
    # The ORDERING flips relative to raw units. This is the whole point:
    # pre-fix the matrix held (0.002, 2.0) and `trend` was invisible.
    assert t > s
    assert raw_trend < raw_seasonal


def test_a_channel_with_no_recorded_null_is_blank_not_raw():
    """A cell with no null to normalize against cannot be expressed in null
    units at all. It must be NaN -- falling back to the raw value would put
    an un-normalized number on the same colour scale as normalized ones,
    which is the defect this module exists to prevent, re-entered through the
    degenerate path (sec 11.37: absent and measured must not render alike)."""
    c = _cand(1, {BIG: 2.0}, [BIG])
    for direction in ("up", "down"):
        c[direction]["channels"][BIG]["null_p95"] = None
    _, cols, values, clears = feature_channel_matrix([c], CH)
    v = values[0, cols.index(BIG)]
    assert np.isnan(v), v
    # Specifically NOT the raw value.
    assert not np.isclose(np.nan_to_num(v), 2.0)


def test_a_zero_null_is_blank_rather_than_infinite():
    """A degenerate null with no spread divides to infinity, which would pin
    the colour scale for the whole figure and blank every other cell -- the
    original defect with the sign reversed."""
    c = _cand(1, {BIG: 2.0}, [BIG])
    for direction in ("up", "down"):
        c[direction]["channels"][BIG]["null_p95"] = 0.0
    _, cols, values, _ = feature_channel_matrix([c], CH)
    v = values[0, cols.index(BIG)]
    assert np.isnan(v), v


def test_the_larger_magnitude_direction_is_still_the_one_kept():
    """Both steering directions share one p95, so normalizing after the
    `max(..., key=abs)` pick cannot change WHICH direction is reported. Pinned
    because a future edit that normalizes each direction before comparing
    would be a silent no-op here and a real change if the two ever diverge."""
    c = _cand(1, {BIG: 3.0}, [BIG])
    c["down"]["channels"][BIG]["signed_mean"] = -7.0     # down now dominates
    _, cols, values, _ = feature_channel_matrix([c], CH)
    assert values[0, cols.index(BIG)] == -14.0           # -7.0 / 0.5


def test_channels_become_comparable_across_columns():
    """The property the figure's caption now claims: two features each moving
    a different channel by the same MULTIPLE of that channel's null render at
    the same intensity, despite raw magnitudes 2500x apart."""
    a = _cand(1, {TINY: 3 * P95[TINY]}, [TINY])
    b = _cand(2, {BIG: 3 * P95[BIG]}, [BIG])
    _, cols, values, _ = feature_channel_matrix([a, b], CH)
    assert values[0, cols.index(TINY)] == values[1, cols.index(BIG)] == 3.0


def test_pooled_model_matrix_keeps_only_rows_that_cleared_something():
    """`model_feature_channel_matrix` is the live consumer. A feature that
    cleared nothing is COUNTED, never drawn as a blank row -- the user's
    'get rid of empty rows'. Checked alongside normalization because both
    must hold at once for the figure to be legible."""
    by_target = {
        "M/layer.0": [_cand(1, {TINY: 0.002}, [TINY]),
                      _cand(2, {BIG: 0.9}, [])],          # clears nothing
        "M/layer.1": [_cand(3, {BIG: 1.5}, [BIG])],
    }
    targets = ["M/layer.0", "M/layer.1"]
    labels, cols, values, clears, n_silent = model_feature_channel_matrix(
        by_target, "M", targets, CH)

    assert n_silent == 1
    assert labels == ["layer.0 · f1", "layer.1 · f3"]
    # Depth order is the caller's target order, not artifact-key order.
    assert values[0, cols.index(TINY)] == 10.0            # 0.002 / 0.0002
    assert values[1, cols.index(BIG)] == 3.0             # 1.5   / 0.5
    # Every drawn row has at least one cell that cleared.
    assert clears.any(axis=1).all()


def test_no_row_drawn_is_entirely_invisible_on_a_shared_scale():
    """The user-visible symptom: a row admitted to the figure must have at
    least one cell that is actually distinguishable from white once the
    shared colour limit is set. Pre-fix this failed for 13 of Sundial's 55
    rows; with per-channel nulls it holds by construction for cells whose
    clearance was decided on the same signed statistic."""
    by_target = {"M/layer.0": [_cand(1, {TINY: 0.002}, [TINY]),
                               _cand(2, {BIG: 2.0}, [BIG])]}
    labels, cols, values, clears, _ = model_feature_channel_matrix(
        by_target, "M", ["M/layer.0"], CH)
    shown = np.where(clears, values, np.nan)
    limit = np.nanmax(np.abs(shown))
    for i, label in enumerate(labels):
        strongest = np.nanmax(np.abs(shown[i]))
        assert strongest / limit > 0.05, (label, strongest, limit)


if __name__ == "__main__":                                 # pragma: no cover
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
