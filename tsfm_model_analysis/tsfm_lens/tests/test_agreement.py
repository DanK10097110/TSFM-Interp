"""Tests for `ROADMAP.md` sec 20 H4, cross-model agreement as a reliability signal.

All synthetic with planted answers, for the reason `CLAUDE.md` sec 2.4 keeps
insisting on: the quantity under test is a *correlation between two derived
signals*, which will happily return a confident number on data where the
relationship it names does not exist. Every case here fixes the true answer by
construction, and two of them exist specifically because the module's first
version got them wrong on real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.agreement import (_ranks, _spearman, agreement_reliability,
                                           calibration_curve,
                                           distributional_disagreement,
                                           pointwise_disagreement, quantile_width)

_H, _N = 16, 120
_Q = (0.1, 0.5, 0.9)


class _FakeStore:
    """Just enough of `ActivationStore` to serve stored forecasts."""

    def __init__(self, preds: dict):
        self._p = preds

    def load_predictions(self, model: str) -> dict:
        return self._p[model]


def _corpus(seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(_H + 64, dtype=np.float64)
    base = np.sin(2 * np.pi * t / 24)[None, :] * rng.uniform(0.5, 4.0, (_N, 1))
    series = base + rng.normal(0, 0.1, (_N, _H + 64))
    return series[:, :64], series[:, 64:]


def _band(point: np.ndarray, width: np.ndarray) -> np.ndarray:
    """A quantile band of a given per-series width around a point forecast."""
    offs = np.array([-0.5, 0.0, 0.5])
    return point[:, :, None] + width[:, None, None] * offs[None, None, :]


def _preds(point: np.ndarray, width: np.ndarray) -> dict:
    return {"point": point.astype(np.float32), "quantiles": _band(point, width).astype(np.float32)}


def test_ranks_average_ties_so_a_constant_column_correlates_with_nothing():
    """The regression that matters: argsort+arange turns a constant column into
    a clean 0..n-1 ramp in ROW order, which then correlates with whatever else
    happens to be ordered in the corpus. On the first real sweep this handed a
    zero-width quantile band a Spearman of -0.192 against error and produced a
    confident "cross-model agreement BEATS own width" verdict out of a column
    containing no information at all.
    """
    const = np.zeros(10)
    assert np.all(_ranks(const) == 4.5)
    assert _spearman(const, np.arange(10.0)) == 0.0
    assert _spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert _spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    # Partial ties must not silently degrade either.
    assert _ranks(np.array([1.0, 1.0, 2.0])).tolist() == [0.5, 0.5, 2.0]


def test_a_model_with_no_quantile_head_gets_no_baseline_rather_than_a_free_win():
    """A zero-width band is an absent baseline, not a bad one. Scoring against
    it would let cross-model agreement 'beat' nothing (`GenericHFAdapter`
    exposes only a point head, so this is a real configuration, not a
    hypothetical).
    """
    contexts, targets = _corpus()
    rng = np.random.default_rng(1)
    pa = targets + rng.normal(0, 0.3, targets.shape)
    pb = targets + rng.normal(0, 0.3, targets.shape)
    store = _FakeStore({"flat": _preds(pa, np.zeros(_N)),
                        "wide": _preds(pb, rng.uniform(0.2, 2.0, _N))})
    out = agreement_reliability(store, "flat", "wide", contexts, targets, n_boot=100)
    v = out["verdict"]["flat"]
    assert v["own_width_available"] is False
    assert v["beats_own_width"] is False and v["adds_nothing"] is False
    assert "width 0" in v["reason"]
    # The other model, which has a real band, is still scored normally.
    assert out["verdict"]["wide"]["own_width_available"] is True


def test_disagreement_predicts_error_when_it_is_constructed_to():
    contexts, targets = _corpus(2)
    rng = np.random.default_rng(3)
    hard = rng.random(_N)
    # Both models err in proportion to `hard`, in opposite directions, so
    # disagreement and error are both driven by it.
    off = hard[:, None] * rng.normal(0, 1, (_N, _H))
    store = _FakeStore({"a": _preds(targets + off, rng.uniform(0.2, 2.0, _N)),
                        "b": _preds(targets - off, rng.uniform(0.2, 2.0, _N))})
    out = agreement_reliability(store, "a", "b", contexts, targets, n_boot=200)
    r = out["correlations"]["disagreement_point"]["mean"]["spearman"]
    assert r["value"] > 0.5 and r["lo"] > 0.0


def test_a_perfectly_self_aware_model_makes_cross_model_agreement_add_nothing():
    """H4's acceptance criterion, planted: when a model's own quantile width is
    an exact function of its own error, a second checkpoint cannot help, and
    the verdict must say `adds_nothing` rather than reporting the (still
    positive) disagreement correlation on its own.
    """
    contexts, targets = _corpus(4)
    rng = np.random.default_rng(5)
    scale = np.abs(np.diff(contexts, axis=1)).mean(1)
    sev = rng.random(_N)
    err_a = sev[:, None] * rng.normal(0, 1, (_N, _H)) * scale[:, None]
    point_a = targets + err_a
    own = np.abs(err_a).mean(1) / scale  # exactly this model's own MASE
    store = _FakeStore({"a": _preds(point_a, own * scale),
                        "b": _preds(targets + rng.normal(0, 0.2, (_N, _H)) * scale[:, None],
                                    rng.uniform(0.2, 2.0, _N) * scale)})
    out = agreement_reliability(store, "a", "b", contexts, targets, n_boot=300)
    own_rho = out["correlations"]["own_width_a"]["a"]["spearman"]["value"]
    assert own_rho == pytest.approx(1.0, abs=1e-6), "the planted baseline is exact"
    assert out["verdict"]["a"]["adds_nothing"] is True
    assert out["verdict"]["a"]["beats_own_width"] is False


def test_the_verdict_is_the_gap_not_the_raw_correlation():
    """A large positive disagreement correlation coexisting with `adds_nothing`
    is the whole point of the item, and must be representable.
    """
    contexts, targets = _corpus(4)
    rng = np.random.default_rng(5)
    scale = np.abs(np.diff(contexts, axis=1)).mean(1)
    sev = rng.random(_N)
    err_a = sev[:, None] * rng.normal(0, 1, (_N, _H)) * scale[:, None]
    store = _FakeStore({"a": _preds(targets + err_a, np.abs(err_a).mean(1)),
                        "b": _preds(targets - err_a, rng.uniform(0.2, 2.0, _N) * scale)})
    out = agreement_reliability(store, "a", "b", contexts, targets, n_boot=300)
    assert out["correlations"]["disagreement_point"]["a"]["spearman"]["value"] > 0.5
    assert out["verdict"]["a"]["adds_nothing"] is True


def test_distributional_disagreement_sees_what_the_pointwise_one_cannot():
    """Two models can agree exactly on the median and disagree completely about
    how uncertain they are; a pointwise-only signal reports zero there.
    """
    point = np.zeros((_N, _H))
    scale = np.ones(_N)
    a = _band(point, np.full(_N, 0.1))
    b = _band(point, np.full(_N, 5.0))
    assert pointwise_disagreement(point, point, scale).max() == 0.0
    assert distributional_disagreement(a, b, scale).min() > 0.0


def test_signals_are_in_mase_units_so_amplitude_does_not_drive_the_correlation():
    contexts, targets = _corpus(6)
    rng = np.random.default_rng(7)
    pa = targets + rng.normal(0, 0.3, targets.shape)
    pb = targets + rng.normal(0, 0.3, targets.shape)
    w = rng.uniform(0.2, 2.0, _N)
    base = agreement_reliability(_FakeStore({"a": _preds(pa, w), "b": _preds(pb, w)}),
                                 "a", "b", contexts, targets, n_boot=100)
    k = 37.0
    scaled = agreement_reliability(
        _FakeStore({"a": _preds(pa * k, w * k), "b": _preds(pb * k, w * k)}),
        "a", "b", contexts * k, targets * k, n_boot=100)
    for key in ("disagreement_point", "own_width_a"):
        assert scaled["signal_means"][key] == pytest.approx(base["signal_means"][key], rel=1e-4)


def test_calibration_bins_are_equal_count_not_equal_width():
    """Disagreement is heavy-tailed: equal-width bins put nearly every series
    in the first bin and describe the tail rather than the corpus.
    """
    signal = np.exp(np.linspace(0, 8, 100))
    curve = calibration_curve(signal, np.arange(100.0), n_bins=10)
    counts = {b["n"] for b in curve["bins"]}
    assert counts == {10}
    means = [b["mean_error"] for b in curve["bins"]]
    assert means == sorted(means)


def test_small_families_are_omitted_rather_than_given_a_meaningless_ci():
    contexts, targets = _corpus(8)
    rng = np.random.default_rng(9)
    fam = np.array(["big"] * (_N - 4) + ["tiny"] * 4)
    w = rng.uniform(0.2, 2.0, _N)
    store = _FakeStore({"a": _preds(targets + rng.normal(0, 0.3, targets.shape), w),
                        "b": _preds(targets + rng.normal(0, 0.3, targets.shape), w)})
    out = agreement_reliability(store, "a", "b", contexts, targets,
                                families=fam, n_boot=50)
    assert "big" in out["by_family"] and "tiny" not in out["by_family"]


def test_horizon_resolved_curve_has_one_entry_per_step():
    contexts, targets = _corpus(10)
    rng = np.random.default_rng(11)
    w = rng.uniform(0.2, 2.0, _N)
    store = _FakeStore({"a": _preds(targets + rng.normal(0, 0.3, targets.shape), w),
                        "b": _preds(targets + rng.normal(0, 0.3, targets.shape), w)})
    out = agreement_reliability(store, "a", "b", contexts, targets, n_boot=50)
    assert len(out["spearman_by_horizon_step"]) == _H
    assert all(np.isfinite(out["spearman_by_horizon_step"]))


def test_quantile_width_is_the_band_not_the_point():
    point = np.ones((5, 4))
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = quantile_width(_band(point, w), np.ones(5))
    assert got.tolist() == pytest.approx(w.tolist())


def test_output_serializes_without_numpy_types():
    import json
    contexts, targets = _corpus(12)
    rng = np.random.default_rng(13)
    w = rng.uniform(0.2, 2.0, _N)
    store = _FakeStore({"a": _preds(targets + rng.normal(0, 0.3, targets.shape), w),
                        "b": _preds(targets + rng.normal(0, 0.3, targets.shape), np.zeros(_N))})
    out = agreement_reliability(store, "a", "b", contexts, targets,
                                families=np.array(["f"] * _N), n_boot=50)
    json.dumps(out)
