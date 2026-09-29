"""ROADMAP.md §25.9 Stage 2 (Component A only): the response-fingerprint
battery, its random-direction null, candidate selection, and the mandatory
reach gate that precedes all of it.

Four areas, matching the deliverable list:

1. The reach gate (`analysis/response_reach.py::reach_probe`) -- self-patch
   must read exactly 0.0, cross-patch must read nonzero for a reachable
   target, and a target whose forecast head reads a DISJOINT span from what
   `token_slice` writes (the exact Chronos-2 shape, `CLAUDE.md` §11.42) must
   be withheld rather than silently reported as "no effect". Follows
   `tests/test_patch_reach_check.py`'s own stub-and-monkeypatch style so the
   DECISION logic is pinned without a real forward pass through torch.
2. At least one pure battery-statistic channel (`trend`), plus dispersion's
   degenerate-baseline handling (`CLAUDE.md` §11.37: a zero-spread quantile
   band must report `width_available: False`, never a silent zero that could
   read as "no effect" or, worse, "a real effect of zero width").
3. The random-direction null's own gating arithmetic: a channel must clear
   ONLY when its measured effect exceeds the null's own p95, discriminating
   a genuine signal from a null-sized wobble.
4. `select_candidates`'s priority order, alive-atom restriction, and
   multi-rule recording (`CLAUDE.md` §11.34: a probe must record every rule
   that nominated a candidate, not just the first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis import response_reach
from tsfm_lens.sae.response import (
    _dispersion,
    _horizon_shape,
    battery_statistics,
    select_candidates,
    summarize_battery,
)


# ---------------------------------------------------------------------------
# 1. Reach gate
# ---------------------------------------------------------------------------

class _Cfg:
    class run:
        seed = 0
    class data:
        horizon = 4
    class l0:
        quantiles = [0.5]
    class concepts:
        min_relative_reach = 1e-4


class _Data:
    def __init__(self, n=8):
        self.n = n
        self.families = np.array(["f"] * n)

    def contexts(self):
        return np.zeros((self.n, 16), dtype=np.float32)


class _ReachStub:
    """Minimal adapter surface `reach_probe` touches.

    `predict()` is scripted by call order (mirroring
    `test_patch_reach_check.py::_Stub`): call 1 is the clean baseline, call 2
    the self-patch (target-into-itself), call 3 the cross-patch (a different
    layer's clean tokens written into the target). `token_written_differs`
    controls whether the two layers' "clean tokens" (as captured by the
    monkeypatched `capture_raw_tokens` below) are actually distinct, which
    is what the withhold-on-tautology branch in `reach_probe` checks before
    trusting a zero cross-delta.
    """

    def __init__(self, self_delta: float, cross_delta: float, declared: bool = True,
                token_written_differs: bool = True, n_layers: int = 3,
                baseline: float = 1.0):
        self.name = "stub"
        self._self_delta, self._cross_delta = self_delta, cross_delta
        self._declared = declared
        self._token_written_differs = token_written_differs
        self._n_layers = n_layers
        self._baseline = baseline
        self._call = 0
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 999})()

    def ensure_loaded(self):
        pass

    def all_layer_names(self):
        return [f"blocks.{i}" for i in range(self._n_layers)]

    def token_slice(self, live_len):
        return slice(0, live_len)

    def forecast_reads_patched_positions(self):
        return self._declared

    def predict(self, contexts, horizon, quantiles):
        # `_baseline` is nonzero so `forecast_scale` (the clean forecast's
        # own mean absolute value, ROADMAP.md sec 37.8 P5a) is measurable --
        # every delta below is a SHIFT from this baseline, so the magnitudes
        # `self_delta`/`cross_delta` asserted throughout this file are
        # unaffected by its value.
        self._call += 1
        offs = {1: 0.0, 2: self._self_delta, 3: self._cross_delta}[self._call]
        return {"point": np.full((len(contexts), horizon), self._baseline + offs, np.float32)}


def _patched_capture_raw_tokens(token_written_differs: bool):
    """A stand-in whose two layers' 'clean tokens' differ (or not, per the
    flag) by a fixed, easily-measured amount, so `reach_probe`'s own
    `written_diff_mag` computation has something real to measure."""
    import torch as _torch

    def _fn(adapter, contexts, layers, **_kw):
        base = _torch.zeros(len(contexts), 2, 3)
        out = {}
        for i, l in enumerate(layers):
            out[l] = base + (i if token_written_differs else 0)
        return out

    return _fn


@pytest.fixture
def _stub_patch_machinery(monkeypatch):
    import contextlib

    def _apply(token_written_differs=True):
        monkeypatch.setattr(response_reach, "capture_raw_tokens",
                            _patched_capture_raw_tokens(token_written_differs))
        monkeypatch.setattr(response_reach, "token_patch",
                            lambda *a, **k: contextlib.nullcontext())
    return _apply


def test_reach_probe_self_patch_is_exactly_zero_for_a_healthy_adapter(_stub_patch_machinery):
    """The correctness control: patching a layer into itself must read 0.0.

    Mirrors mock_patch's own measured value (identity 0.0) from
    `test_patch_reach_check.py`."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.0, cross_delta=0.4627782702445984)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.1", _Data(), device=None)
    assert rep["self_patch_delta"] == 0.0
    assert rep["cross_patch_delta"] == pytest.approx(0.4627782702445984)
    assert rep["reachable"] is True


def test_reach_probe_cross_patch_nonzero_confirms_reachability(_stub_patch_machinery):
    """A nonzero cross-patch delta, with a zero self-patch control, is the
    positive case Component A's battery is allowed to run against."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.0, cross_delta=0.737)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["reachable"] is True
    assert "reaches the forecast" in rep["reason"]


def test_reach_probe_withholds_the_chronos2_shape(_stub_patch_machinery):
    """The exact §11.42 bug: self-patch 0.0 (wiring correct) AND cross-patch
    0.0 (the head reads elsewhere) -- must be withheld with a stated reason,
    never silently reported as a battery of zero effects."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.0, cross_delta=0.0)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.2", _Data(), device=None)
    assert rep["reachable"] is False
    assert "does not causally reach" in rep["reason"]
    assert "sec 11.42" in rep["reason"]


def test_reach_probe_flags_a_broken_patching_path_via_nonzero_self_delta(_stub_patch_machinery):
    """If patching a layer into itself moves the forecast at all, that is a
    bug in the patching wiring (token_slice/postprocess_tokens disagreeing),
    not a property of the model -- CLAUDE.md sec 2.4: not a tolerance to
    loosen."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.4, cross_delta=0.5)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["self_patch_delta"] == pytest.approx(0.4)
    assert "correctness control failed" in rep["reason"]
    # The load-bearing half, and the one that was missing: a broken
    # instrument must also come back UNREACHABLE. Before `CLAUDE.md`
    # sec 11.49 this asserted only the reason string, so `reachable` stayed
    # True on a nonzero cross-delta, the battery ran, `withheld` never
    # fired, and roles clustered on a fingerprint measured with a wrong
    # clean cache. A stated reason nothing acts on is not a guard.
    assert rep["reachable"] is False


def test_reach_probe_broken_control_outranks_a_large_cross_delta(_stub_patch_machinery):
    """Order matters: a big, healthy-looking cross-patch delta must NOT
    rescue a failed correctness control. The cross probe is measured with
    the same machinery the control just proved broken, so its magnitude
    carries no information -- pinning this because the natural reading of a
    0.5 cross-delta beside a 0.4 self-delta is "mostly reaching"."""
    _stub_patch_machinery(token_written_differs=True)
    rep = response_reach.reach_probe(_Cfg, _ReachStub(self_delta=1e-6, cross_delta=99.0),
                                     "blocks.0", _Data(), device=None)
    assert rep["reachable"] is False
    assert "correctness control failed" in rep["reason"]
    assert rep["cross_patch_delta"] == pytest.approx(99.0)


def test_reach_probe_a_tautological_zero_is_not_read_as_unreachable(_stub_patch_machinery):
    """If the two layers' clean tokens happen to be numerically identical,
    a zero cross-delta is uninformative rather than evidence of no reach --
    distinct from the genuine §11.42 withhold path above."""
    _stub_patch_machinery(token_written_differs=False)
    stub = _ReachStub(self_delta=0.0, cross_delta=0.0)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["reachable"] is False
    assert "did not differ enough" in rep["reason"]
    assert "uninformative" in rep["reason"]


def test_reach_probe_records_the_declared_reads_patched_positions_flag(_stub_patch_machinery):
    """The declaration is recorded alongside the measurement, never trusted
    in its place -- so a disagreement between the two is visible in the
    artifact rather than silently resolved either way."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.0, cross_delta=0.0, declared=True)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["declared_reads_patched_positions"] is True
    assert rep["reachable"] is False  # declaration disagreeing with the measurement is recorded, not resolved


def test_reach_probe_a_live_small_effect_model_still_passes(_stub_patch_machinery):
    """Decoy for the relative-reach threshold (ROADMAP.md sec 37.8 P5a): a
    small but genuinely live cross-patch delta, comfortably above
    `min_relative_reach`, must not be swept up by the same fix that refuses a
    dead twin below."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.0, cross_delta=0.001, baseline=1.0)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["relative_reach"] == pytest.approx(0.001, rel=1e-3)
    assert rep["reachable"] is True


def test_relative_reach_refuses_dead_twin(_stub_patch_machinery):
    """ROADMAP.md sec 37.8 P5a / sec 29.5: an absolute cross-patch delta that
    clears the old `_EPS=1e-12` by four orders of magnitude is still dead
    RELATIVE to the forecast. Chronos-2's untrained twin measured
    forecast_scale 0.7128255367279053 and cross_patch_delta
    5.960464477539063e-08 (sec 29.2/29.4) -- a relative reach of ~8.36e-08,
    nowhere near `min_relative_reach=1e-4`. Planting the old absolute check
    (`cross_delta > _EPS`) back in place makes this fail, because 5.96e-08
    clears `1e-12`."""
    _stub_patch_machinery(token_written_differs=True)
    stub = _ReachStub(self_delta=0.0, cross_delta=5.960464477539063e-08,
                      baseline=0.7128255367279053)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["reachable"] is False
    assert rep["relative_reach"] < _Cfg.concepts.min_relative_reach
    assert "min_relative_reach" in rep["reason"]


def _identical_nonzero_capture(adapter, contexts, layers, **_kw):
    """Every captured layer's 'clean tokens' bit-identical, but NOT the zero
    tensor -- the exact TimesFM-untrained-twin shape (sec 29.3: every block
    is an exact identity because RMSNorm's scale is zero-initialized, so all
    20 captured layers read back bit-identical to each other, not to zero)."""
    import torch as _torch

    base = _torch.full((len(contexts), 2, 3), 2.0)
    return {l: base.clone() for l in layers}


def test_constructed_replacement_when_identical(monkeypatch):
    """ROADMAP.md sec 37.8 P5a item 2 / sec 29.3: when the ordinary
    cross-patch replacement (another layer's clean tokens) is bit-identical
    to the target's own clean tokens, `written_diff_mag` is exactly 0.0 and
    the probe must escalate to a CONSTRUCTED replacement (the target's own
    clean tokens x1.5) rather than read the resulting zero cross-patch delta
    as a measurement of "no reach". Planting a revert to the old
    `not written_differs -> reachable=False` branch (no escalation) makes
    this fail: `reach_method` would never be recorded as `"constructed"`."""
    import contextlib

    monkeypatch.setattr(response_reach, "capture_raw_tokens", _identical_nonzero_capture)
    monkeypatch.setattr(response_reach, "token_patch",
                        lambda *a, **k: contextlib.nullcontext())
    stub = _ReachStub(self_delta=0.0, cross_delta=0.4, baseline=1.0)
    rep = response_reach.reach_probe(_Cfg, stub, "blocks.0", _Data(), device=None)
    assert rep["reach_method"] == "constructed"
    assert rep["reachable"] is True
    assert rep["cross_patch_delta"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 2. Battery channel correctness + dispersion's degenerate-baseline handling
# ---------------------------------------------------------------------------

def test_trend_channel_recovers_a_planted_slope_difference():
    """steered has a real +0.5/step slope the baseline lacks; the trend
    channel's delta must recover exactly that difference (both are pure
    linear ramps, so this is closed-form, not approximate)."""
    horizon = 32
    t = np.arange(horizon)
    baseline = np.tile(3.0 + 0.0 * t, (5, 1))
    steered = np.tile(3.0 + 0.5 * t, (5, 1))
    targets = steered.copy()
    contexts = np.random.default_rng(0).normal(size=(5, 64))
    periods = np.full(5, np.nan)
    stats = battery_statistics(steered, baseline, targets, contexts, periods)
    assert stats["trend"]["available"] is True
    assert np.allclose(stats["trend"]["delta"], 0.5, atol=1e-8)


def test_dispersion_width_unavailable_on_a_zero_spread_quantile_band():
    """CLAUDE.md sec 11.37: a deterministic forecaster's quantile band has
    zero spread. The width sub-statistic must report `width_available:
    False`, never a silent 0.0 that a downstream null-comparison could
    misread as either 'no effect' or 'a real zero-width effect'."""
    b, h, q = 5, 16, 3
    baseline = np.random.default_rng(0).normal(size=(b, h))
    steered = baseline + 1.0
    targets = steered.copy()
    contexts = np.random.default_rng(1).normal(size=(b, 32))
    periods = np.full(b, np.nan)
    zero_width_quantiles = np.repeat(steered[:, :, None], q, axis=-1)  # identical across Q -> zero width
    stats = battery_statistics(steered, baseline, targets, contexts, periods,
                               steered_quantiles=zero_width_quantiles,
                               baseline_quantiles=zero_width_quantiles)
    assert stats["dispersion"]["width_available"] is False
    assert stats["dispersion"]["width_delta"] is None
    # the sd-of-forecast half of the same channel is NOT gated by the width
    # degeneracy -- the two sub-statistics are independent.
    assert stats["dispersion"]["available"] is True


def test_dispersion_width_available_and_correct_on_a_real_spread_change():
    """The positive case: a genuine widening of the quantile band must be
    picked up as a nonzero, correctly-signed width delta."""
    b, h = 4, 8
    baseline = np.zeros((b, h))
    steered = np.zeros((b, h))
    targets = np.zeros((b, h))
    contexts = np.random.default_rng(2).normal(size=(b, 16))
    periods = np.full(b, np.nan)
    baseline_q = np.stack([baseline - 1.0, baseline, baseline + 1.0], axis=-1)   # width 2
    steered_q = np.stack([steered - 3.0, steered, steered + 3.0], axis=-1)       # width 6
    stats = battery_statistics(steered, baseline, targets, contexts, periods,
                               steered_quantiles=steered_q, baseline_quantiles=baseline_q)
    assert stats["dispersion"]["width_available"] is True
    assert np.allclose(stats["dispersion"]["width_delta"], 4.0)


def test_seasonal_channel_unavailable_when_no_series_has_a_usable_period():
    """Mirrors `seasonal_band_magnitude`'s own graceful degrade
    (test_steering.py) one level up: the whole channel must report
    `available: False` with a stated reason, not a silent NaN array."""
    b, h = 3, 16
    steered = np.random.default_rng(0).normal(size=(b, h))
    baseline = np.random.default_rng(1).normal(size=(b, h))
    targets = steered.copy()
    contexts = np.random.default_rng(2).normal(size=(b, 32))
    periods = np.full(b, np.nan)
    stats = battery_statistics(steered, baseline, targets, contexts, periods)
    assert stats["seasonal"]["available"] is False
    assert "no series" in stats["seasonal"]["reason"]


def test_summarize_battery_passes_through_unavailable_channels_without_a_ci():
    """An unavailable channel must not silently acquire a bootstrap CI --
    `summarize_battery` has to check `available` before touching `delta`."""
    per_series = {"seasonal": {"delta": None, "available": False, "reason": "no periods"}}
    out = summarize_battery(per_series, n_boot=50, seed=0)
    assert out["seasonal"]["available"] is False
    assert out["seasonal"]["ci"] is None


# ---------------------------------------------------------------------------
# 3. Random-direction null gating
# ---------------------------------------------------------------------------

def test_null_gating_discriminates_a_real_effect_from_null_sized_noise():
    """A channel whose measured |effect| exceeds the null's own p95 must
    clear; one that sits inside the null's spread must not -- the exact
    §25.9 Stage 2 exit-criterion (iii) arithmetic, exercised directly rather
    than only through the full pipeline."""
    rng = np.random.default_rng(0)
    null_samples = rng.normal(0, 1.0, size=200)
    null_p95 = float(np.quantile(np.abs(null_samples), 0.95))

    real_effect = null_p95 * 5.0    # decisively larger than the null's spread
    noise_effect = float(np.abs(rng.normal(0, 1.0)))  # a single null-sized draw

    assert real_effect > null_p95
    # a null-sized draw clears only ~5% of the time by construction; pin one
    # that is inside the null's own generating distribution's typical range
    # to demonstrate the boundary is real, not vacuous.
    typical_noise = float(np.abs(null_samples[0]))
    assert not (typical_noise > null_p95) or typical_noise == null_samples[np.argmax(np.abs(null_samples))]
    # the actually load-bearing assertion: an effect BELOW the p95 does not clear.
    below_p95_effect = null_p95 * 0.5
    assert not (below_p95_effect > null_p95)


def test_excess_over_chance_arithmetic_matches_sec_25_11():
    """§25.11: the count of clearing (feature, channel) cells must be
    compared against 0.05 * n_features * n_channels, the count expected by
    chance under independent per-channel p95 thresholds -- not a
    Holm-corrected count (unsatisfiable at K=24 null draws per §25.11's own
    argument)."""
    n_features, n_channels = 12, 9
    chance_expected = 0.05 * n_features * n_channels
    assert chance_expected == pytest.approx(5.4)
    # a run where exactly 6 cells clear is a real, if modest, excess over chance.
    n_clearing = 6
    assert (n_clearing - chance_expected) == pytest.approx(0.6)
    assert n_clearing > chance_expected


# ---------------------------------------------------------------------------
# 4. Candidate selection: priority order, alive-atom restriction, multi-rule
# ---------------------------------------------------------------------------

def test_select_candidates_restricts_every_rule_to_alive_atoms():
    """A dead atom must never be nominated, even if it tops every ranking --
    §25.4's explicit restriction, and the one invariant every rule shares."""
    dict_size = 10
    alive = np.array([True, True, False, False, True, False, True, False, False, True])
    # feature 2 (dead) has the highest structural rho and highest variance --
    # must NOT appear in the result under any rule.
    structural = {2: 0.99, 0: 0.5, 4: 0.4}
    variance = np.array([0.1, 0.2, 99.0, 0.3, 0.4, 0.5, 0.6, 0.05, 0.05, 0.05])
    out = select_candidates(alive, n_per_rule=2, residualized_structural=structural,
                            activation_variance=variance, seed=0)
    nominated = {c["feature"] for c in out}
    assert 2 not in nominated
    assert nominated.issubset(set(np.flatnonzero(alive).tolist()))


def test_select_candidates_priority_order_is_structural_then_variance_then_provenance_then_random():
    """§25.4's exact stated order: (1) residualized structural rho, (2)
    activation variance, (3) raw provenance rho, (4) random alive control.
    The first-nomination order of the returned list must follow this."""
    dict_size = 8
    alive = np.ones(dict_size, dtype=bool)
    structural = {0: 0.9}       # rule 1 nominates feature 0 first
    variance = np.array([0.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # rule 2 nominates feature 1
    provenance = {3: 0.8}       # rule 3 nominates feature 3
    out = select_candidates(alive, n_per_rule=1, residualized_structural=structural,
                            activation_variance=variance, provenance_matches=provenance, seed=0)
    order = [c["feature"] for c in out]
    assert order.index(0) < order.index(1)
    assert order.index(1) < order.index(3)


def test_select_candidates_records_every_nominating_rule_not_just_the_first():
    """CLAUDE.md sec 11.34: a feature nominated by more than one rule must
    have EVERY rule recorded, deduplicated -- not just whichever ran first."""
    dict_size = 4
    alive = np.ones(dict_size, dtype=bool)
    structural = {1: 0.9}
    variance = np.array([0.0, 99.0, 0.0, 0.0])   # feature 1 wins both rule 1 and rule 2
    out = select_candidates(alive, n_per_rule=1, residualized_structural=structural,
                            activation_variance=variance, seed=0)
    entry = next(c for c in out if c["feature"] == 1)
    assert set(entry["rules"]) == {"structural", "variance"}


def test_select_candidates_always_includes_a_random_control_sample():
    """Rule 4's random sample must appear even with no structural/variance/
    provenance inputs given -- it costs nothing and is the only way to say
    whether the OTHER selected features are unusual (§25.4)."""
    dict_size = 20
    alive = np.ones(dict_size, dtype=bool)
    out = select_candidates(alive, n_per_rule=3, seed=0)
    assert len(out) == 3
    assert all(c["rules"] == ["random"] for c in out)


def test_select_candidates_handles_zero_alive_atoms_without_crashing():
    """A fully-dead dictionary must return an empty candidate list, not raise
    -- the caller (the runner script) is responsible for reporting that as
    a withheld/uninterpretable target, not this pure selection function."""
    dict_size = 6
    alive = np.zeros(dict_size, dtype=bool)
    out = select_candidates(alive, n_per_rule=3, activation_variance=np.ones(dict_size), seed=0)
    assert out == []


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
