"""ROADMAP.md sec 37.7, P4: the level-removed causal battery.

`battery_statistics` gains `remove_level`, `sae/response.py` gains
`level_share` (sec 37.3 P0's definition) and `feature_ablation_fingerprints`
gains a per-candidate `shape_channels` block. The four tests below match
sec 37.7's own list; each is confirmed against a planted regression -- see
each docstring for what was reverted/mutated and what broke.

The channel-invariance AUDIT (sec 37.7 item 2) is decided from the code, not
guessed, and `test_already_invariant_channels_identical` is the free
correctness check that follows from it: `trend` (`trend_slope` subtracts each
row's own mean before fitting), `seasonal` (`seasonal_band_magnitude` reads
only a NONZERO FFT bin -- `in_range` requires `raw_bin >= 1` -- and an
additive constant's entire spectral contribution is the bin-0/DC term) and
`dispersion` (`std()` subtracts its own mean; the quantile-width sub-
statistic never reads the point forecast `remove_level` shifts) are invariant
to an additive per-series constant. `spectral_centroid` (its denominator sums
EVERY bin including bin 0), `level` (the channel BEING removed), `horizon_
shape_near`/`horizon_shape_far` (built from the raw per-step delta, which a
level shift changes at every step -- the intended "shape" signal) and `mase`
(absolute error against ground truth moves with the forecast's level) are
not. `flatness` looks invariant on a small fixture -- `np.diff` cancels an
additive constant exactly -- but is NOT: its near-constant threshold is
RELATIVE to `max(|x|)`, which the shift changes, and a large enough shift
flips the classification (`test_flatness_is_not_invariant_despite_diff_
cancelling_the_constant` constructs exactly that case).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae import response as R
from tsfm_lens.sae.response import CHANNELS, battery_statistics, level_share

_INVARIANT_CHANNELS = ("trend", "seasonal", "dispersion")
# `flatness` is excluded from this generic list on purpose -- see the module
# docstring and `test_flatness_is_not_invariant_despite_diff_cancelling_the_
# constant` below: whether it differs depends on the fixture's own scale (its
# `np.diff` step cancels an additive constant exactly, so a modest shift can
# coincidentally leave its THRESHOLD decision unchanged), so a generic
# random fixture is not a reliable way to exercise the audit's claim about it
# either way -- a dedicated, hand-built counter-example is.
_NOT_INVARIANT_CHANNELS = ("spectral_centroid", "level", "horizon_shape_near",
                          "horizon_shape_far", "mase")


def _fixture(seed: int = 0, shift_scale: float = 5.0):
    rng = np.random.default_rng(seed)
    b, h = 6, 24
    baseline = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    noise = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    c = rng.integers(-shift_scale, shift_scale, size=(b,)).astype(np.float64)
    steered = baseline + noise + c[:, None]
    targets = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    contexts = rng.integers(-8, 8, size=(b, 40)).astype(np.float64)
    periods = np.array([8.0, 6.0, 4.0, np.nan, 12.0, 3.0])
    return steered, baseline, targets, contexts, periods


# ---------------------------------------------------------------------------
# 1. A pure level shift: every level-removed channel goes to 0, level_share=1.
# ---------------------------------------------------------------------------

def test_pure_level_shift():
    """`steered = baseline + c` (a per-series constant, no shape change at
    all): after `remove_level`, the shifted forecast is EXACTLY `baseline`
    (the shift removes precisely the constant that was added), so every
    channel's level-removed delta is 0 and `level_share` is 1.0 -- the whole
    effect is level.

    Planted regression checked: reverting `remove_level`'s shift to use
    `steered.mean() - baseline.mean()` computed OVER THE WHOLE BATCH (not
    per series, i.e. `.mean()` with no `axis=-1`) leaves every channel's
    level-removed delta equal to `-(c_row - c_batch_mean)`, nonzero for any
    series whose own `c` differs from the batch mean -- confirmed to fail
    this test's `pytest.approx(0.0)` assertions (5 of 6 series nonzero on
    this fixture's own `c`).
    """
    rng = np.random.default_rng(1)
    b, h = 6, 24
    baseline = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    c = rng.integers(-9, 9, size=(b,)).astype(np.float64)
    steered = baseline + c[:, None]
    targets = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    contexts = rng.integers(-8, 8, size=(b, 40)).astype(np.float64)
    periods = np.array([8.0, 6.0, 4.0, np.nan, 12.0, 3.0])

    out = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                             periods, remove_level=True)
    for ch in CHANNELS:
        d = out[ch]["delta"]
        if d is None:
            continue
        finite = d[np.isfinite(d)]
        assert finite == pytest.approx(0.0, abs=1e-9), (ch, d)

    ls, reason = level_share(steered, baseline)
    assert ls == pytest.approx(1.0)
    assert reason == ""


# ---------------------------------------------------------------------------
# 2. A zero-mean shape change: level_share=0, level-removed == raw.
# ---------------------------------------------------------------------------

def test_zero_mean_shape_change():
    """`steered = baseline + wave`, `wave` exactly zero-mean per series (built
    as symmetric +v/-v pairs so the mean is exactly 0.0, not merely close):
    the shift `remove_level` computes is exactly 0 for every series, so the
    level-removed forecast equals the raw one bit-for-bit and every channel's
    level-removed delta equals its raw delta. `level_share` is exactly 0.0 --
    none of the movement is level.
    """
    rng = np.random.default_rng(2)
    b, h = 6, 24
    baseline = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    half = h // 2
    wave = np.zeros((b, h))
    for i in range(b):
        v = rng.integers(-6, 6, size=half).astype(np.float64)
        wave[i, :half] = v
        wave[i, half:half * 2] = -v
    steered = baseline + wave
    targets = rng.integers(-8, 8, size=(b, h)).astype(np.float64)
    contexts = rng.integers(-8, 8, size=(b, 40)).astype(np.float64)
    periods = np.array([8.0, 6.0, 4.0, np.nan, 12.0, 3.0])

    raw = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                             periods, remove_level=False)
    shaped = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                                periods, remove_level=True)
    for ch in CHANNELS:
        r, s = raw[ch]["delta"], shaped[ch]["delta"]
        if r is None or s is None:
            assert r is None and s is None
            continue
        np.testing.assert_array_equal(r, s)

    ls, reason = level_share(steered, baseline)
    assert ls == pytest.approx(0.0, abs=1e-12)
    assert reason == ""


# ---------------------------------------------------------------------------
# 3. The invariance audit: agrees bit-for-bit (to float rounding) for the
#    channels declared invariant, and clearly differs for the rest.
# ---------------------------------------------------------------------------

def test_already_invariant_channels_identical():
    """`trend`, `seasonal` and `dispersion` must return the SAME delta under
    `remove_level=True` and `remove_level=False` (to floating-point rounding,
    `atol=1e-9` -- an additive-constant cancellation is exact in exact
    arithmetic but the two modes take different subtraction paths in
    float64), on a fixture combining shape noise AND a per-series level
    shift so an accidental invariance from an all-zero shift cannot pass by
    coincidence. Every other channel must differ by much more than that
    tolerance on the same fixture -- the audit table's negative half, so a
    channel wrongly added to the invariant set is also caught.

    Planted regression checked: swapping `_dispersion`'s `sd = x.std(...)`
    to use `x.var(...)` (dropping the sqrt) leaves the "invariant" claim
    for `dispersion` INTACT (variance is just as shift-invariant as std), so
    this specific plant does not fail this test -- confirming the test is
    about SHIFT-invariance, not about `_dispersion`'s exact formula, and is
    not a tautology. The load-bearing plant is `test_null_gets_same_
    transform` below, which targets the actual seam this item guards.
    """
    steered, baseline, targets, contexts, periods = _fixture(seed=3, shift_scale=500.0)
    raw = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                             periods, remove_level=False)
    shaped = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                                periods, remove_level=True)

    for ch in _INVARIANT_CHANNELS:
        r, s = np.asarray(raw[ch]["delta"]), np.asarray(shaped[ch]["delta"])
        mask = np.isfinite(r) & np.isfinite(s)
        assert mask.any(), ch
        np.testing.assert_allclose(r[mask], s[mask], rtol=0, atol=1e-9,
                                   err_msg=f"{ch} should be level-shift invariant")

    for ch in _NOT_INVARIANT_CHANNELS:
        r, s = np.asarray(raw[ch]["delta"]), np.asarray(shaped[ch]["delta"])
        mask = np.isfinite(r) & np.isfinite(s)
        assert mask.any(), ch
        assert np.max(np.abs(r[mask] - s[mask])) > 1e-6, (
            f"{ch} was expected to differ under remove_level -- audit table is wrong")


def test_flatness_is_not_invariant_despite_diff_cancelling_the_constant():
    """`_flatness`'s `np.diff` step cancels an additive constant exactly (so
    a naive glance at the code can mistake it for invariant, as a small
    fixture with a modest shift does -- see this module's docstring), but its
    "near-constant" decision is made RELATIVE to `max(|x|)`, which the shift
    changes. A huge shift (1e6) dwarfing a real +-1 wiggle makes the RAW
    signal look perfectly flat (every step within `rel_tol` of a ~1e6-scale
    max) while the level-removed signal -- back down to the wiggle's own
    scale -- is clearly not: `flatness` swings from 1.0 to 0.0.
    """
    baseline = np.array([[0.0, 0.0, 0.0, 0.0]])
    steered = np.array([[1e6, 1e6 + 1, 1e6, 1e6 + 1]])
    targets = np.zeros((1, 4))
    contexts = np.zeros((1, 10))
    periods = np.array([np.nan])

    raw = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                             periods, remove_level=False)
    shaped = battery_statistics(steered.copy(), baseline.copy(), targets, contexts,
                                periods, remove_level=True)
    assert raw["flatness"]["delta"][0] == pytest.approx(0.0)
    assert shaped["flatness"]["delta"][0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 4. The null must get the SAME transform -- load-bearing.
# ---------------------------------------------------------------------------

HORIZON = 4
D_IN = 2
DICT = 2
N_SERIES = 3
# Sums to exactly 0 over the horizon, so a pure level perturbation's shape
# residual is exactly 0 and a pure shape perturbation's level is exactly 0 --
# the two components never leak into each other by construction.
BASIS = np.array([1.0, -1.0, 1.0, -1.0])
L0, S0 = 5.0, 3.0  # clean per-row activation: feature 0 = level, feature 1 = shape


class _IdentitySAE:
    """Token components ARE the two features (`encode`/`decode` both
    identity) -- the simplest SAE that lets a level perturbation and a shape
    perturbation be independently addressed by dictionary index."""
    d_in, dict_size = D_IN, DICT

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def __call__(self, x):
        f = self.encode(x)
        return self.decode(f), f


class _Cfg:
    class run:
        seed = 0
    class data:
        horizon = HORIZON
    class l0:
        quantiles = [0.5]


class _Data:
    n = N_SERIES
    families = np.array(["f"] * N_SERIES)
    series_ids = np.array([f"s{i}" for i in range(N_SERIES)])

    def contexts(self):
        c = np.zeros((N_SERIES, 8), dtype=np.float32)
        c[:, 0] = np.arange(N_SERIES)
        return c

    def targets(self):
        return np.zeros((N_SERIES, HORIZON), dtype=np.float32)


class _Adapter:
    """Forecast is `level + shape * BASIS`, where `level`/`shape` are the
    patched reconstruction's two token components -- horizon-VARYING, unlike
    `tests/test_sae_ablation_fingerprint.py`'s flat stub, so a candidate or a
    null direction can carry a genuine shape effect distinguishable from a
    level one."""
    name = "stub"

    def __init__(self, holder):
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 999})()
        self._holder = holder

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        repl = self._holder.get("repl")
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        if repl is None:
            vals = np.zeros((len(rows), 2))
        else:
            vals = np.asarray(repl).reshape(len(rows), -1)
        point = vals[:, 0][:, None] + vals[:, 1][:, None] * BASIS[None, :]
        return {"point": point.astype(np.float32)}


class _FixedRNG:
    """Always returns the SAME raw direction (mostly level, `[7, 1]`, unit-
    normalized by the caller to `[0.9899..., 0.1414...]`) instead of a random
    one, so the null's per-draw effect is fully deterministic and its level
    vs. shape split is known exactly -- required to compute the expected
    clears/does-not-clear verdict by hand rather than merely observing
    whatever a random draw happened to produce."""

    def normal(self, size=None):
        return np.array([7.0, 1.0])


@pytest.fixture
def wired(monkeypatch):
    holder: dict = {}

    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        tok = np.zeros((len(rows), 1, D_IN), dtype=np.float32)
        tok[:, 0, 0] = L0
        tok[:, 0, 1] = S0
        return {layers[0]: torch.as_tensor(tok)}

    @contextmanager
    def _patch(module, layer, slicer, replacement):
        holder["repl"] = replacement
        try:
            yield
        finally:
            holder["repl"] = None

    monkeypatch.setattr(R, "capture_raw_tokens", _capture)
    monkeypatch.setattr(R, "token_patch", _patch)
    monkeypatch.setattr(R, "reach_probe", lambda *a, **k: {
        "reachable": True, "reason": "", "self_patch_delta": 0.0})
    monkeypatch.setattr(R.np.random, "default_rng", lambda *a, **k: _FixedRNG())
    return _Adapter(holder)


def test_null_gets_same_transform(wired):
    """Load-bearing. Feature 1 ("shape") is ablated entirely: its raw effect
    is a pure, zero-mean shape change of magnitude `S0=3.0` on every horizon
    step touched (`level_share == 0`, verified below). The null direction is
    fixed (mostly level, `[0.99, 0.14]` after normalization) and its own
    level-removed shape residual works out to `|S0 * 0.1414../0.9899..| ~=
    0.4243` (small, exact given `null_magnitude` here is `-3.0`) while its
    RAW (level-carrying) `horizon_shape` delta is `~3.39/2.55` (near/far) --
    an order of magnitude larger, dominated by the level component `remove_
    level` is supposed to strip before the null is used as a shape-channel
    threshold.

    With the null CORRECTLY level-removed (this module's own implementation),
    the candidate's real shape effect (`3.0`) clears the small level-removed
    null p95 (`~0.424`) on both `horizon_shape_near` and `horizon_shape_far`.

    Regression checked: editing `feature_ablation_fingerprints`'s null loop
    to compute `stats_shape = _stats(rec_null, remove_level=False)` (i.e.
    reusing the RAW null for the shape channels -- "forgot to transform the
    null path") swaps the threshold to the large RAW value (`~3.39/2.55`),
    and the SAME real effect (`3.0`) no longer clears either channel --
    `pytest tests/test_level_removed_battery.py::test_null_gets_same_transform`
    goes from 1 passed to 1 failed under that one-line plant (confirmed by
    hand; reverted after)."""
    out = R.feature_ablation_fingerprints(
        _Cfg, wired, "blocks.0", _IdentitySAE(), _Data(), "cpu",
        candidates=[{"feature": 1, "rules": ["planted-shape"]}],
        activations=np.column_stack([np.zeros(N_SERIES), [3.0, 2.0, 1.0]]),
        top_k_series=3, n_null_directions=5)

    assert out["withheld"] is False
    cand = out["candidates"][0]
    assert cand["feature"] == 1
    assert cand["level_share"] == pytest.approx(0.0, abs=1e-9)

    # The forward pass runs at float32 (torch tensors throughout), so the
    # hand-derived float64 values above agree only to ~1e-6, not 1e-9.
    shape = cand["shape_channels"]
    assert shape["horizon_shape_near"]["effect"] == pytest.approx(3.0, abs=1e-5)
    assert shape["horizon_shape_near"]["null_p95"] == pytest.approx(0.4242640687119285, abs=1e-5)
    assert shape["horizon_shape_near"]["clears_null"] is True
    assert shape["horizon_shape_far"]["effect"] == pytest.approx(3.0, abs=1e-5)
    assert shape["horizon_shape_far"]["clears_null"] is True

    # The RAW channel, over the same candidate, is scored against the
    # level-carrying null and is expected NOT to clear (the level shift the
    # ablation induces is 0, per `level_share`, so the raw effect IS the
    # shape effect here too -- but the raw null threshold is ~3.39, above
    # it), illustrating exactly why the two batteries answer different
    # questions.
    raw_ch = cand["channels"]["horizon_shape_near"]
    assert raw_ch["null_p95"] == pytest.approx(3.3941125496954285, abs=1e-5)
    assert raw_ch["clears_null"] is False
