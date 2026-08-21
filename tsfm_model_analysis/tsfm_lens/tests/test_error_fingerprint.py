"""Idiosyncratic-error fingerprinting (`ROADMAP.md` sec 6.3.1 Option C).

All synthetic, with planted answers. The load-bearing tests are the two
guards, not the happy path: that the difficulty adjustment refuses to call
itself an adjustment when its basis explains nothing out of fold, and that
residuals are genuinely cross-fitted -- an in-sample residual is shrunk by
whatever the ridge overfit, and that shrinkage is *shared across models*,
which would inject a spurious common component into the exact quantity being
correlated.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.error_fingerprint import (_ALPHA_GRID, _corr,
                                                  _cross_fitted_residuals,
                                                  _oof_r2, difficulty_features,
                                                  error_fingerprint)


class _FakeStore:
    """Stands in for ActivationStore: only `load_predictions` is ever read."""

    def __init__(self, preds):
        self._p = preds

    def load_predictions(self, model):
        return {"point": self._p[model]}


def _corpus(n=240, t=128, h=16, seed=0):
    rng = np.random.default_rng(seed)
    period = rng.integers(6, 20, n)
    amp = np.exp(rng.normal(0, 1.0, n))
    phase = rng.uniform(0, 2 * np.pi, n)
    full = np.arange(t + h)[None, :] * (2 * np.pi / period[:, None]) + phase[:, None]
    series = amp[:, None] * np.sin(full) + rng.normal(0, 0.1, (n, t + h)) * amp[:, None]
    return series[:, :t], series[:, t:]


def test_planted_shared_quirk_separates_from_independent_quirks():
    contexts, targets = _corpus()
    n, h = targets.shape
    rng = np.random.default_rng(1)
    # Difficulty is real and shared by everyone: a per-series error scale.
    hard = np.exp(rng.normal(0, 0.6, n))[:, None]
    quirk = rng.normal(0, 1.0, (n, h))
    shared_a = targets + hard * (quirk + 0.4 * rng.normal(0, 1, (n, h)))
    shared_b = targets + hard * (quirk + 0.4 * rng.normal(0, 1, (n, h)))
    indep_a = targets + hard * rng.normal(0, 1, (n, h))
    indep_b = targets + hard * rng.normal(0, 1, (n, h))

    store = _FakeStore({"sa": shared_a, "sb": shared_b, "ia": indep_a, "ib": indep_b})
    shared = error_fingerprint(store, "sa", "sb", contexts, targets, n_boot=200)
    indep = error_fingerprint(store, "ia", "ib", contexts, targets, n_boot=200)

    assert shared["shape"]["residual_corr"] > 0.6
    assert abs(indep["shape"]["residual_corr"]) < 0.15
    # And the CIs must not overlap -- a point-estimate gap alone would not
    # support the comparison this method exists to make.
    assert indep["shape"]["residual_ci"]["hi"] < shared["shape"]["residual_ci"]["lo"]


def test_adjustment_is_flagged_when_the_difficulty_basis_explains_nothing():
    contexts, targets = _corpus(seed=3)
    n, h = targets.shape
    rng = np.random.default_rng(4)
    # Difficulty must be absent IN THE UNITS THE FINGERPRINT USES. Errors are
    # divided by each series' own MASE scale, so a raw error whose size is
    # proportional to that scale leaves a scaled error whose magnitude varies
    # only with an independent draw -- nothing in the basis can predict it.
    # (The first version of this test planted a *constant* raw error, which
    # after scaling is exactly 1/scale and therefore maximally predictable
    # from `log_scale`: the plant demonstrated the opposite of its name.)
    scale = np.abs(np.diff(contexts, axis=1)).mean(1)[:, None]
    unrelated = np.exp(rng.normal(0, 0.8, n))[:, None]
    store = _FakeStore({
        "a": targets + scale * unrelated * rng.normal(0, 1, (n, h)),
        "b": targets + scale * unrelated * rng.normal(0, 1, (n, h)),
    })
    out = error_fingerprint(store, "a", "b", contexts, targets, n_boot=100)
    assert out["adjustment_ok"] is False
    assert all(v <= 0.0 for v in out["difficulty_oof_r2"].values())


def test_difficulty_basis_actually_explains_a_difficulty_driven_error():
    contexts, targets = _corpus(seed=5)
    n, h = targets.shape
    rng = np.random.default_rng(6)
    # A fixed-size raw error: in MASE units this is 1/scale, which `log_scale`
    # predicts well, so the adjustment is real and must say so.
    store = _FakeStore({"a": targets + rng.normal(0, 1, (n, h)),
                        "b": targets + rng.normal(0, 1, (n, h))})
    out = error_fingerprint(store, "a", "b", contexts, targets, n_boot=50)
    assert out["adjustment_ok"] is True
    assert min(out["difficulty_oof_r2"].values()) > 0.3


def test_residuals_are_cross_fitted_not_in_sample():
    # More features than series: an in-sample ridge fit at a small penalty
    # drives residuals to ~0 regardless of any real relationship. Cross
    # fitting must not.
    rng = np.random.default_rng(7)
    n = 40
    feats = rng.normal(size=(n, 200))
    y = rng.normal(size=n)
    oof = _cross_fitted_residuals(feats, y, n_folds=5, alpha=1e-3, seed=0)
    assert np.var(oof) > 0.5 * np.var(y)
    assert _oof_r2(feats, y, 5, 1e-3, 0) < 0.2


def test_alpha_grid_search_can_reach_a_positive_oof_r2_on_a_linear_signal():
    rng = np.random.default_rng(8)
    n = 200
    feats = rng.normal(size=(n, 5))
    y = feats @ np.array([1.0, -0.5, 0.25, 0.0, 0.0]) + rng.normal(0, 0.3, n)
    best = max(_ALPHA_GRID, key=lambda a: _oof_r2(feats, y, 5, a, 0))
    assert _oof_r2(feats, y, 5, best, 0) > 0.7


def test_corr_is_zero_for_a_constant_side_rather_than_nan():
    assert _corr(np.ones(10), np.arange(10.0)) == 0.0


def test_difficulty_features_are_finite_and_named():
    contexts, targets = _corpus(seed=9)
    scale = np.abs(np.diff(contexts, axis=1)).mean(1)
    feats, names = difficulty_features(contexts, targets, scale)
    assert feats.shape == (contexts.shape[0], len(names))
    assert np.isfinite(feats).all()
    # A zero-variance series must not produce NaN via a divide-by-zero.
    flat_c = np.zeros((4, contexts.shape[1]))
    flat_t = np.zeros((4, targets.shape[1]))
    f2, _ = difficulty_features(flat_c, flat_t, np.zeros(4))
    assert np.isfinite(f2).all()


def test_fingerprint_is_invariant_to_per_series_amplitude_rescaling():
    # Errors are divided by each series' own MASE scale, so multiplying a
    # series by a constant must not move the fingerprint -- otherwise the
    # correlation would partly measure which corpus happens to have wider
    # amplitude spread.
    contexts, targets = _corpus(seed=11)
    n, h = targets.shape
    rng = np.random.default_rng(12)
    ea, eb = rng.normal(0, 1, (n, h)), rng.normal(0, 1, (n, h))
    base = _FakeStore({"a": targets + ea, "b": targets + eb})
    plain = error_fingerprint(base, "a", "b", contexts, targets, n_boot=50)

    g = np.exp(rng.normal(0, 1.0, n))[:, None]
    scaled = _FakeStore({"a": (targets + ea) * g, "b": (targets + eb) * g})
    rescaled = error_fingerprint(scaled, "a", "b", contexts * g, targets * g, n_boot=50)
    assert rescaled["shape"]["residual_corr"] == pytest.approx(
        plain["shape"]["residual_corr"], abs=0.05)


def test_near_zero_scale_series_are_excluded_and_do_not_inflate_the_correlation():
    """ROADMAP.md sec 6.3.1 Option E's lineage-pair run: a handful of
    near-flat-context series turn any nonzero error into a multi-million-unit
    scaled value once divided by the (floored) near-zero MASE denominator.
    Two models with genuinely UNCORRELATED errors on the real series can
    still look near-perfectly correlated once a few such outliers dominate
    the sum -- unless they are excluded first, the way L0's own
    `mase_reliable` already excludes them from its aggregate MASE.

    Uses a constant-scale (not `_corpus`'s heavy-tailed amplitude) clean
    corpus deliberately: `_corpus`'s own varying amplitude is itself a
    shared-difficulty confound the residual step is designed to remove, and
    would muddy this test's actual target -- outlier-driven RAW correlation
    inflation -- with that unrelated, already-covered mechanism.
    """
    rng = np.random.default_rng(21)
    n, t, h = 200, 64, 16
    contexts = rng.normal(0, 1.0, (n, t))
    targets = rng.normal(0, 1.0, (n, h))
    # The real, non-degenerate series: independent errors, constant scale --
    # genuinely no relationship between the two models' magnitude/shape.
    ea = rng.normal(0, 1, (n, h))
    eb = rng.normal(0, 1, (n, h))

    # Append k near-flat-context series (diff ~ 0 -> scale hits the 1e-8
    # floor) where both models make the SAME (but otherwise unremarkable)
    # absolute error -- exactly the "two lightly-diverged children of one
    # parent" scenario that surfaced this.
    k = 8
    flat_contexts = np.full((k, t), 5.0) + rng.normal(0, 1e-10, (k, t))
    flat_targets = np.full((k, h), 5.0)
    shared_abs_error = rng.normal(0, 0.5, (k, h))

    all_contexts = np.concatenate([contexts, flat_contexts])
    all_targets = np.concatenate([targets, flat_targets])
    a_full = np.concatenate([targets + ea, flat_targets + shared_abs_error])
    b_full = np.concatenate([targets + eb, flat_targets + shared_abs_error])
    store = _FakeStore({"a": a_full, "b": b_full})

    filtered = error_fingerprint(store, "a", "b", all_contexts, all_targets, n_boot=200)
    unfiltered = error_fingerprint(store, "a", "b", all_contexts, all_targets, n_boot=200,
                                   min_scale_frac=0.0)

    assert filtered["n_excluded_unreliable"] == k
    assert unfiltered["n_excluded_unreliable"] == 0
    # With no shared-scale confound in the clean 200 series, the filtered
    # fingerprint correctly reads low (sampling noise at n=200, nowhere near
    # the near-perfect outlier-dominated value below)...
    assert abs(filtered["magnitude"]["raw_corr"]) < 0.3
    assert abs(filtered["shape"]["raw_corr"]) < 0.3
    # ...while the unfiltered one is dominated by the bit-identical outlier
    # values and reads as a near-perfect, entirely spurious correlation.
    assert unfiltered["magnitude"]["raw_corr"] > 0.9
    assert unfiltered["shape"]["raw_corr"] > 0.9


def test_min_scale_frac_zero_reproduces_no_filtering_default_excludes_nothing_when_clean():
    """A corpus with no degenerate series must be untouched by the new
    filter at its default threshold -- the fix must not move any
    already-recorded number on a healthy corpus."""
    contexts, targets = _corpus(seed=22)
    n, h = targets.shape
    rng = np.random.default_rng(23)
    store = _FakeStore({"a": targets + rng.normal(0, 1, (n, h)),
                        "b": targets + rng.normal(0, 1, (n, h))})
    out = error_fingerprint(store, "a", "b", contexts, targets, n_boot=50)
    assert out["n_excluded_unreliable"] == 0
