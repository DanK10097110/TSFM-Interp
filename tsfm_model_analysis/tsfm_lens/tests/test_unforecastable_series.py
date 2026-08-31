"""A future the context cannot anticipate, and how the report says so.

The measurement (`analysis/l3_perturbation.py::future_excursion`) and its one
rendering rule (`report/report.py::_excursion_clause`). Both exist because a
user read a case-study panel whose forecast was flat against a truth line that
spiked, and reasonably concluded the pipeline was plotting the wrong series.
It was not -- the stored target matches the corpus exactly -- so the gap was
never in the plumbing, only in the fact that nothing said which series are
unforecastable and which are simply forecast badly.

All synthetic, with planted answers: a burst whose size is constructed, and a
well-behaved series that must stay silent.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l3_perturbation import future_excursion  # noqa: E402
from tsfm_lens.report.report import _excursion_clause  # noqa: E402


def _series(rng, n=256, h=32, sd=1.0):
    return rng.normal(0.0, sd, size=n), rng.normal(0.0, sd, size=h)


def test_excursion_recovers_a_planted_burst_in_context_sd_units():
    """The value is a ratio past a known boundary, so a plant has an answer.

    Measured beyond the context's observed max, not from its mean, so the
    expected value is `(spike - ctx.max()) / ctx.std()` exactly.
    """
    rng = np.random.default_rng(0)
    ctx, tgt = _series(rng, sd=2.0)
    spike = ctx.max() + 8.0 * ctx.std()
    tgt[5] = spike
    want = (spike - ctx.max()) / ctx.std()
    got = future_excursion(ctx[None, :], tgt[None, :])[0]
    assert np.isclose(got, want), (got, want)
    assert 7.5 < got < 8.5, got


def test_excursion_is_scale_free():
    """Multiplying a whole series by 100 cannot change how surprising it is."""
    rng = np.random.default_rng(1)
    ctx, tgt = _series(rng)
    a = future_excursion(ctx[None, :], tgt[None, :])[0]
    b = future_excursion(ctx[None, :] * 100.0, tgt[None, :] * 100.0)[0]
    assert np.isclose(a, b), (a, b)


def test_a_flat_context_is_nan_not_a_huge_number():
    """A zero-sd context would divide by zero and manufacture an excursion.

    This is the degenerate case the *existing* MASE guard already catches
    from the other direction; here it must not become a false positive.
    """
    ctx = np.zeros((1, 128))
    tgt = np.ones((1, 16))
    assert np.isnan(future_excursion(ctx, tgt)[0])
    assert _excursion_clause(future_excursion(ctx, tgt)[0]) == ""


def test_the_clause_is_silent_on_an_ordinary_series():
    """The load-bearing negative: a caveat on every panel is read by nobody.

    A routine series -- future inside the spread its context showed -- must
    render no clause at all, not a softened one.
    """
    rng = np.random.default_rng(2)
    ctx, tgt = _series(rng)
    exc = future_excursion(ctx[None, :], tgt[None, :])[0]
    assert exc < 3.0, exc
    assert _excursion_clause(exc) == ""


def test_the_clause_fires_and_states_the_size_when_the_future_leaves_the_range():
    rng = np.random.default_rng(3)
    ctx, tgt = _series(rng)
    tgt[0] = ctx.max() + 9.0 * ctx.std()
    exc = future_excursion(ctx[None, :], tgt[None, :])[0]
    out = _excursion_clause(exc)
    assert out, "a 9-sd excursion must be named"
    assert "9.0" in out, out
    assert "ceiling" in out, "it must say the error is a ceiling, not a failure"
    assert "clean-vs-corrupted" in out, "and point at the quantity under study"


def test_the_clause_is_missing_data_safe():
    """`None` is what a run without stored targets hands this."""
    assert _excursion_clause(None) == ""
    assert _excursion_clause(float("nan")) == ""


def test_excursion_is_per_series_not_pooled():
    """One bursty series in a batch must not flag its well-behaved neighbours."""
    rng = np.random.default_rng(4)
    ctx = rng.normal(0.0, 1.0, size=(3, 200))
    ctx = ctx - ctx.mean(axis=1, keepdims=True)
    tgt = rng.normal(0.0, 1.0, size=(3, 24))
    tgt[1, 3] = ctx[1].max() + 12.0 * ctx[1].std()
    got = future_excursion(ctx, tgt)
    assert got[1] > 8.0, got
    assert got[0] < 4.0 and got[2] < 4.0, got


def test_a_strong_seasonal_swing_is_not_an_excursion():
    """The load-bearing distinction: amplitude is not novelty.

    This is the defect the first implementation had. Measured from the
    context *mean*, a clean sinusoid's every peak sits several standard
    deviations out, so a perfectly ordinary seasonal series scores as
    unforecastable -- on this repo's own corpus that version flagged 21.9%
    of series where the range-based one flags 1.4%. A future that merely
    revisits the range its context already covered must score ~0.
    """
    t = np.arange(512)
    ctx = 10.0 * np.sin(2 * np.pi * t / 24.0)
    fut = 10.0 * np.sin(2 * np.pi * np.arange(512, 576) / 24.0)
    exc = future_excursion(ctx[None, :], fut[None, :])[0]
    assert exc < 0.1, f"a repeating seasonal peak is not an excursion: {exc}"
    assert _excursion_clause(exc) == ""
    # And the rejected definition would have called it one, decisively.
    from_mean = (np.abs(fut - ctx.mean()) / ctx.std()).max()
    assert from_mean > 1.3, from_mean


def test_a_future_inside_the_context_range_scores_zero_not_negative():
    """Clipped at 0: 'less extreme than the context' is not negative novelty."""
    rng = np.random.default_rng(5)
    ctx = rng.normal(0.0, 1.0, size=(1, 400))
    tgt = np.zeros((1, 32))
    assert future_excursion(ctx, tgt)[0] == 0.0
