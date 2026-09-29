"""A skip lens is only meaningful if the forecast head reads what patching writes.

Chronos-2 slices `hidden_states[:, -num_output_patches:]` into its forecast
head, so the context-patch positions `token_slice` writes are never read.
Patching them moved the forecast by exactly 0.0 at every layer, which the
report rendered as a perfectly flat depth curve and a crystallization depth
of 0.0 at all 64 horizon steps -- a no-op presented as a finding.

These tests pin the declaration, the withholding it drives, and -- the
load-bearing negative -- that withholding the skip lens does NOT withhold the
tuned lens, which uses no patching and stays valid for such a model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsfm_lens.models.base import ModelAdapter

ROOT = Path(__file__).resolve().parents[1]


class _Bare(ModelAdapter):
    """Minimal concrete adapter; only the default we care about is exercised."""

    def load(self):  # pragma: no cover - never called
        raise NotImplementedError

    def predict(self, contexts, horizon, quantiles):  # pragma: no cover
        raise NotImplementedError


def test_the_default_is_that_the_head_reads_patched_positions():
    """A model whose forecast continues its context necessarily reads it."""
    assert _Bare.forecast_reads_patched_positions(object.__new__(_Bare)) is True


def test_every_registered_adapter_answers_the_question():
    """The flag must be a real boolean everywhere -- a missing override that
    returned None would be falsy and silently withhold a working skip lens."""
    from tsfm_lens import models as m

    for name, cls in sorted(m.ADAPTERS.items()):
        val = cls.forecast_reads_patched_positions(object.__new__(cls))
        assert isinstance(val, bool), f"{name} returned {val!r}, not a bool"


def test_chronos2_declares_that_its_head_does_not_read_patched_positions():
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter

    assert Chronos2Adapter.forecast_reads_patched_positions(
        object.__new__(Chronos2Adapter)) is False


def test_a_flat_skip_curve_is_what_the_bug_looked_like():
    """The regression's signature, so its shape is documented in a test.

    Every layer returning the identical forecast makes crystallization depth
    0.0 at every horizon step -- the first layer is trivially 'within
    tolerance of final' because it IS final. That is indistinguishable from a
    real finding by value alone, which is why it is gated on a declaration
    rather than detected from the curve.
    """
    from tsfm_lens.analysis.lens import crystallization_depths

    n_layers, horizon = 12, 8
    flat = np.repeat(np.full((1, horizon), 1.56, np.float32), n_layers, axis=0)
    depths = np.linspace(0.0, 1.0, n_layers)
    got = crystallization_depths(flat, flat[-1], 0.1, depths)
    assert got == [0.0] * horizon


def test_a_genuine_curve_does_not_crystallize_at_zero():
    """The contrast case: a curve that actually improves with depth."""
    from tsfm_lens.analysis.lens import crystallization_depths

    horizon = 4
    curve = np.array([[3.0], [2.5], [2.0], [1.6], [1.5]], np.float32)
    curve = np.repeat(curve, horizon, axis=1)
    depths = np.linspace(0.0, 1.0, curve.shape[0])
    got = crystallization_depths(curve, curve[-1], 0.1, depths)
    assert all(g is not None and g > 0.0 for g in got)


def test_the_whole_sequence_widening_and_its_norm_fix_are_both_recorded_as_rejected():
    """The two fixes tried for Chronos-2's withheld skip lens, and why each failed.

    A future session reading "skip lens withheld" will reach for the same two
    ideas in the same order: patch the whole live sequence instead of the
    context span, then -- on seeing the resulting curve wander -- blame
    activation-norm growth and rescale. Both were measured. The first produces
    a non-monotonic curve, which answers a different question than the one a
    skip lens asks. The second changes that curve by <=0.0002 MASE at any
    block and is therefore REFUTED, despite the norm growth it appeals to
    (~25x across depth) being entirely real.

    Pinned as a text assertion rather than a numeric one on purpose: there is
    no checkpoint in the test environment, so the alternative is no guard at
    all, and the failure mode this prevents is someone deleting the refutation
    while re-proposing the fix it refutes.
    """
    src = (ROOT / "tsfm_lens" / "models" / "chronos2_adapter.py").read_text(encoding="utf-8")
    doc = src.split("def forecast_reads_patched_positions")[1].split("return False")[0]

    assert "non-monotonic" in doc, "the whole-sequence curve's actual defect must be stated"
    low = doc.lower()
    assert "refuted" in low, "the norm-mismatch mechanism must be marked refuted, not merely dropped"
    assert "0.0002" in doc, "the refutation needs its measured magnitude, not just the verdict"
    # The norm growth is real and must stay quotable -- what changed is its
    # causal status. A correction that deleted the number would lose evidence.
    assert "25x" in doc or "25×" in doc, "the measured norm growth stays on record"


def test_the_lens_warning_does_not_assert_the_refuted_mechanism():
    """The WARNING is the only one of the three statements a user actually sees.

    It named activation-norm mismatch as the reason widening fails. That claim
    is refuted, so it must not survive here -- a stale reason in the one
    user-visible surface is worse than in a docstring.
    """
    src = (ROOT / "tsfm_lens" / "analysis" / "lens.py").read_text(encoding="utf-8")
    warn = src.split("skip lens withheld")[1].split("adapter.name)")[0]

    assert "activation-norm mismatch rather than by what the layer encodes" not in warn
    assert "non-monotonic" in warn, "state the measured defect"
    assert "norm-matching" in warn or "norm-match" in warn, (
        "the WARNING should preempt the rescaling fix, since that is the next "
        "thing a reader will try")
