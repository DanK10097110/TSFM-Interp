"""Tests for the impulse-alignment probe's self-calibrating amplitude
(ROADMAP.md sec 15 A20).

`impulse_alignment_check`'s original fixed impulse amplitude (0.25x the
probe signal's own amplitude) was calibrated once, against chronos-t5-small
at context_len 512, by directly diffing tokenized ids before/after the
perturbation (`CLAUDE.md` sec 11.16). Sec 15 A20 found that constant is not
safe at every (checkpoint, context_len) pair -- it silently under-calibrates
at others, letting a real span bug hide behind tokenizer-rescaling noise, or
(the more dangerous direction) letting the gate pass on noise it never
should have. `calibrate_impulse_amplitude` generalizes the original
diagnosis into an automatic sweep; these tests exercise the sweep logic
directly against a deterministic double, the true no-op path against the
mock adapters (which have no re-quantizing tokenizer at all), and that
`run_alignment_gate`'s artifact records what was chosen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.alignment import (
    calibrate_impulse_amplitude,
    impulse_alignment_check,
    run_alignment_gate,
)
from tsfm_lens.models.conformance import make_test_adapter
from tsfm_lens.models.mock import MockStepAdapter


class _RequantizingAdapter(MockStepAdapter):
    """Deterministic stand-in for a context-adaptive, globally-rescaling
    tokenizer (the real example is Chronos-T5's `MeanScaleUniformBins`, sec
    11.16). Any single-sample deviation from the exact clean probe signal
    whose magnitude exceeds `CONFOUND_THRESHOLD` (0.075, strictly between the
    0.10 and 0.05 calibration candidates) is treated as "the sequence's scale
    statistic moved" and shifts *every* token's id, not just the perturbed
    one -- reproducing sec 11.16's actual finding (472 of 513 tokens changed
    for a too-large impulse) in a form whose calibrated outcome is exactly
    predictable: since `base` is 0 at the probed sample (see
    `_clean_ref_for`), the injected deviation IS the tested amplitude, so
    0.25/0.15/0.10 must confound and 0.05/0.02 must not.
    """

    CONFOUND_THRESHOLD = 0.075
    # Class-level default: tests swap `__class__` onto an already-constructed
    # adapter (matching `test_alignment_gate.py`'s `_PermutedSpansAdapter`
    # pattern), which never re-runs `__init__` -- an instance default set
    # there would silently never apply. Each test sets this explicitly.
    _clean_ref = None

    def token_ids(self, prepared):
        x = prepared.detach().cpu().numpy()
        ids = np.round(x * 1000).astype(np.int64)
        if self._clean_ref is None:
            return ids
        deviation = np.abs(x - self._clean_ref[None, :]).max()
        return ids + 1 if deviation > self.CONFOUND_THRESHOLD else ids


def _clean_ref_for(context_len: int) -> np.ndarray:
    """Bit-identical to `calibrate_impulse_amplitude`'s internal `base`."""
    t = np.arange(context_len, dtype=np.float32)
    return np.sin(2 * np.pi * t / (context_len / 4)).astype(np.float32)


def test_calibration_is_a_noop_for_adapters_with_no_requantizing_tokenizer():
    """Mock adapters have no `token_ids` override -- the base class returns
    None -- so calibration must terminate at the historical default (0.25)
    without probing anything, exactly reproducing sec 11.16's TimesFM-side
    finding (stayed at a perfect diagonal-hit fraction across the whole
    sweep because there was nothing to confound it)."""
    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    calibration = calibrate_impulse_amplitude(adapter, window=32)
    assert calibration["amplitude"] == 0.25
    assert calibration["calibrated"] is False
    assert calibration["sweep"] == []
    assert "no re-quantizing tokenizer" in calibration["reason"]


def test_calibration_reproduces_default_when_the_confound_never_fires():
    """A `token_ids`-exposing adapter whose tokenizer is NOT actually
    sensitive to the probe (confound threshold above every candidate) must
    still choose 0.25 -- the sweep tries largest-first, so a checkpoint the
    historical constant was already safe for is unaffected by this fix."""
    adapter = make_test_adapter("mock_step", context_len=128, horizon=8)
    adapter.__class__ = _RequantizingAdapter
    adapter._clean_ref = _clean_ref_for(128)
    adapter.CONFOUND_THRESHOLD = 10.0  # unreachable by any candidate amplitude
    calibration = calibrate_impulse_amplitude(adapter, window=32)
    assert calibration["amplitude"] == 0.25
    assert calibration["calibrated"] is True
    assert calibration["sweep"][0]["amplitude"] == 0.25
    assert calibration["sweep"][0]["unrelated_frac"] == 0.0
    assert len(calibration["sweep"]) == 1  # stopped at the first candidate


def test_calibration_sweeps_down_to_the_largest_safe_amplitude():
    """The core case sec 15 A20 exists for: 0.25/0.15/0.10 confound (every
    unrelated token's id shifts), 0.05/0.02 don't -- the sweep must land on
    0.05, the largest amplitude that is actually safe, and stop there
    without also trying 0.02."""
    adapter = make_test_adapter("mock_step", context_len=128, horizon=8)
    adapter.__class__ = _RequantizingAdapter
    adapter._clean_ref = _clean_ref_for(128)
    calibration = calibrate_impulse_amplitude(
        adapter, window=32, candidates=(0.25, 0.15, 0.10, 0.05, 0.02))
    assert calibration["calibrated"] is True
    assert calibration["amplitude"] == 0.05
    tried = [row["amplitude"] for row in calibration["sweep"]]
    assert tried == [0.25, 0.15, 0.10, 0.05]  # early-stopped, never tried 0.02
    fracs = {row["amplitude"]: row["unrelated_frac"] for row in calibration["sweep"]}
    assert fracs[0.25] == 1.0 and fracs[0.15] == 1.0 and fracs[0.10] == 1.0
    assert fracs[0.05] == 0.0


def test_calibration_falls_back_to_the_smallest_candidate_if_none_pass():
    """If even the smallest candidate confounds, calibration must not raise
    or silently pick an unsafe amplitude -- it returns the smallest tried,
    since that is the least-bad option available, and the sweep still
    records every attempt for a reader to see the gate was never actually
    satisfied."""
    adapter = make_test_adapter("mock_step", context_len=128, horizon=8)
    adapter.__class__ = _RequantizingAdapter
    adapter._clean_ref = _clean_ref_for(128)
    # Strictly between 0 (the clean-vs-itself deviation, which must NOT
    # confound or base_ids/probe_ids would shift identically and cancel out)
    # and the smallest candidate (0.02) -- so every candidate confounds.
    adapter.CONFOUND_THRESHOLD = 0.001
    calibration = calibrate_impulse_amplitude(
        adapter, window=32, candidates=(0.25, 0.15, 0.10, 0.05, 0.02))
    assert calibration["amplitude"] == 0.02
    assert len(calibration["sweep"]) == 5
    assert all(row["unrelated_frac"] == 1.0 for row in calibration["sweep"])


def test_impulse_alignment_check_default_amplitude_is_unchanged():
    """`impulse_alignment_check`'s new `amplitude` parameter must default to
    the exact historical constant (0.25) so no existing caller's behavior
    moves just from this parameter existing (ROADMAP.md sec 15 A20(a))."""
    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    layers = adapter.layer_names()
    default_call = impulse_alignment_check(adapter, window=32, layers=layers)
    explicit_call = impulse_alignment_check(adapter, window=32, layers=layers, amplitude=0.25)
    assert default_call == explicit_call


def test_run_alignment_gate_records_the_calibrated_amplitude():
    adapter = make_test_adapter("mock_step", context_len=128, horizon=8)
    adapter.__class__ = _RequantizingAdapter
    adapter._clean_ref = _clean_ref_for(128)
    layers = adapter.layer_names()
    record = run_alignment_gate(adapter, window=32, layers=layers,
                                min_diagonal_frac=0.5, on_failure="fail")
    assert record["amplitude"] == 0.05
    assert record["calibration"]["calibrated"] is True
    assert record["calibration"]["amplitude"] == 0.05


def test_run_alignment_gate_calibrate_false_uses_the_fixed_default():
    """`calibrate=False` must reproduce the pre-A20 behavior exactly: fixed
    0.25, `calibration` absent (None), so a caller that needs a pre-A20
    run's exact artifact shape/values can still get one."""
    adapter = make_test_adapter("mock_step", context_len=128, horizon=8)
    adapter.__class__ = _RequantizingAdapter
    adapter._clean_ref = _clean_ref_for(128)
    layers = adapter.layer_names()
    record = run_alignment_gate(adapter, window=32, layers=layers,
                                min_diagonal_frac=0.5, on_failure="fail",
                                calibrate=False)
    assert record["amplitude"] == 0.25
    assert record["calibration"] is None


if __name__ == "__main__":
    test_calibration_is_a_noop_for_adapters_with_no_requantizing_tokenizer()
    test_calibration_reproduces_default_when_the_confound_never_fires()
    test_calibration_sweeps_down_to_the_largest_safe_amplitude()
    test_calibration_falls_back_to_the_smallest_candidate_if_none_pass()
    test_impulse_alignment_check_default_amplitude_is_unchanged()
    test_run_alignment_gate_records_the_calibrated_amplitude()
    test_run_alignment_gate_calibrate_false_uses_the_fixed_default()
    print("alignment calibration tests passed")
