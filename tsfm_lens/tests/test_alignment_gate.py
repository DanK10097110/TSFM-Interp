"""Regression test for the alignment gate (`ROADMAP.md` sec 15 A2).

Locks in that `run_alignment_gate` actually enforces invariant 7 -- a broken
`token_time_spans` mapping must fail loudly by default, not just log a line
and let the pipeline continue with silently misaligned activations. Mirrors
the "deliberately broken adapter" pattern already used by
`test_adapter_conformance.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.alignment import run_alignment_gate
from tsfm_lens.models.conformance import make_test_adapter
from tsfm_lens.models.mock import MockPatchAdapter


class _PermutedSpansAdapter(MockPatchAdapter):
    """Individually valid spans, but permuted across tokens -- every window's
    declared source token is now wrong, without tripping `token_time_spans`'s
    own monotonicity checks (unlike `test_conformance_catches_a_broken_token_time_spans`'s
    two-row swap), so this specifically exercises the alignment gate.
    """

    def token_time_spans(self) -> np.ndarray:
        spans = super().token_time_spans()
        return np.roll(spans, shift=len(spans) // 2, axis=0)


def test_alignment_gate_fails_loudly_on_broken_spans():
    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    adapter.__class__ = _PermutedSpansAdapter
    layers = adapter.layer_names()
    with pytest.raises(RuntimeError, match="alignment check FAILED"):
        run_alignment_gate(adapter, window=32, layers=layers,
                           min_diagonal_frac=0.5, on_failure="fail")


def test_alignment_gate_warns_instead_of_raising_when_configured():
    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    adapter.__class__ = _PermutedSpansAdapter
    layers = adapter.layer_names()
    record = run_alignment_gate(adapter, window=32, layers=layers,
                                min_diagonal_frac=0.5, on_failure="warn")
    assert record["passed"] is False
    assert record["shallowest_frac"] < 0.5


def test_alignment_gate_passes_on_a_correct_adapter():
    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    layers = adapter.layer_names()
    record = run_alignment_gate(adapter, window=32, layers=layers,
                                min_diagonal_frac=0.5, on_failure="fail")
    assert record["passed"] is True
    assert record["min"] > 0.5
    assert record["n_layers_probed"] == len(layers)


if __name__ == "__main__":
    test_alignment_gate_fails_loudly_on_broken_spans()
    test_alignment_gate_warns_instead_of_raising_when_configured()
    test_alignment_gate_passes_on_a_correct_adapter()
    print("alignment gate tests passed")
