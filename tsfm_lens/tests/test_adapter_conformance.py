"""Adapter-conformance test (ROADMAP.md §10, Phase 5's "lightweight
adapter-conformance test" CI item).

Runs `models/conformance.py::check_adapter_conformance` against every
registered mock adapter, including `mock_wave` -- the third architecture
that already exists in `models/mock.py` specifically to prove a new model
needs no changes outside its own adapter file (`CLAUDE.md` §6.2's "nothing
downstream changes" claim, ROADMAP.md §9's Phase-4 "no changes needed
outside the new adapter file" success bar), but which no default config
actually exercises. This test is the automated version of that manual
checklist item, run here on mocks since real-checkpoint adapters need a
download and a GPU -- see `run.py --check-alignment` for the real-weights
version of the same judgment call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.models.conformance import check_adapter_conformance, make_test_adapter


@pytest.mark.parametrize("adapter_name", ["mock_patch", "mock_step", "mock_wave"])
def test_mock_adapter_passes_conformance(adapter_name):
    adapter = make_test_adapter(adapter_name, context_len=128, horizon=8)
    report = check_adapter_conformance(adapter, window=32, horizon=8)
    assert report["model"] == adapter_name
    assert report["n_layers"] > 0
    assert report["n_tokens"] > 0
    assert report["predict_point_shape"] == [4, 8]
    # every mock is built with real hookable attention/mlp modules
    assert report["has_attention_info"] is True
    assert report["has_mlp_info"] is True
    assert report["has_attention_patterns"] is True
    # mocks are decoder-only-shaped (no encoder-decoder cross-attention)
    assert report["has_cross_attention_patterns"] is False
    print(f"{adapter_name} conformance report: {report}")


def test_conformance_accepts_an_adapter_with_no_optional_capabilities():
    """An architecture that exposes none of the optional capabilities (no
    hookable o_proj, no exposed attention patterns) must still pass -- the
    contract is "return None", not "every model must support everything"
    (`CLAUDE.md` invariant 8)."""
    from tsfm_lens.models.mock import MockPatchAdapter

    class _NoOptionalCapabilitiesAdapter(MockPatchAdapter):
        def attention_info(self):
            return None

        def mlp_info(self):
            return None

        def attention_patterns(self, prepared):
            return None

    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    adapter.__class__ = _NoOptionalCapabilitiesAdapter
    report = check_adapter_conformance(adapter, window=32, horizon=8)
    assert report["has_attention_info"] is False
    assert report["has_mlp_info"] is False
    assert report["has_attention_patterns"] is False
    assert report["has_cross_attention_patterns"] is False


def test_conformance_catches_a_broken_token_time_spans():
    """A deliberately broken adapter (non-monotonic spans) must fail loudly,
    not be silently accepted -- the whole point of an automated checker."""
    import numpy as np

    from tsfm_lens.models.mock import MockPatchAdapter

    class _BrokenSpansAdapter(MockPatchAdapter):
        def token_time_spans(self) -> np.ndarray:
            spans = super().token_time_spans()
            spans[[0, 1]] = spans[[1, 0]]  # swap two rows -> starts no longer increasing
            return spans

    adapter = make_test_adapter("mock_patch", context_len=128, horizon=8)
    adapter.__class__ = _BrokenSpansAdapter
    with pytest.raises(AssertionError, match="starts are not increasing"):
        check_adapter_conformance(adapter, window=32, horizon=8)


if __name__ == "__main__":
    for name in ("mock_patch", "mock_step", "mock_wave"):
        test_mock_adapter_passes_conformance(name)
    test_conformance_accepts_an_adapter_with_no_optional_capabilities()
    test_conformance_catches_a_broken_token_time_spans()
    print("adapter conformance tests passed")
