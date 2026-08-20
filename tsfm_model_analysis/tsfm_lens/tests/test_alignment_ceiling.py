"""The diagonal-hit gate's own arithmetic ceiling (`CLAUDE.md` sec 11.35).

`run_alignment_gate` asks whether perturbing window *w* moves aligned window
*w* most, and refuses the run below `min_diagonal_frac`. That question has no
answer at all for windows that are indistinguishable from each other -- which
is exactly what happens when a model's tokens are **coarser than the analysis
window**, since several consecutive windows then read one token and receive
identical pooled activations by construction.

This was not hypothetical and was not caught by reading: `thuml/timer-base-84m`
(96-step tokens) at the repo's usual `alignment.window: 32` scored 0.33 against
a 0.5 gate while its token-level argmax was a *perfect* diagonal with exactly
zero leakage. 1/3 is the ceiling, and it was sitting on it.

The most important test here is the no-op one: the correction must leave every
model whose tokens fit inside a window at a ceiling of exactly 1.0, so no
alignment number ever recorded in this repo moves (`CLAUDE.md` sec 2.1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.extraction.alignment import resolvable_hit_ceiling
from tsfm_lens.models.mock import MockPatchAdapter, MockStepAdapter, MockWaveAdapter


class _CoarseTokenAdapter(MockPatchAdapter):
    """Tokens three times the analysis window -- Timer's shape, minus the download."""

    def token_time_spans(self) -> np.ndarray:
        n = self.data_cfg.context_len // 96
        return np.stack([np.arange(n) * 96, np.arange(1, n + 1) * 96], axis=1).astype(float)


def _adapter(cls, context_len=192):
    a = cls(ModelConfig(name="m", adapter="mock_patch", batch_size=8),
            DataConfig(context_len=context_len, horizon=8),
            torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    return a


@pytest.mark.parametrize("cls", [MockPatchAdapter, MockStepAdapter, MockWaveAdapter])
def test_every_model_whose_tokens_fit_a_window_has_a_ceiling_of_exactly_one(cls):
    """The no-op guarantee. Normalizing by a ceiling of 1.0 is the identity, so
    every previously recorded diagonal-hit verdict is re-derived bit-for-bit."""
    out = resolvable_hit_ceiling(_adapter(cls), window=32)
    assert out["ceiling"] == 1.0
    assert out["n_distinguishable"] == out["n_windows"]


def test_tokens_coarser_than_the_window_cap_the_reachable_fraction():
    out = resolvable_hit_ceiling(_adapter(_CoarseTokenAdapter), window=32)
    assert out["n_windows"] == 6            # 192 / 32
    assert out["n_distinguishable"] == 2    # 192 / 96
    assert out["ceiling"] == pytest.approx(1.0 / 3.0)


def test_the_ceiling_rises_to_one_when_the_window_matches_the_token_width():
    """The fix a user is told to apply, verified rather than asserted in prose."""
    a = _adapter(_CoarseTokenAdapter)
    assert resolvable_hit_ceiling(a, window=96)["ceiling"] == 1.0


def test_the_ceiling_is_read_off_the_pooling_matrix_not_a_token_width_rule():
    """Ragged spans: a `min(token_width)` rule gets this wrong in the unsafe
    direction, which is why the ceiling is computed from the pooling matrix.

    Spans [0,64) [64,96) [96,192) at window 32 give windows w0,w1 -> token 0;
    w2 -> token 1; w3,w4,w5 -> token 2. Three distinct supports of six windows,
    so the ceiling is 1/2. A heuristic keyed on the *narrowest* token (32,
    equal to the window) would report 1.0 and let a capped model be judged
    against an unreachable bar -- exactly the failure this whole module exists
    to remove.
    """
    class _Ragged(MockPatchAdapter):
        def token_time_spans(self):
            return np.array([[0., 64.], [64., 96.], [96., 192.]])

    out = resolvable_hit_ceiling(_adapter(_Ragged), window=32)
    assert out["n_windows"] == 6
    assert out["n_distinguishable"] == 3
    assert out["ceiling"] == pytest.approx(0.5)
    assert min(s[1] - s[0] for s in _Ragged.token_time_spans(None)) == 32
