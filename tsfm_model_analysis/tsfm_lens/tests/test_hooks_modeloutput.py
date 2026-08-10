"""Regression test for the hooks.py bug found and fixed while adding the
Chronos-2 adapter (ROADMAP.md §9's Phase-4 "any exception outside the new
adapter file is a bug in the abstraction" finding).

`extraction/hooks.py`'s `_primary()` only special-cased plain `tuple`
outputs before indexing `[0]`; it crashed the moment a real block's forward
returned an HF `transformers.utils.ModelOutput`-style dataclass (dict-like,
but *not* an instance of `tuple`, even though it supports `output[0]`
indexing) -- exactly `Chronos2EncoderBlock.forward`'s return type. Rather
than special-case Chronos-2 in its own adapter, the fix generalized
`_primary`/added `_rebuild` in the shared hook module. This test exercises
that fix directly against a small synthetic module returning a real
`ModelOutput` dataclass, with no real checkpoint/adapter/pipeline involved,
so the fix has fast, offline regression coverage independent of whether a
future session can reach the network for a live Chronos-2 run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers.utils import ModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.hooks import ActivationCatcher, output_mean_ablate, token_patch


@dataclass
class _BlockOutput(ModelOutput):
    hidden_states: torch.Tensor = None
    aux_weights: torch.Tensor = None


class _ModelOutputBlock(nn.Module):
    """Mimics an HF encoder block: returns a dataclass, not a tensor/tuple."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> _BlockOutput:
        return _BlockOutput(hidden_states=self.proj(x), aux_weights=torch.ones(x.shape[0]))


class _ModelOutputNet(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.block = _ModelOutputBlock(dim)

    def forward(self, x: torch.Tensor) -> _BlockOutput:
        return self.block(x)


def test_activation_catcher_handles_modeloutput_block():
    net = _ModelOutputNet()
    x = torch.randn(3, 8)
    with ActivationCatcher(net, ["block"]) as catcher:
        out = net(x)
    captured = catcher.collect()
    assert torch.allclose(captured["block"], out.hidden_states)
    # the hook must not have mutated the block's own return type/fields
    assert isinstance(out, _BlockOutput)
    assert out.aux_weights.shape == (3,)


def test_token_patch_preserves_modeloutput_type_and_other_fields():
    """token_patch needs a token axis to index; wrap the [B, D] toy output as
    [B, T=1, D] so `index_fn` has something to slice, then confirm the
    dataclass container (and its other field) survive the patch."""
    net = _ModelOutputNet()
    x = torch.randn(3, 1, 8)
    with token_patch(net, "block", lambda live_len: slice(0, 1), torch.zeros(2, 1, 8)):
        out = net(x)
    assert isinstance(out, _BlockOutput)
    assert out.aux_weights.shape == (3,)
    assert torch.allclose(out.hidden_states[:2, 0], torch.zeros(2, 8))


def test_output_mean_ablate_preserves_modeloutput_type_and_other_fields():
    net = _ModelOutputNet()
    x = torch.randn(3, 8)
    value = torch.full((8,), 5.0)
    with output_mean_ablate(net, "block", value):
        out = net(x)
    assert isinstance(out, _BlockOutput)
    assert out.aux_weights.shape == (3,)
    assert torch.allclose(out.hidden_states, torch.full((3, 8), 5.0))
