"""ROADMAP.md sec 20 H8 Stage 0: `hooks.multi_slice_ablate` and the widened
`input_slice_ablate` contract (mean vector *or* per-position patch).

Covers the three things Stage 0's own exit criteria name before any set
search is trusted: `multi_slice_ablate` over N heads must equal N nested
`input_slice_ablate` contexts exactly (a set intervention composed wrong
would make every downstream minimal-set result meaningless); a per-position
patch of *every* head at its own clean value must reproduce the clean
output exactly (the identity rung -- if a full patch doesn't restore, no
partial one is interpretable); and the batch-size guard `CLAUDE.md` sec
11.5 already requires for `token_patch` must also fire for a per-position
`input_slice_ablate`/`multi_slice_ablate` call, since this is the first time
that hook has ever carried a batched replacement. No adapter or checkpoint
involved -- a small synthetic net with an `o_proj`-shaped Linear, mirroring
`analysis/attention.py::_ablate_heads`'s own head-slicing convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.hooks import (InputCatcher, input_slice_ablate,
                                        multi_slice_ablate)


class _AttnLikeNet(nn.Module):
    """`o_proj` takes the concatenation of `n_heads` `head_dim`-wide heads,
    exactly the shape `input_slice_ablate`'s real callers hook into."""

    def __init__(self, n_heads: int = 4, head_dim: int = 2):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.o_proj = nn.Linear(n_heads * head_dim, n_heads * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


def _head_slice(net: _AttnLikeNet, h: int) -> slice:
    return slice(h * net.head_dim, (h + 1) * net.head_dim)


def test_multi_slice_ablate_equals_nested_input_slice_ablate():
    torch.manual_seed(0)
    net = _AttnLikeNet()
    x = torch.randn(3, 5, net.n_heads * net.head_dim)
    mean0 = torch.randn(net.head_dim)
    mean2 = torch.randn(net.head_dim)

    with input_slice_ablate(net, "o_proj", _head_slice(net, 0), mean0):
        with input_slice_ablate(net, "o_proj", _head_slice(net, 2), mean2):
            nested = net(x)

    with multi_slice_ablate(net, [("o_proj", _head_slice(net, 0), mean0),
                                  ("o_proj", _head_slice(net, 2), mean2)]):
        composed = net(x)

    assert torch.allclose(nested, composed)


def test_multi_slice_ablate_actually_changes_the_output_from_baseline():
    """A composed set ablation that happens to equal the unablated forward
    would pass the equality test above vacuously -- confirm it really
    changes something relative to no intervention at all."""
    torch.manual_seed(1)
    net = _AttnLikeNet()
    x = torch.randn(2, 3, net.n_heads * net.head_dim)
    baseline = net(x)
    with multi_slice_ablate(net, [("o_proj", _head_slice(net, 0), torch.zeros(net.head_dim)),
                                  ("o_proj", _head_slice(net, 1), torch.zeros(net.head_dim))]):
        ablated = net(x)
    assert not torch.allclose(baseline, ablated)


def test_full_per_position_patch_of_every_head_reproduces_clean_output_exactly():
    """The identity rung (ROADMAP.md sec 20 H8 Stage 0's exit criterion):
    capture every head's clean input slice, run a *different* (stand-in
    for "corrupted") input through the net, but patch every head slice back
    to its clean per-position value -- the o_proj input is then, cell for
    cell, exactly the clean input, so the output must match the clean
    forward bit-for-bit (up to float tolerance)."""
    torch.manual_seed(2)
    net = _AttnLikeNet(n_heads=3, head_dim=4)
    x_clean = torch.randn(4, 6, net.n_heads * net.head_dim)
    x_other = torch.randn(4, 6, net.n_heads * net.head_dim)

    with InputCatcher(net, ["o_proj"]) as catcher:
        clean_out = net(x_clean)
    clean_in = catcher.collect()["o_proj"]

    specs = [("o_proj", _head_slice(net, h), clean_in[:, :, _head_slice(net, h)])
            for h in range(net.n_heads)]
    with multi_slice_ablate(net, specs):
        patched_out = net(x_other)

    assert torch.allclose(patched_out, clean_out, atol=1e-6)


def test_per_position_replacement_raises_on_a_smaller_live_batch():
    """CLAUDE.md sec 11.5's token_patch guard, reintroduced for
    input_slice_ablate: a per-position [B, T, head_dim] value is only valid
    if the live forward's batch is at least as large as the cache it was
    built from. A stale/mismatched cache must fail loudly, not silently
    misalign rows."""
    net = _AttnLikeNet()
    small_batch = torch.randn(2, 5, net.n_heads * net.head_dim)
    stale_cache = torch.randn(5, 5, net.head_dim)  # built for a larger batch

    with pytest.raises(RuntimeError, match="ablation_max_series"):
        with input_slice_ablate(net, "o_proj", _head_slice(net, 0), stale_cache):
            net(small_batch)


def test_per_position_replacement_only_patches_the_caches_own_leading_rows_on_a_larger_batch():
    """The reverse direction (live batch larger than the cache) is not an
    error -- exactly `token_patch`'s own convention (`patched[:
    replacement.shape[0]]`): only the cache's leading rows are patched, the
    rest of the batch passes through untouched."""
    torch.manual_seed(4)
    net = _AttnLikeNet()
    big_batch = torch.randn(4, 5, net.n_heads * net.head_dim)
    baseline = net(big_batch)
    cache = torch.ones(2, 5, net.head_dim)

    with input_slice_ablate(net, "o_proj", _head_slice(net, 0), cache):
        out = net(big_batch)

    assert out.shape[0] == 4
    assert not torch.allclose(out[:2], baseline[:2])   # patched rows changed
    assert torch.allclose(out[2:], baseline[2:])        # untouched rows didn't


def test_mean_vector_path_is_unchanged_bit_for_bit():
    """The 1-D mean-ablation path (every caller before this session) must
    not move at all -- no already-recorded head-ablation number should
    shift because of the new per-position branch."""
    torch.manual_seed(3)
    net = _AttnLikeNet()
    x = torch.randn(3, 5, net.n_heads * net.head_dim)
    mean_vec = torch.randn(net.head_dim)

    def _old_style_hook_output():
        module = dict(net.named_modules())["o_proj"]
        handle_input = {}

        def hook(_module, inputs):
            xi = inputs[0].clone()
            xi[..., _head_slice(net, 1)] = mean_vec.to(device=xi.device, dtype=xi.dtype)
            return (xi,) + tuple(inputs[1:])

        h = module.register_forward_pre_hook(hook)
        try:
            return net(x)
        finally:
            h.remove()

    reference = _old_style_hook_output()
    with input_slice_ablate(net, "o_proj", _head_slice(net, 1), mean_vec):
        current = net(x)
    assert torch.allclose(reference, current)


if __name__ == "__main__":
    test_multi_slice_ablate_equals_nested_input_slice_ablate()
    test_multi_slice_ablate_actually_changes_the_output_from_baseline()
    test_full_per_position_patch_of_every_head_reproduces_clean_output_exactly()
    test_per_position_replacement_raises_on_a_smaller_live_batch()
    test_per_position_replacement_only_patches_the_caches_own_leading_rows_on_a_larger_batch()
    test_mean_vector_path_is_unchanged_bit_for_bit()
    print("multi_slice_ablate tests passed")
