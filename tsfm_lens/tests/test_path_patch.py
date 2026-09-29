"""ROADMAP.md sec 20 H8 Stage 3 / E18: `hooks.path_patch_delta`, the two-
forward path-patching primitive.

No adapter or checkpoint involved -- a small synthetic two-block net chained
through an explicit residual addition, mirroring how a real transformer's
blocks are chained (`CLAUDE.md` sec 6.2's o_proj-input-slice convention for
"a head", the same one `test_multi_slice_ablate.py` uses). The property under
test is causal order, which `path_patch_delta` claims to get right WITHOUT
an explicit check (its own docstring): a source head's patched value must
reach a receiver strictly downstream of it, and must NOT reach a receiver in
the same block or an earlier one -- the forward pass's own sequencing should
enforce that for free, and these tests confirm it actually does rather than
trusting the claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.extraction.hooks import InputCatcher, path_patch_delta


class _TwoBlockNet(nn.Module):
    """Two `o_proj`-shaped blocks chained through an explicit residual
    stream: `block1.o_proj`'s input is `x + block0.o_proj(x)`, so a patch to
    `block0`'s input slice (a "head" of block 0) causally reaches
    `block1.o_proj`'s input, but a patch to `block1`'s input slice cannot
    reach `block0.o_proj`'s input (already captured earlier in the same
    forward pass) and a patch to one of `block0`'s own head slices cannot
    reach `block0.o_proj`'s OTHER head slice (disjoint dims of the same
    input tensor, no computation between them)."""

    def __init__(self, n_heads: int = 2, head_dim: int = 3):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        d = n_heads * head_dim
        self.block0 = nn.Module()
        self.block0.o_proj = nn.Linear(d, d)
        self.block1 = nn.Module()
        self.block1.o_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.block0.o_proj(x)
        resid = x + h0
        h1 = self.block1.o_proj(resid)
        return resid + h1


def _head_slice(net: _TwoBlockNet, h: int) -> slice:
    return slice(h * net.head_dim, (h + 1) * net.head_dim)


def _clean_slices(net: _TwoBlockNet, x: torch.Tensor) -> dict:
    with InputCatcher(net, ["block0.o_proj", "block1.o_proj"]) as catcher:
        net(x)
    return catcher.collect()


def test_path_patch_delta_is_nonzero_for_a_causally_downstream_pair():
    """block0's head 0 -> block1's head 1: a real causal path through the
    residual stream, so patching the source's slice must move the
    receiver's captured input slice away from its clean value."""
    torch.manual_seed(5)
    net = _TwoBlockNet()
    x = torch.randn(2, 4, net.n_heads * net.head_dim)
    src_slice, dst_slice = _head_slice(net, 0), _head_slice(net, 1)

    caps = _clean_slices(net, x)
    dst_clean = caps["block1.o_proj"][..., dst_slice]
    src_value = torch.randn(2, 4, net.head_dim)  # stand-in "corrupted" value

    delta = path_patch_delta(net, lambda: net(x),
                             "block0.o_proj", src_slice, src_value,
                             "block1.o_proj", dst_slice, dst_clean)

    assert delta.shape == dst_clean.shape
    assert not torch.allclose(delta, torch.zeros_like(delta))


def test_path_patch_delta_is_exactly_zero_within_the_same_block():
    """Two heads of the SAME block share one o_proj input tensor but occupy
    disjoint dim slices -- patching head 0's slice cannot alter head 1's,
    since `input_slice_ablate`'s hook only ever assigns the patched slice."""
    torch.manual_seed(6)
    net = _TwoBlockNet()
    x = torch.randn(2, 4, net.n_heads * net.head_dim)
    src_slice, dst_slice = _head_slice(net, 0), _head_slice(net, 1)

    caps = _clean_slices(net, x)
    dst_clean = caps["block0.o_proj"][..., dst_slice]
    src_value = torch.randn(2, 4, net.head_dim)

    delta = path_patch_delta(net, lambda: net(x),
                             "block0.o_proj", src_slice, src_value,
                             "block0.o_proj", dst_slice, dst_clean)

    assert torch.equal(delta, torch.zeros_like(delta))


def test_path_patch_delta_is_exactly_zero_for_a_causally_upstream_dst():
    """block1's head cannot causally affect block0.o_proj's input --
    block0's forward already ran, and was already captured, before
    block1's own module (and this patch) ever executes."""
    torch.manual_seed(7)
    net = _TwoBlockNet()
    x = torch.randn(2, 4, net.n_heads * net.head_dim)
    src_slice, dst_slice = _head_slice(net, 0), _head_slice(net, 0)

    caps = _clean_slices(net, x)
    dst_clean = caps["block0.o_proj"][..., dst_slice]
    src_value = torch.randn(2, 4, net.head_dim)

    delta = path_patch_delta(net, lambda: net(x),
                             "block1.o_proj", src_slice, src_value,
                             "block0.o_proj", dst_slice, dst_clean)

    assert torch.equal(delta, torch.zeros_like(delta))


def test_path_patch_delta_is_zero_when_src_value_equals_its_own_clean_value():
    """The identity check: patching `src` to its OWN clean value is a no-op
    intervention, so the downstream delta at any receiver must be exactly
    zero -- mirroring `test_multi_slice_ablate.py`'s own identity rung."""
    torch.manual_seed(8)
    net = _TwoBlockNet()
    x = torch.randn(2, 4, net.n_heads * net.head_dim)
    src_slice, dst_slice = _head_slice(net, 0), _head_slice(net, 1)

    caps = _clean_slices(net, x)
    src_clean = caps["block0.o_proj"][..., src_slice]
    dst_clean = caps["block1.o_proj"][..., dst_slice]

    delta = path_patch_delta(net, lambda: net(x),
                             "block0.o_proj", src_slice, src_clean,
                             "block1.o_proj", dst_slice, dst_clean)

    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


if __name__ == "__main__":
    test_path_patch_delta_is_nonzero_for_a_causally_downstream_pair()
    test_path_patch_delta_is_exactly_zero_within_the_same_block()
    test_path_patch_delta_is_exactly_zero_for_a_causally_upstream_dst()
    test_path_patch_delta_is_zero_when_src_value_equals_its_own_clean_value()
    print("path_patch_delta tests passed")
