"""Parameter/FLOPs/latency accounting (ROADMAP.md sec 18 F2).

Everything here runs against a tiny synthetic stack on CPU rather than a real
checkpoint, because the properties worth pinning are architectural rather than
numeric: that the role split is exact and total-preserving, that a layer regex
missing part of the body surfaces as a named `interleaved` bucket instead of a
plausible-looking head size, and that a FLOP counter which cannot see the
model's work produces a warning rather than a quietly-small number.

The one number that *is* checked against a hand-computed value is the analytic
matmul bound, since it is the sanity floor everything else is judged against
and a wrong constant there would silently disarm the check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis import model_budget as mb


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return torch.relu(self.lin(x))


class _Stack(nn.Module):
    """Front-end -> N blocks -> head, the shape the role split assumes."""

    def __init__(self, dim: int = 8, n_blocks: int = 12):
        super().__init__()
        self.embed = nn.Linear(1, dim, bias=False)          # front-end
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(n_blocks))
        self.out = nn.Linear(dim, 1, bias=False)            # head

    def forward(self, x):
        h = self.embed(x.unsqueeze(-1))
        for b in self.blocks:
            h = b(h)
        return self.out(h)


class _Adapter:
    """The slice of `ModelAdapter` that `model_budget` actually touches."""

    def __init__(self, model: nn.Module, blocks: list, n_tokens: int = 4):
        self._model, self._blocks, self._n_tokens = model, blocks, n_tokens
        self.name = "stub"

    def ensure_loaded(self):
        pass

    @property
    def module(self):
        return self._model

    def all_layer_names(self):
        return list(self._blocks)

    def hidden_size(self):
        return 8

    def prepare(self, contexts):
        return torch.as_tensor(contexts, dtype=torch.float32)

    def forward(self, prepared):
        self._model(prepared)

    def token_time_spans(self):
        return np.stack([np.arange(self._n_tokens),
                         np.arange(self._n_tokens) + 1], axis=1)

    def predict(self, contexts, horizon, quantiles):
        return {"point": np.zeros((len(contexts), horizon), dtype=np.float32)}


def _stub(n_blocks: int = 12, kept: list = None) -> _Adapter:
    model = _Stack(n_blocks=n_blocks)
    idx = range(n_blocks) if kept is None else kept
    return _Adapter(model, [f"blocks.{i}" for i in idx])


def test_role_of_matches_dotted_prefixes_not_substrings():
    """`blocks.1` must not swallow `blocks.11` -- the bug a naive `startswith`
    or `in` test would introduce, and one that would silently move eight
    blocks' parameters into the wrong bucket on any 10+ layer model."""
    blocks = ["blocks.1", "blocks.11"]
    assert mb._role_of("blocks.1.lin.weight", blocks) == "blocks.1"
    assert mb._role_of("blocks.11.lin.weight", blocks) == "blocks.11"
    assert mb._role_of("blocks.111.lin.weight", blocks) is None
    assert mb._role_of("embed.weight", blocks) is None
    # An exact match with no trailing path is still that block.
    assert mb._role_of("blocks.1", blocks) == "blocks.1"


def test_census_roles_are_exact_and_sum_to_total():
    census = mb.parameter_census(_stub(n_blocks=12))
    dim = 8
    assert census["n_blocks"] == 12
    assert census["front_end"] == 1 * dim            # embed
    assert census["head"] == dim * 1                 # out
    assert census["body"] == 12 * dim * dim
    assert census["interleaved"] == 0
    assert (census["front_end"] + census["body"] + census["head"]
            + census["interleaved"] == census["total"])
    assert set(census["per_block"]) == {f"blocks.{i}" for i in range(12)}
    assert all(v == dim * dim for v in census["per_block"].values())
    assert census["trainable"] == census["total"]


def test_skipped_middle_blocks_surface_as_interleaved_not_as_a_bigger_head(caplog):
    """A layer regex that captures only some blocks is the realistic failure
    (CLAUDE.md sec 11.8: a library bump renames modules). When the ones it
    misses sit between captured blocks, folding them into `head` would report
    a head many times the true head size and look entirely plausible.
    """
    adapter = _stub(n_blocks=12, kept=[0, 1, 2, 3, 8, 9, 10, 11])
    with caplog.at_level("WARNING"):
        census = mb.parameter_census(adapter)
    dim = 8
    assert census["body"] == 8 * dim * dim
    assert census["interleaved"] == 4 * dim * dim  # blocks 4..7, unseen
    assert census["head"] == dim * 1               # unchanged, not inflated
    assert (census["front_end"] + census["body"] + census["head"]
            + census["interleaved"] == census["total"])
    assert any("belong to no role" in r.getMessage() for r in caplog.records)


def test_uncaptured_trailing_blocks_are_indistinguishable_from_a_head():
    """The stated limit of the guard above, pinned so it is not mistaken for
    full coverage: nothing about *position* separates a missed final block
    from a genuine head, so `interleaved` stays 0 and `head` absorbs them.
    `n_blocks` is the check that catches this case."""
    adapter = _stub(n_blocks=12, kept=[0, 1, 2, 3])
    census = mb.parameter_census(adapter)
    dim = 8
    assert census["interleaved"] == 0
    assert census["head"] == 8 * dim * dim + dim   # the eight, plus the real head
    assert census["n_blocks"] == 4                 # the signal that is available
    assert (census["front_end"] + census["body"] + census["head"]
            + census["interleaved"] == census["total"])


def test_analytic_bound_is_two_flops_per_body_weight_per_token():
    census = {"body": 1000}
    assert mb._analytic_body_flops(census, n_tokens=4, batch=2) == 2.0 * 1000 * 4 * 2
    # A model with no captured body has no bound to compare against, rather
    # than a bound of zero that every measurement would trivially clear.
    assert mb._analytic_body_flops({"body": 0}, n_tokens=4, batch=2) is None


def test_flops_sanity_flags_a_counter_that_missed_the_work():
    census = {"body": 1000}
    forward = {"n_tokens": 4, "batch": 2}          # analytic bound = 16000
    low = mb.flops_sanity(census, dict(forward, flops=1000.0))
    assert low["verdict"] == "suspicious_low"
    assert low["measured_over_analytic"] == pytest.approx(1000.0 / 16000.0)

    # Measured *above* the bound is the normal case: the bound covers only the
    # body's dense projections, not attention scores or the head.
    assert mb.flops_sanity(census, dict(forward, flops=40000.0))["verdict"] == "plausible"
    # Exactly at the floor is not suspicious.
    assert mb.flops_sanity(census, dict(forward, flops=8000.0))["verdict"] == "plausible"


def test_flops_sanity_declines_to_compare_when_either_side_is_missing():
    forward = {"n_tokens": 4, "batch": 2, "flops": None}
    assert mb.flops_sanity({"body": 1000}, forward)["verdict"] == "not_comparable"
    assert mb.flops_sanity({"body": 0}, dict(forward, flops=1.0))["verdict"] == "not_comparable"


def test_per_block_flops_are_recovered_from_the_same_measured_pass():
    """F1's D2 compute-fraction depth axis needs cumulative FLOPs per block,
    and F2's acceptance criterion is that it costs no second pass -- so these
    come out of the module breakdown the counter already produced."""
    adapter = _stub(n_blocks=6)
    contexts = np.zeros((3, 4), dtype=np.float32)
    cost = mb.measure_forward_cost(adapter, contexts, torch.device("cpu"),
                                   repeats=1, warmup=0)
    blocks = cost["blocks"]
    assert blocks is not None
    names = adapter.all_layer_names()
    assert list(blocks["per_block"]) == names          # adapter's forward order
    assert all(v > 0 for v in blocks["per_block"].values())
    # Cumulative is a running sum, so it is non-decreasing and ends at the body
    # total -- the property the depth axis actually relies on.
    cum = [blocks["cumulative"][n] for n in names]
    assert cum == sorted(cum)
    assert cum[-1] == pytest.approx(blocks["body_total"])
    assert blocks["body_total"] == pytest.approx(sum(blocks["per_block"].values()))
    # The body is most of a forward pass but not all of it (embed + head).
    assert 0.0 < blocks["fraction_of_forward"] < 1.0


def test_per_block_flops_degrade_to_none_when_blocks_cannot_be_resolved(caplog):
    """A depth axis silently pinned at compute fraction 0 for every block
    would look like a finding rather than a missing measurement."""
    adapter = _stub(n_blocks=4)
    measured = {"total": 100.0, "by_module": {"_Stack.blocks.0": 10.0}}
    with caplog.at_level("WARNING"):
        assert mb._per_block_flops(measured, adapter.module,
                                   adapter.all_layer_names()) is None
    assert any("could not be matched to this model's 4 blocks" in r.getMessage()
               for r in caplog.records)
    assert mb._per_block_flops(None, adapter.module, adapter.all_layer_names()) is None


class _TwoStack(nn.Module):
    """The Chronos-T5 shape: two same-class sub-stacks under one root, and an
    adapter whose forward enters only the first of them."""

    def __init__(self, dim: int = 8, n_blocks: int = 4):
        super().__init__()
        self.encoder = _Stack(dim=dim, n_blocks=n_blocks)
        self.decoder = _Stack(dim=dim, n_blocks=n_blocks)

    def forward(self, x):
        return self.decoder(self.encoder(x).squeeze(-1))


class _EncoderOnlyAdapter(_Adapter):
    """`ChronosAdapter.forward` calls `self._t5.encoder(...)`, not the model."""

    def forward(self, prepared):
        self._model.encoder(prepared)


def test_per_block_flops_resolve_when_the_adapter_enters_only_a_sub_stack():
    """The live bug behind ROADMAP.md sec 18 F4's upper-bound headline: nothing
    resolved 0 of Chronos-T5-Base's 12 blocks because `FlopCounterMode` keys its
    hierarchy from the module actually *entered*, so an encoder-only forward
    yields `_Stack.blocks.0`, never `_TwoStack.encoder.blocks.0`. The lookup has
    to follow the counter's rule rather than assume the model was the root."""
    model = _TwoStack(n_blocks=4)
    adapter = _EncoderOnlyAdapter(model, [f"encoder.blocks.{i}" for i in range(4)])
    cost = mb.measure_forward_cost(adapter, np.zeros((2, 4), dtype=np.float32),
                                   torch.device("cpu"), repeats=1, warmup=0)
    blocks = cost["blocks"]
    assert blocks is not None, "the encoder-only shape must resolve, not degrade"
    assert list(blocks["per_block"]) == adapter.all_layer_names()
    assert all(v > 0 for v in blocks["per_block"].values())
    # The decoder ran in neither the counted pass nor the sum, so the captured
    # body is a real fraction of a pass that only contains the encoder.
    assert 0.0 < blocks["fraction_of_forward"] < 1.0


def test_same_class_stacks_entered_separately_withhold_per_block_flops(caplog):
    """A bare suffix match would resolve `blocks.0` here too -- from a key whose
    value is encoder-plus-decoder, putting decoder FLOPs on an encoder depth
    axis. That is the exact error F4 exists to prevent, so this must degrade."""
    model = _TwoStack(n_blocks=4)

    class _BothStacksAdapter(_Adapter):
        def forward(self, prepared):
            self._model.encoder(prepared)
            self._model.decoder(prepared)

    adapter = _BothStacksAdapter(model, [f"encoder.blocks.{i}" for i in range(4)])
    with caplog.at_level("WARNING"):
        cost = mb.measure_forward_cost(adapter, np.zeros((2, 4), dtype=np.float32),
                                       torch.device("cpu"), repeats=1, warmup=0)
    assert cost["blocks"] is None
    assert any("entered as a FLOP-counter root" in r.getMessage() for r in caplog.records)
    # The pass total is still measured -- only the per-block split is withheld.
    assert cost["flops"] is not None and cost["flops"] > 0


def test_entering_the_root_still_uses_the_fully_qualified_keys():
    """The pre-existing path must be untouched: when the adapter's forward does
    enter the model, keys are `<model class>.<block path>` and the exact match
    resolves them without the entry reconstruction ever being consulted."""
    model = _TwoStack(n_blocks=3)
    adapter = _Adapter(model, [f"encoder.blocks.{i}" for i in range(3)])
    cost = mb.measure_forward_cost(adapter, np.zeros((2, 4), dtype=np.float32),
                                   torch.device("cpu"), repeats=1, warmup=0)
    assert cost["blocks"] is not None
    keys = mb._keys_from_entry({"entered": set(), "by_module": {}}, model,
                               adapter.all_layer_names())
    assert keys is None, "no entry record must mean no reconstruction, not a guess"


def test_measure_flops_degrades_to_none_instead_of_failing_the_run(caplog):
    def boom():
        raise RuntimeError("no dispatcher here")

    with caplog.at_level("WARNING"):
        assert mb._measure_flops(boom) is None
    assert any("FLOP measurement failed" in r.getMessage() for r in caplog.records)


def test_forward_cost_measures_a_real_pass_on_cpu():
    adapter = _stub(n_blocks=6)
    contexts = np.zeros((3, 4), dtype=np.float32)
    cost = mb.measure_forward_cost(adapter, contexts, torch.device("cpu"),
                                   repeats=2, warmup=1)
    assert cost["batch"] == 3 and cost["context_len"] == 4 and cost["n_tokens"] == 4
    assert cost["timing"]["n"] == 2
    assert cost["timing"]["min_s"] <= cost["timing"]["median_s"] <= cost["timing"]["max_s"]
    assert cost["peak_vram_bytes"] is None          # CPU reports no VRAM
    # The counter should see this model's matmuls, and the measurement should
    # clear its own sanity floor -- if it does not, the floor is miscalibrated.
    assert cost["flops"] is not None and cost["flops"] > 0
    assert cost["flops_per_series"] == pytest.approx(cost["flops"] / 3)
    census = mb.parameter_census(adapter)
    assert mb.flops_sanity(census, cost)["verdict"] == "plausible"
