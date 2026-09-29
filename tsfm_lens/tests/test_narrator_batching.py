"""`sae/describe.py::_generate`'s batching and out-of-memory backoff (sec 28).

No GPU and no weights: `_generate_one` is the seam, so a stub standing in for
it exercises every path the real one reaches. What is under test is the
BATCHING, which is arithmetic over a list, not the generation.

Motivated by a real failure rather than by symmetry -- `run_sae_describe.py`
hands `_generate` every packet in one call, which was 116 on the three-model
run and 192 on the four-model panel, where it raised `OutOfMemoryError`
inside Qwen's MLP. The packet count is a property of the panel, so any bound
that is not enforced here is a bound that holds until someone adds a model.
"""

from __future__ import annotations

import pytest
import torch

from tsfm_lens.sae import describe as d


class _Stub:
    """Records the sub-batch sizes it was called with, echoes its input.

    `oom_above` reproduces the real failure shape: a call succeeds at or
    below some width and raises above it, which is what a fixed VRAM budget
    against a variable batch actually looks like.
    """

    def __init__(self, oom_above: int | None = None):
        self.oom_above = oom_above
        self.calls: list[int] = []

    def __call__(self, narrator, msgs):
        self.calls.append(len(msgs))
        if self.oom_above is not None and len(msgs) > self.oom_above:
            raise torch.OutOfMemoryError("stub: CUDA out of memory")
        return [f"out:{m}" for m in msgs]


def _msgs(n):
    return [f"m{i}" for i in range(n)]


def _run(monkeypatch, n, oom_above=None):
    stub = _Stub(oom_above)
    monkeypatch.setattr(d, "_generate_one", stub)
    return d._generate(object(), _msgs(n)), stub


def test_output_order_matches_input_across_chunks(monkeypatch):
    """The whole contract. `describe_batch` zips these back against the
    packets positionally, so a reordering would attach every description to
    the wrong feature while looking entirely well-formed."""
    n = d.GEN_BATCH_SIZE * 3 + 5
    out, _ = _run(monkeypatch, n)
    assert out == [f"out:m{i}" for i in range(n)]


def test_a_large_batch_is_split_at_the_bound(monkeypatch):
    n = d.GEN_BATCH_SIZE * 2 + 3
    _, stub = _run(monkeypatch, n)
    assert stub.calls == [d.GEN_BATCH_SIZE, d.GEN_BATCH_SIZE, 3]
    assert sum(stub.calls) == n


def test_a_small_batch_is_one_call(monkeypatch):
    _, stub = _run(monkeypatch, 3)
    assert stub.calls == [3]


def test_the_bound_is_small_enough_to_have_prevented_the_real_failure():
    """192 packets in one call is what raised. A bound is only a bound if it
    is below the quantity that broke -- pinned so raising it later is a
    deliberate act with a test to answer to."""
    assert d.GEN_BATCH_SIZE <= 32


# ---------------------------------------------------------------------------
# The backoff.
# ---------------------------------------------------------------------------

def test_oom_halves_and_still_returns_every_result_once(monkeypatch):
    n = d.GEN_BATCH_SIZE * 2
    out, stub = _run(monkeypatch, n, oom_above=4)
    assert out == [f"out:m{i}" for i in range(n)], "duplicated or dropped"
    assert len(out) == n
    assert max(stub.calls) <= d.GEN_BATCH_SIZE


class _StubFailsPartway:
    """Fails on a LATER sub-batch of a window, not the first one.

    Necessary, not decorative: with a purely width-based failure the retry
    always trips on the window's FIRST inner sub-batch, so nothing has been
    appended yet and the discard-before-retry is a no-op. A stub like that
    passes whether or not the discard exists -- confirmed by planting the
    regression against it, which caught nothing. Real OOM is not purely a
    function of width either (prompt length varies per packet), so this is
    also the more faithful stub.
    """

    def __init__(self, wide_fails_above: int, poison: str):
        self.wide_fails_above = wide_fails_above
        self.poison = poison
        self.poison_spent = False
        self.calls: list[int] = []

    def __call__(self, narrator, msgs):
        self.calls.append(len(msgs))
        if len(msgs) > self.wide_fails_above:
            raise torch.OutOfMemoryError("stub: too wide")
        if self.poison in msgs and not self.poison_spent:
            self.poison_spent = True
            raise torch.OutOfMemoryError("stub: poison packet")
        return [f"out:{m}" for m in msgs]


def test_a_midwindow_oom_does_not_duplicate_the_part_that_succeeded(monkeypatch):
    """🔴 The load-bearing negative. The first version of this backoff retried
    the window WITHOUT discarding what the failed attempt had already
    appended, so a window that OOM'd partway through returned MORE
    descriptions than packets. `describe_batch` zips its result against the
    packets positionally, so that misaligns every description after the
    failure -- each one attached to the wrong feature, all of them
    well-formed, and nothing anywhere raising."""
    half = d.GEN_BATCH_SIZE // 2
    stub = _StubFailsPartway(wide_fails_above=half, poison=f"m{half}")
    monkeypatch.setattr(d, "_generate_one", stub)
    n = d.GEN_BATCH_SIZE
    out = d._generate(object(), _msgs(n))
    assert len(out) == n, f"expected {n} results, got {len(out)}"
    assert out == [f"out:m{i}" for i in range(n)]
    assert stub.poison_spent, "the mid-window failure never fired"


def test_oom_backoff_actually_reaches_a_working_width(monkeypatch):
    out, stub = _run(monkeypatch, 16, oom_above=3)
    assert out == [f"out:m{i}" for i in range(16)]
    # 16 -> 8 -> 4 -> 2, so the widths tried are halvings, and the ones that
    # produced output are all at or below what the stub tolerates.
    assert [c for c in stub.calls if c <= 3], stub.calls
    assert all(c <= 3 for c in stub.calls[-8:]), stub.calls


def test_oom_at_a_single_prompt_is_raised_not_swallowed(monkeypatch):
    """One packet too large for the GPU is a real failure with no smaller
    retry available. Returning a partial list would hand `describe_batch` a
    short list to zip against its packets (sec 11.37: absent and bad must be
    different outcomes)."""
    with pytest.raises(torch.OutOfMemoryError):
        _run(monkeypatch, 8, oom_above=0)


def test_an_empty_batch_makes_no_call(monkeypatch):
    out, stub = _run(monkeypatch, 0)
    assert out == [] and stub.calls == []
