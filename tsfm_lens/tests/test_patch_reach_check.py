"""The two-sided control that distinguishes "no effect" from "no reach".

`CLAUDE.md` sec 11.42: Chronos-2's forecast head reads
`hidden_states[:, -num_output_patches:]` while `token_slice` writes the
leading context patches, so every `token_patch`-based intervention on that
model returned a bit-identical forecast -- which renders as a flat depth
curve, not as an error.

The identity check alone (patch the final block into itself, require 0.0)
passes for a hook writing into the void, which is why it never caught this.
These tests pin that the reach check is what discriminates, using the
measured deltas from the live diagnosis rather than a synthetic stand-in --
a decoy built with forward hooks cannot represent "the head reads elsewhere",
because a hook registered on the same module runs in the same chain as the
patch it is meant to bypass (attempted, and it failed to sever the path).
"""
from __future__ import annotations

import numpy as np
import pytest

from tsfm_lens.models import conformance


class _Stub:
    """Minimal adapter surface `_check_patch_reaches_the_forecast` touches."""

    def __init__(self, reach_delta: float, declared: bool, identity_delta: float = 0.0,
                 early_delta: float = 0.0, n_layers: int = 2):
        self.name = "stub"
        self._reach, self._declared, self._identity = reach_delta, declared, identity_delta
        self._early, self._n_layers = early_delta, n_layers
        self._call = 0
        self.module = object()

    def all_layer_names(self):
        return [f"blocks.{i}" for i in range(self._n_layers)]

    def final_block_name(self):
        return f"blocks.{self._n_layers - 1}"

    def token_slice(self, live_len):
        return slice(0, live_len)

    def forecast_reads_patched_positions(self):
        return self._declared

    def predict(self, contexts, horizon, quantiles):
        # 1 = baseline, 2 = identity patch, 3 = other-layer patch,
        # 4 = early-block patch (the probe that separates "head never reads
        # this span" from "a final-block patch cannot reach it")
        self._call += 1
        offs = {1: 0.0, 2: self._identity, 3: self._reach,
                4: self._early}[self._call]
        return {"point": np.full((len(contexts), horizon), offs, np.float32)}


@pytest.fixture(autouse=True)
def _stub_patch_machinery(monkeypatch):
    """Neutralize capture/patching; these tests exercise the DECISION, not torch."""
    import contextlib

    monkeypatch.setattr(conformance_deps := conformance, "np", np, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tsfm_lens.extraction.extract",
        type("M", (), {"capture_raw_tokens": staticmethod(
            lambda ad, ctx, layers: {l: np.zeros((len(ctx), 2, 3), np.float32)
                                     for l in layers})})())
    monkeypatch.setitem(
        __import__("sys").modules,
        "tsfm_lens.extraction.hooks",
        type("M", (), {"token_patch": staticmethod(
            lambda *a, **k: contextlib.nullcontext())})())


def _run(stub):
    rep: dict = {}
    conformance._check_patch_reaches_the_forecast(
        stub, np.zeros((2, 8), np.float32), 4, [0.5], rep)
    return rep


def test_a_healthy_adapter_passes_both_halves():
    """mock_patch's real measured values: identity 0.0, reach 0.4627782702445984."""
    rep = _run(_Stub(reach_delta=0.4627782702445984, declared=True))
    assert rep["patch_identity_delta"] == 0.0
    assert rep["patch_reaches_forecast"] is True


def test_a_write_the_head_never_reads_is_caught():
    """Chronos-2's measured shape: identity 0.0 (passes) AND reach 0.0 (fails).

    The load-bearing case -- identity alone would have declared this healthy.
    """
    with pytest.raises(AssertionError, match="bit-identical"):
        _run(_Stub(reach_delta=0.0, declared=True))


def test_identity_alone_would_not_have_caught_it():
    """Pins WHY the old single-sided control was insufficient."""
    stub = _Stub(reach_delta=0.0, declared=True)
    stub._call = 1                      # skip baseline; take the identity patch
    identity = stub.predict(np.zeros((2, 8), np.float32), 4, [0.5])["point"].max()
    assert identity == 0.0              # indistinguishable from a healthy adapter


def test_a_declared_exemption_makes_the_zero_expected():
    """Chronos-2 declares False, so its zero is the documented behavior."""
    rep = _run(_Stub(reach_delta=0.0, declared=False))
    assert rep["patch_reaches_forecast"] is False


def test_a_stale_exemption_is_caught_in_the_permissive_direction():
    """An adapter that declares False but DOES reach is needlessly withholding
    the skip lens -- the declaration must not be allowed to rot either way."""
    with pytest.raises(AssertionError, match="stale"):
        _run(_Stub(reach_delta=0.9, declared=False))


def test_a_broken_token_slice_fails_the_identity_half():
    """If patching a layer into itself moves the forecast, the positions
    token_slice names are not the ones postprocess_tokens captured."""
    with pytest.raises(AssertionError, match="should be exactly"):
        _run(_Stub(reach_delta=0.5, declared=True, identity_delta=0.4))


def test_early_block_probe_records_a_causally_connected_span():
    """A `False` declaration does NOT imply the span is causally disconnected.

    Chronos-2 is the live case: patching context-only at the final block moves
    the forecast by exactly 0.0 (tautologically -- no layer remains to mix the
    written positions into the read ones), while patching the same span at
    block 0 moves it by 0.737. An earlier declaration read the first number as
    "the head never reads context" and was wrong; this probe is what separates
    the two readings, so it must be recorded rather than inferred.
    """
    rep = _run(_Stub(reach_delta=0.0, declared=False, early_delta=0.737, n_layers=12))
    assert rep["patch_reach_delta"] == 0.0
    assert rep["patch_early_block_delta"] == pytest.approx(0.737)
    assert rep["patched_span_is_causally_connected"] is True


def test_early_block_probe_distinguishes_a_genuinely_disconnected_span():
    """The other side: a span nothing downstream reads stays 0.0 at every depth."""
    rep = _run(_Stub(reach_delta=0.0, declared=False, early_delta=0.0, n_layers=12))
    assert rep["patch_early_block_delta"] == 0.0
    assert rep["patched_span_is_causally_connected"] is False


def test_a_causally_connected_span_is_not_treated_as_a_stale_declaration():
    """The load-bearing negative: the probe must RECORD, never raise.

    A non-zero early delta beside a `False` declaration is exactly Chronos-2's
    correct state -- the context representation reaches the head, but not
    through the final-block patch a skip lens performs. Raising here would
    force the declaration back to True and re-introduce the flat curve that
    sec 11.42 exists to prevent.
    """
    rep = _run(_Stub(reach_delta=0.0, declared=False, early_delta=0.9, n_layers=12))
    assert rep["patched_span_is_causally_connected"] is True
    assert rep["patch_reaches_forecast"] is False


def test_early_probe_is_skipped_on_a_stack_too_short_to_have_an_early_block():
    """With <=3 layers there is no meaningful "early" block; record nothing
    rather than a value computed one layer from the end."""
    rep = _run(_Stub(reach_delta=0.0, declared=False, early_delta=0.5, n_layers=2))
    assert "patch_early_block_delta" not in rep
    assert "patched_span_is_causally_connected" not in rep
