"""`capture_raw_tokens`' autocast policy (`CLAUDE.md` sec 11.49).

A clean token cache is only a *clean* cache with respect to the numerical
regime of the forward it is written back into. Every patching consumer (L3
patching, the skip lens, SAE forecast-preservation, the reach probe) pairs
this cache with `adapter.predict()`, which runs in the model's own weight
precision under a bare `no_grad` -- so capturing under bf16 autocast writes
values the model never computed, and the self-patch control that must be
exactly 0.0 by construction instead reads nonzero.

These tests key off the **reported policy** (the `enabled` argument actually
handed to `torch.autocast`), not off observed numerics: autocast is a no-op
on CPU, so a test comparing two CPU captures cannot tell a correct default
from an inert one -- it would pass either way, which is the sec 9 thread-cap
lesson in a different module.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tsfm_lens.extraction import extract as extract_mod


class _CudaLikeAdapter:
    """Claims device.type == 'cuda' without needing a GPU, so the policy is
    exercised on the branch where autocast is not already inert."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.device = type("D", (), {"type": "cuda"})()
        self.dtype = torch.bfloat16
        self.module = torch.nn.Module()
        self.calls = 0

    def ensure_loaded(self):
        pass

    def prepare(self, contexts):
        return contexts

    def forward(self, prepared):
        self.calls += 1

    def postprocess_tokens(self, name, acts):
        return torch.zeros(2, 3, 4)


@pytest.fixture
def _record_autocast(monkeypatch):
    """Capture the `enabled` value passed to torch.autocast, and stub out the
    catcher so no real module tree is needed."""
    import contextlib

    seen = {}

    def fake_autocast(device_type, dtype=None, enabled=True):
        seen["enabled"] = enabled
        seen["dtype"] = dtype
        return contextlib.nullcontext()

    class _Catcher:
        def __init__(self, root, layers):
            self._layers = layers

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def collect(self):
            return {l: torch.zeros(2, 3, 4) for l in self._layers}

    monkeypatch.setattr(extract_mod.torch, "autocast", fake_autocast)
    monkeypatch.setattr(extract_mod, "ActivationCatcher", _Catcher)
    return seen


def test_default_is_autocast_off_because_patching_consumers_run_in_fp32(_record_autocast):
    """The default must be OFF. Every consumer but one patches this cache
    into a bare-no_grad `predict()`; a bf16-computed cache patched into an
    fp32 forward is what made TimesFM's self-patch control read 0.00214
    instead of exactly 0.0."""
    extract_mod.capture_raw_tokens(_CudaLikeAdapter(), np.zeros((2, 8), np.float32),
                                   ["blocks.0"])
    assert _record_autocast["enabled"] is False


def test_the_flag_is_not_inert_when_asked_for(_record_autocast):
    """`sae/real_data.py` needs the STORE's regime, not the patching one, so
    the opt-in has to actually reach torch.autocast -- a default-correct
    parameter that is ignored on the one call site that sets it would put
    the two halves of the SAE's training set on different footings."""
    extract_mod.capture_raw_tokens(_CudaLikeAdapter(), np.zeros((2, 8), np.float32),
                                   ["blocks.0"], autocast=True)
    assert _record_autocast["enabled"] is True
    assert _record_autocast["dtype"] is torch.bfloat16


def test_real_data_collection_opts_into_the_store_regime(monkeypatch):
    """`sae/real_data.py` is the one consumer that must NOT take the patching
    default: its cache is pooled and concatenated with the store's own
    activations, which are written under autocast, so matching the store is
    what keeps the two halves of the SAE's training set on one footing.

    Asserted through the call, not by grepping the source -- the first
    version of this test searched the module text for "autocast=True" and
    passed even with the call reverted, because the *explanatory comment*
    above the call contains that string. A test that can be satisfied by
    prose is green before the feature exists.
    """
    from tsfm_lens.sae import real_data

    seen = {}

    def fake_capture(adapter, contexts, layers, autocast=False):
        seen["autocast"] = autocast
        return {l: torch.zeros(len(contexts), 2, 4) for l in layers}

    monkeypatch.setattr(real_data, "capture_raw_tokens", fake_capture)
    monkeypatch.setattr(real_data, "pooling_matrix", lambda *a, **k: torch.zeros(1, 2))
    monkeypatch.setattr(real_data, "align", lambda t, p: torch.zeros(t.shape[0], 1, 4))

    ad = _CudaLikeAdapter()
    ad.cfg = type("C", (), {"batch_size": 4})()
    ad.token_time_spans = lambda: np.zeros((2, 2))
    real_data.extract_real_activations(ad, "blocks.0", np.zeros((4, 8), np.float32),
                                       alignment_window=4, batch_size=4, device=None)
    assert seen["autocast"] is True
