"""Adapter-specific tests for `models/contrib/lag_llama_adapter.py`
(ROADMAP.md Item D2).

Unlike `tests/test_adapter_conformance.py`'s mock-only suite, this
architecture has no offline mock counterpart: `LagLlamaAdapter.load()`
downloads the real checkpoint's `.ckpt` file even under `random_init=True`
(only the weight-loading step is skipped, not the hyperparameter read), so a
fully offline construction is not possible the way the mock adapters allow.
Matching this repo's own stated convention
(`tests/test_adapter_conformance.py`'s docstring: "real-checkpoint adapters
need a download and a GPU -- see `run.py --check-alignment` for the
real-weights version of the same judgment call"), the full
`check_adapter_conformance` pass against real weights is recorded in
`ROADMAP.md` Item D2's Findings block, not re-run here on every CI
invocation. What IS tested here, offline and without any checkpoint:

- `_stub_removed_loss_module`'s unpickling shim (CLAUDE.md sec 11.9-class
  gluonts-version trap, specific to this adapter) in isolation.
- The real, GPU/network-gated conformance pass, `skipif`-guarded so it runs
  when available (matching this file's own honest scope) and is skipped,
  not silently omitted, otherwise.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.models.contrib.lag_llama_adapter import _stub_removed_loss_module

_HAS_CUDA = torch.cuda.is_available()
try:
    from huggingface_hub import HfApi
    _HAS_NETWORK = True
except ImportError:
    _HAS_NETWORK = False


def test_stub_removed_loss_module_is_idempotent():
    """Calling the stub installer twice must not clobber an already-real module."""
    sys.modules.pop("gluonts.torch.modules.loss", None)
    _stub_removed_loss_module()
    first = sys.modules["gluonts.torch.modules.loss"]
    _stub_removed_loss_module()
    second = sys.modules["gluonts.torch.modules.loss"]
    assert first is second, "a second call replaced an already-installed module"
    sys.modules.pop("gluonts.torch.modules.loss", None)


def test_stub_removed_loss_module_leaves_a_real_module_untouched():
    """If gluonts ever reintroduces the module, the stub must not shadow it."""
    sys.modules.pop("gluonts.torch.modules.loss", None)
    import types
    real = types.ModuleType("gluonts.torch.modules.loss")
    real.DistributionLoss = object()
    sys.modules["gluonts.torch.modules.loss"] = real
    _stub_removed_loss_module()
    assert sys.modules["gluonts.torch.modules.loss"] is real
    sys.modules.pop("gluonts.torch.modules.loss", None)


def test_stub_classes_unpickle_via_setstate():
    """The checkpoint's pickled hyperparameters call __new__ then __setstate__

    without ever invoking __init__ -- this is exactly how `pickle` reconstructs
    an object whose class defines `__setstate__`, and is the mechanism
    `torch.load` relies on to unpickle the Lightning checkpoint's
    `hyper_parameters['loss']` entry (CLAUDE.md sec "torch.load() on
    lag-llama.ckpt fails" -- the gluonts.torch.modules.loss removal trap).
    """
    sys.modules.pop("gluonts.torch.modules.loss", None)
    _stub_removed_loss_module()
    mod = sys.modules["gluonts.torch.modules.loss"]
    obj = mod.NegativeLogLikelihood.__new__(mod.NegativeLogLikelihood)
    obj.__setstate__({"beta": 0.0})
    assert obj.beta == 0.0
    assert isinstance(obj, mod.DistributionLoss)
    sys.modules.pop("gluonts.torch.modules.loss", None)


@pytest.mark.skipif(not _HAS_CUDA, reason="LagLlamaAdapter's default config targets CUDA")
@pytest.mark.skipif(not _HAS_NETWORK, reason="huggingface_hub not importable")
def test_lag_llama_conformance_against_real_checkpoint():
    """Full `check_adapter_conformance` against real
    `time-series-foundation-models/Lag-Llama` weights.

    Downloads ~10MB and requires a GPU; skipped automatically when either is
    unavailable rather than failing CI. See ROADMAP.md Item D2's Findings
    for the recorded pass and every measured number (tier, contrast,
    contiguity, patch_identity_delta, patch_reach_delta) from the session
    that first ran this.
    """
    from tsfm_lens.config import DataConfig, ModelConfig
    from tsfm_lens.models import build_adapter
    from tsfm_lens.models.conformance import check_adapter_conformance

    mcfg = ModelConfig(name="LagLlama", adapter="lag_llama",
                       checkpoint="time-series-foundation-models/Lag-Llama",
                       batch_size=8, capture_layer_stride=1)
    dcfg = DataConfig(source="smoke", context_len=480, horizon=64)
    adapter = build_adapter(mcfg, dcfg, torch.device("cuda"), torch.float32)
    report = check_adapter_conformance(adapter, window=32, horizon=8)
    assert report["model"] == "LagLlama"
    assert report["tier"] == 2
    assert report["n_layers"] == 8
    assert report["has_attention_info"] is True
    assert report["has_mlp_info"] is True
    assert report["has_attention_patterns"] is False
    assert report["patch_identity_delta"] < 1e-3
    assert report["patch_reaches_forecast"] is True
