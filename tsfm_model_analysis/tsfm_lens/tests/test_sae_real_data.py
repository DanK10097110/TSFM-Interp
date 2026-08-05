"""SAE real-data augmentation tests (ROADMAP.md §6.2 follow-up, user request
2026-08-05: pull additional SAE training data from an established HF
dataset rather than being limited to one run's own small benchmark).

`bootstrap_catalog` is monkeypatched everywhere here -- these tests must
not depend on live network access or Hugging Face Hub availability, the
same discipline `test_smoke.py` follows for the rest of the pipeline.
Actually exercising a real fetch is a separate, manual check (see
ROADMAP.md §6.2's Findings for the one real run that did).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tsfm_lens.sae.real_data as real_data
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models.mock import MockPatchAdapter


def _fake_pool(seed=0):
    rng = np.random.default_rng(seed)
    return [
        (None, rng.normal(size=n).astype(np.float32))
        for n in [50, 600, 1000, 300, 2000]
    ]


def test_sample_real_context_windows_excludes_too_short(monkeypatch):
    monkeypatch.setattr(real_data, "bootstrap_catalog", lambda **kw: _fake_pool())
    windows = real_data.sample_real_context_windows(context_len=512, n_windows=30, seed=0)
    assert windows.shape == (30, 512)
    assert np.isfinite(windows).all()
    print("sample_real_context_windows shape/finite test passed")


def test_sample_real_context_windows_raises_when_none_qualify(monkeypatch):
    monkeypatch.setattr(real_data, "bootstrap_catalog",
                        lambda **kw: [(None, np.zeros(10, dtype=np.float32))])
    try:
        real_data.sample_real_context_windows(context_len=512, n_windows=5, seed=0)
        assert False, "expected ValueError when no series reach context_len"
    except ValueError:
        pass
    print("no-qualifying-series guard test passed")


def test_extract_real_activations_shape():
    model_cfg = ModelConfig(name="patchy", adapter="mock_patch", batch_size=8)
    data_cfg = DataConfig(context_len=128, horizon=32)
    adapter = MockPatchAdapter(model_cfg, data_cfg, torch.device("cpu"), torch.float32)
    layer = adapter.all_layer_names()[-1]

    contexts = np.random.default_rng(0).normal(size=(6, 128)).astype(np.float32)
    alignment_window = 32
    n_windows = data_cfg.context_len // alignment_window

    pooled = real_data.extract_real_activations(
        adapter, layer, contexts, alignment_window, batch_size=4, device=torch.device("cpu"))
    assert pooled.shape == (6 * n_windows, adapter.dim)
    assert np.isfinite(pooled).all()
    print("extract_real_activations shape test passed")


if __name__ == "__main__":
    class _Ctx:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    ctx = _Ctx()
    test_sample_real_context_windows_excludes_too_short(ctx)
    test_sample_real_context_windows_raises_when_none_qualify(ctx)
    test_extract_real_activations_shape()
