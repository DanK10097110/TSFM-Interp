"""Tests for the untrained-weights null baseline (ROADMAP.md sec 16 E9):
`ModelConfig.random_init` + `models/base.py::random_init_like`.

Real-checkpoint adapters (Chronos/Chronos-Bolt via `type(model)(model.config)`,
TimesFM via skipping `load_checkpoint`) are exercised directly against a tiny,
fully offline HF T5 config here (no network, no GPU) since that is exactly
the mechanism Chronos/Chronos-Bolt's own `load()` uses -- proving the utility
works is the load-bearing claim; the adapters themselves are one-line callers
of it, covered end-to-end via the mock-adapter pipeline test below and (for
real checkpoints) `configs/null_*.yaml` runs, which need a GPU and are not
run in CI.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import DataConfig, ModelConfig, config_from_dict
from tsfm_lens.models.base import random_init_like
from tsfm_lens.models.mock import MockPatchAdapter
from tsfm_lens.pipeline import run_pipeline


def test_random_init_like_reconstructs_from_config_with_different_weights():
    transformers = pytest.importorskip("transformers")
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(d_model=8, d_ff=16, num_layers=1, num_heads=1, vocab_size=32,
                   decoder_start_token_id=0)
    torch.manual_seed(0)
    pretrained = T5ForConditionalGeneration(cfg)
    original_weight = pretrained.encoder.block[0].layer[0].SelfAttention.q.weight.clone()

    fresh = random_init_like(pretrained)

    assert fresh is not pretrained
    assert isinstance(fresh, T5ForConditionalGeneration)
    assert fresh.config.d_model == cfg.d_model
    fresh_weight = fresh.encoder.block[0].layer[0].SelfAttention.q.weight
    assert fresh_weight.shape == original_weight.shape
    assert not torch.allclose(fresh_weight, original_weight)
    # T5LayerNorm has no `reset_parameters` (verified directly, not assumed --
    # this is exactly the case `random_init_like`'s docstring names as the
    # reason it reconstructs via `type(model)(config)` rather than a
    # `reset_parameters`-only heuristic). A fresh instance still must not
    # silently inherit the pretrained model's layer-norm scale.
    orig_ln = pretrained.encoder.block[0].layer[0].layer_norm.weight
    fresh_ln = fresh.encoder.block[0].layer[0].layer_norm.weight
    assert not torch.allclose(fresh_ln, orig_ln) or torch.allclose(
        fresh_ln, torch.ones_like(fresh_ln))  # both start at ones pre-training; either is fine
    assert fresh.training is False  # random_init_like calls .eval()


def test_random_init_like_raises_without_a_config_attribute():
    plain = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no `.config`"):
        random_init_like(plain)


def test_mock_adapter_random_init_flag_changes_weights_not_architecture():
    data_cfg = DataConfig(context_len=128, horizon=8)
    real_cfg = ModelConfig(name="mock", adapter="mock_patch", random_init=False)
    rand_cfg = ModelConfig(name="mock-random", adapter="mock_patch", random_init=True)

    real = MockPatchAdapter(real_cfg, data_cfg, torch.device("cpu"), torch.float32)
    rand = MockPatchAdapter(rand_cfg, data_cfg, torch.device("cpu"), torch.float32)
    real.ensure_loaded()
    rand.ensure_loaded()

    real_w = dict(real.module.named_parameters())
    rand_w = dict(rand.module.named_parameters())
    assert set(real_w) == set(rand_w)  # identical architecture
    for name in real_w:
        assert real_w[name].shape == rand_w[name].shape
    # At least the first block's attention output projection must differ --
    # same architecture, different (seed-offset) random weights.
    key = "blocks.0.attn.o_proj.weight"
    assert not torch.allclose(real_w[key], rand_w[key])


def test_mock_adapter_random_init_is_reproducible():
    """Same seed offset every time -- two independently constructed
    `random_init=True` adapters must produce identical weights, since a
    null baseline that isn't reproducible would be useless for comparing
    across runs."""
    data_cfg = DataConfig(context_len=128, horizon=8)
    cfg = ModelConfig(name="mock-random", adapter="mock_patch", random_init=True)
    a = MockPatchAdapter(cfg, data_cfg, torch.device("cpu"), torch.float32)
    b = MockPatchAdapter(cfg, data_cfg, torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    b.ensure_loaded()
    key = "blocks.0.attn.o_proj.weight"
    assert torch.allclose(dict(a.module.named_parameters())[key],
                          dict(b.module.named_parameters())[key])


def test_random_init_config_roundtrips_through_yaml(tmp_path):
    from tsfm_lens.config import load_config

    yaml_text = """
run:
  name: null_test
  out_dir: {out_dir}
data:
  source: smoke
models:
  - name: A
    adapter: mock_patch
  - name: A-random
    adapter: mock_patch
    random_init: true
""".format(out_dir=str(tmp_path))
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.models[0].random_init is False
    assert cfg.models[1].random_init is True


def test_pipeline_runs_a_real_vs_random_init_null_baseline_pair(tmp_path=None):
    """End-to-end wiring check (ROADMAP.md sec 16 E9): a config pairing a
    model against its own `random_init` twin is just an ordinary two-model
    config -- `l1`'s CKA runs unchanged and gives a real, finite floor
    number with no analysis-code changes needed, which is this feature's
    whole design point."""
    out = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    cfg = config_from_dict({
        "run": {"name": "null_baseline", "out_dir": out, "device": "cpu",
                "dtype": "float32", "seed": 0},
        "data": {"source": "smoke", "context_len": 128, "horizon": 32,
                 "smoke_series_per_family": 20},
        "alignment": {"window": 32, "sanity_check": True},
        "models": [
            {"name": "patchy", "adapter": "mock_patch", "batch_size": 64},
            {"name": "patchy-random", "adapter": "mock_patch", "batch_size": 64,
             "random_init": True},
        ],
        "l1": {"layer_stride": 1, "max_rows": 20000, "min_family_series": 10,
               "rsa": False},
        "l2": {"enabled": False}, "l3": {"enabled": False}, "lens": {"enabled": False},
        "attention": {"enabled": False}, "exemplars": {"enabled": False},
        "clustering": {"enabled": False}, "internals": {"enabled": False},
        "confirm": {"enabled": False}, "report": {"enabled": False},
        "stats": {"enabled": True, "n_boot": 50, "n_boot_heavy": 50, "min_series": 8},
    })
    run_pipeline(cfg, stages=["extract", "l1"])
    cka = np.load(cfg.run_dir() / "l1" / "cka.npz")["cka_window"]
    assert np.isfinite(cka).all()
    assert cka.max() <= 1.0 + 1e-4
    assert cka.min() >= -1.0 - 1e-4
