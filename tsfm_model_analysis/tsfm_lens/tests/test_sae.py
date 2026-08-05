"""Baseline TopK SAE unit tests (ROADMAP.md Phase 2b item 4, §6.2).

Runnable directly (`python tests/test_sae.py`) or via pytest. Covers the
model's own contract (shapes, exact sparsity, unit-norm decoder), that
training actually reduces reconstruction error on data an SAE can plausibly
fit, the eval metrics' value ranges, and the ground-truth matching logic
(the part with no I/O) against a planted correlation. End-to-end
integration against a real forward pass lives in `test_smoke.py`'s
extended config, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.eval import dead_feature_rate, reconstruction_fidelity
from tsfm_lens.sae.ground_truth import best_ground_truth_matches
from tsfm_lens.sae.models import TopKSAE
from tsfm_lens.sae.train import SAETrainConfig, load_sae_checkpoint, save_sae, train_sae


def test_topk_sae_shapes_and_sparsity():
    torch.manual_seed(0)
    sae = TopKSAE(d_in=16, dict_size=64, k=8)
    x = torch.randn(5, 16)
    features = sae.encode(x)
    assert features.shape == (5, 64)
    assert (features != 0).sum(dim=-1).eq(8).all(), "exactly k features must fire per row"
    recon = sae.decode(features)
    assert recon.shape == x.shape
    dec_norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(dec_norms, torch.ones_like(dec_norms), atol=1e-5)
    print("TopKSAE shapes/sparsity test passed")


def test_train_sae_reduces_loss_and_reconstructs():
    """A low-rank synthetic activation matrix is exactly the case a small SAE should fit well."""
    rng = np.random.default_rng(0)
    d, r, n = 32, 4, 4000
    basis = rng.normal(size=(r, d))
    codes = rng.normal(size=(n, r))
    activations = (codes @ basis).astype(np.float32)

    cfg = SAETrainConfig(dict_size_mult=4, k=4, lr=1e-2, epochs=15, batch_size=512, seed=0)
    device = torch.device("cpu")
    sae, history = train_sae(activations, cfg, device)

    assert history[-1] < history[0] * 0.5, f"loss should drop substantially: {history}"
    fidelity = reconstruction_fidelity(sae, activations, device)
    assert fidelity > 0.7, f"a rank-4 SAE on rank-4 data should reconstruct well, got {fidelity}"
    print(f"train/reconstruct test passed (fidelity={fidelity:.3f})")


def test_dead_neuron_resampling_improves_not_destroys_reconstruction():
    """Regression test for a real bug found building this: an early version of
    dead-neuron resampling used an uncalibrated fixed encoder scale and never
    reset Adam's per-parameter moment estimates, which together made
    reconstruction catastrophically *worse* (fidelity going deeply negative)
    rather than better. A large, overparameterized dictionary on little data
    is exactly the regime that triggers heavy dead-feature collapse, so this
    uses a similar shape to what broke in practice (see ROADMAP.md §6.2)."""
    rng = np.random.default_rng(0)
    d, r, n = 32, 4, 2000
    basis = rng.normal(size=(r, d))
    codes = rng.normal(size=(n, r))
    activations = (codes @ basis).astype(np.float32)

    device = torch.device("cpu")
    baseline_cfg = SAETrainConfig(dict_size_mult=16, k=4, lr=1e-2, epochs=10,
                                  batch_size=256, seed=0, resample_dead_every_epochs=0)
    baseline_sae, baseline_history = train_sae(activations, baseline_cfg, device)
    baseline_fidelity = reconstruction_fidelity(baseline_sae, activations, device)

    resampled_cfg = SAETrainConfig(dict_size_mult=16, k=4, lr=1e-2, epochs=10,
                                   batch_size=256, seed=0, resample_dead_every_epochs=3)
    resampled_sae, resampled_history = train_sae(activations, resampled_cfg, device)
    resampled_fidelity = reconstruction_fidelity(resampled_sae, activations, device)

    assert np.isfinite(resampled_history).all(), resampled_history
    assert resampled_fidelity > -0.5, (
        f"resampling must not destroy reconstruction (got fidelity={resampled_fidelity})")
    assert resampled_fidelity >= baseline_fidelity - 0.05, (
        f"resampling should not meaningfully hurt reconstruction on this "
        f"overparameterized case (baseline={baseline_fidelity}, resampled={resampled_fidelity})")
    print(f"dead-neuron resampling stability test passed "
          f"(baseline={baseline_fidelity:.3f}, resampled={resampled_fidelity:.3f})")


def test_dead_feature_rate_in_valid_range():
    torch.manual_seed(0)
    sae = TopKSAE(d_in=8, dict_size=32, k=4)
    activations = np.random.default_rng(0).normal(size=(500, 8)).astype(np.float32)
    rate = dead_feature_rate(sae, activations, torch.device("cpu"))
    assert 0.0 <= rate <= 1.0
    print(f"dead-feature-rate range test passed (rate={rate:.3f})")


def test_save_and_load_checkpoint_roundtrip(tmp_path=None):
    import tempfile
    out = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    torch.manual_seed(0)
    sae = TopKSAE(d_in=8, dict_size=32, k=4)
    x = torch.randn(3, 8)
    before = sae.decode(sae.encode(x))

    ckpt = out / "sae.pt"
    save_sae(sae, ckpt)
    loaded = load_sae_checkpoint(str(ckpt))
    after = loaded.decode(loaded.encode(x))
    assert torch.allclose(before, after, atol=1e-6)
    print("checkpoint roundtrip test passed")


def test_ground_truth_matching_finds_planted_correlation():
    rng = np.random.default_rng(0)
    n, f = 200, 6
    trend_order = rng.integers(1, 4, size=n).astype(float)
    noise_scale = rng.uniform(0.1, 2.0, size=n)

    features = rng.normal(size=(n, f))
    features[:, 2] = trend_order + rng.normal(scale=0.05, size=n)  # feature 2 tracks trend_order

    series_ids = np.array([f"s{i}" for i in range(n)])
    gt = pd.DataFrame({"trend_order": trend_order, "noise_scale": noise_scale,
                       "n_changepoints": rng.integers(0, 3, size=n).astype(float)},
                      index=series_ids)

    result = best_ground_truth_matches(features, gt, series_ids,
                                       ["trend_order", "noise_scale", "n_changepoints"])
    assert result["n_series_with_ground_truth"] == n
    match = next(r for r in result["features"] if r["feature"] == 2)
    assert match["best_field"] == "trend_order", match
    assert abs(match["rho"]) > 0.9, match
    print("ground-truth planted-correlation test passed")


def test_ground_truth_matching_skips_when_too_few_valid():
    n, f = 5, 3
    features = np.random.default_rng(0).normal(size=(n, f))
    series_ids = np.array([f"s{i}" for i in range(n)])
    gt = pd.DataFrame({"trend_order": [1.0, 2.0, np.nan, np.nan, np.nan]}, index=series_ids)
    result = best_ground_truth_matches(features, gt, series_ids, ["trend_order"], min_valid=10)
    assert result["features"] == []
    print("too-few-valid guard test passed")


if __name__ == "__main__":
    test_topk_sae_shapes_and_sparsity()
    test_train_sae_reduces_loss_and_reconstructs()
    test_dead_neuron_resampling_improves_not_destroys_reconstruction()
    test_dead_feature_rate_in_valid_range()
    test_save_and_load_checkpoint_roundtrip()
    test_ground_truth_matching_finds_planted_correlation()
    test_ground_truth_matching_skips_when_too_few_valid()
