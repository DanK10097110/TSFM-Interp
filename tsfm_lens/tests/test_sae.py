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

from tsfm_lens.report.sae_exemplars import build_exemplar_table, select_feature_exemplars
from tsfm_lens.sae.eval import (_feature_ablated_replacement, _token_level_replacement,
                                _window_broadcast_replacement, dead_feature_rate,
                                reconstruction_fidelity)
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


def test_select_feature_exemplars_ranks_by_activation_descending():
    features = np.array([[0.1, 5.0], [0.9, 1.0], [0.4, 9.0], [0.7, 0.0]])
    series_ids = np.array(["s0", "s1", "s2", "s3"])
    top = select_feature_exemplars(features, series_ids, feature_idx=0, top_k=2)
    assert [e["series_id"] for e in top] == ["s1", "s3"], top
    assert top[0]["activation"] == 0.9
    print("select_feature_exemplars ranking test passed")


def test_build_exemplar_table_matches_planted_feature_to_field_value():
    """A feature planted to track `trend_order` must surface exemplar series
    whose own `trend_order` ground-truth values actually vary sensibly with
    the shown activation -- the whole point of the panel (eyeball the
    correlation a ρ number claims), not just that the plumbing runs."""
    n = 20
    series_ids = np.array([f"s{i}" for i in range(n)])
    trend_order = np.arange(n, dtype=float)
    features = np.zeros((n, 3))
    features[:, 1] = trend_order  # feature 1 tracks trend_order exactly
    gt = pd.DataFrame({"trend_order": trend_order}, index=series_ids)
    meta = pd.DataFrame({"series_id": series_ids, "family": ["trend"] * n})
    matched_features = [{"feature": 1, "best_field": "trend_order", "rho": 0.99}]

    df = build_exemplar_table(features, series_ids, matched_features, meta, gt,
                              top_features=5, top_examples=3)
    assert len(df) == 3
    assert (df["feature"] == 1).all()
    assert (df["best_field"] == "trend_order").all()
    # Top exemplars must be the series with the highest trend_order (feature
    # 1's own values are exactly trend_order, so ranking by activation must
    # reproduce ranking by trend_order) -- and `field_value` must equal the
    # real ground-truth value, not a coincidence of the activation column.
    assert list(df["series_id"]) == ["s19", "s18", "s17"], df["series_id"].tolist()
    assert list(df["field_value"]) == [19.0, 18.0, 17.0], df["field_value"].tolist()
    print("build_exemplar_table planted-feature test passed")


def test_build_exemplar_table_skips_unmatched_features():
    n = 15
    series_ids = np.array([f"s{i}" for i in range(n)])
    features = np.random.default_rng(0).normal(size=(n, 2))
    gt = pd.DataFrame({"trend_order": np.arange(n, dtype=float)}, index=series_ids)
    meta = pd.DataFrame({"series_id": series_ids, "family": ["trend"] * n})
    matched_features = [{"feature": 0, "best_field": None, "rho": 0.0}]
    df = build_exemplar_table(features, series_ids, matched_features, meta, gt)
    assert df.empty, df
    print("build_exemplar_table unmatched-feature skip test passed")


class _IdentitySAE:
    """Passes activations through unchanged -- isolates the pooling/broadcast
    logic itself from any SAE reconstruction error, for the granularity test
    below."""

    def __call__(self, x):
        return x, None


def test_window_broadcast_collapses_within_window_variation_token_level_does_not():
    """ROADMAP.md sec 16 E15: `forecast_preservation`'s "window" granularity
    broadcasts one window-pooled vector across every token in the window,
    destroying real within-window variation for any model tokenized finer
    than the alignment window (Chronos: 1 token/timestep vs. e.g. an 8-step
    window) -- the confound the "token" granularity is supposed to close.
    Verified directly against the two production helper functions with an
    identity SAE, so this isolates the pooling/broadcast mechanism from any
    SAE reconstruction error: token-level replacement must reproduce the
    original per-token values exactly (no information loss at all), while
    window-broadcast replacement must collapse each window's tokens to their
    shared average (the loss the fix is designed to remove)."""
    n_tokens, window, d = 32, 8, 2
    n_windows = n_tokens // window
    spans = np.stack([np.arange(n_tokens), np.arange(n_tokens) + 1], axis=1).astype(np.float64)

    class _FakeAdapter:
        def token_time_spans(self):
            return spans

    class _NS:
        pass

    cfg = _NS()
    cfg.data = _NS()
    cfg.data.context_len = n_tokens
    cfg.alignment = _NS()
    cfg.alignment.window = window

    torch.manual_seed(0)
    clean_tokens = torch.randn(1, n_tokens, d)  # real per-token variation within every window
    sae = _IdentitySAE()
    device = torch.device("cpu")

    token_repl = _token_level_replacement(clean_tokens, sae, device)
    assert torch.allclose(token_repl, clean_tokens), (
        "token-granularity replacement must reproduce the original per-token "
        "values exactly when the SAE is an identity map")

    window_repl = _window_broadcast_replacement(clean_tokens, sae, _FakeAdapter(), cfg, device)
    expected_window_means = clean_tokens.reshape(1, n_windows, window, d).mean(dim=2)
    expected_broadcast = expected_window_means.repeat_interleave(window, dim=1)
    assert torch.allclose(window_repl, expected_broadcast, atol=1e-5), (
        "window-broadcast replacement must equal each window's mean, repeated "
        "across every token in that window")
    # The whole point: within a window, token-level keeps the real variation
    # that window-broadcast collapses to a single shared value.
    within_window_std_token = token_repl[0, :window].std(dim=0).mean().item()
    within_window_std_broadcast = window_repl[0, :window].std(dim=0).max().item()
    assert within_window_std_token > 0.1, within_window_std_token
    assert within_window_std_broadcast < 1e-5, within_window_std_broadcast
    print("window-broadcast-vs-token-granularity replacement test passed")


class _LinearDecodeSAE:
    """encode returns a fixed, caller-supplied features tensor (ignoring the
    input) so the ablation test can isolate `_feature_ablated_replacement`'s
    zero-then-decode mechanism from any real encoder behavior; decode is a
    plain linear map, so a feature's exact contribution to the reconstruction
    is analytically known (outer product of that feature's activations with
    its own decoder row)."""

    def __init__(self, features_by_row, w_dec):
        self.features_by_row = features_by_row  # [N, F]
        self.w_dec = w_dec  # [F, D]

    def encode(self, x):
        return self.features_by_row.clone()

    def decode(self, features):
        return features @ self.w_dec

    def __call__(self, x):
        features = self.encode(x)
        return self.decode(features), features


def test_feature_ablated_replacement_removes_exactly_that_features_contribution():
    """ROADMAP.md sec 7 bullet 3 / sec 16 E15's second half: ablating feature
    idx must remove exactly that feature's own analytically-known contribution
    from the full token-level reconstruction, leaving every other feature's
    contribution (and any row where the ablated feature never fired) untouched."""
    torch.manual_seed(0)
    n_tokens, d, f = 6, 4, 3
    features_by_row = torch.zeros(n_tokens, f)
    features_by_row[:, 0] = torch.tensor([1.0, 2.0, 0.0, 3.0, 0.0, 1.5])
    features_by_row[:, 1] = torch.tensor([0.5, 0.0, 2.0, 0.0, 1.0, 0.0])
    w_dec = torch.randn(f, d)
    sae = _LinearDecodeSAE(features_by_row, w_dec)

    clean_tokens = torch.randn(1, n_tokens, d)  # shape-only input; this SAE's encode ignores it
    device = torch.device("cpu")
    full_recon = _token_level_replacement(clean_tokens, sae, device)
    ablated = _feature_ablated_replacement(clean_tokens, sae, device, feature_idx=0)

    expected_removed = torch.outer(features_by_row[:, 0], w_dec[0]).reshape(1, n_tokens, d)
    expected = full_recon - expected_removed
    assert torch.allclose(ablated, expected, atol=1e-5), (ablated, expected)

    # A row where the ablated feature never fired must be completely unaffected.
    never_fires_row = 2  # feature 0 is zero there
    assert torch.allclose(ablated[0, never_fires_row], full_recon[0, never_fires_row], atol=1e-5)
    print("feature-ablated-replacement mechanism test passed")


if __name__ == "__main__":
    test_topk_sae_shapes_and_sparsity()
    test_train_sae_reduces_loss_and_reconstructs()
    test_dead_neuron_resampling_improves_not_destroys_reconstruction()
    test_dead_feature_rate_in_valid_range()
    test_save_and_load_checkpoint_roundtrip()
    test_ground_truth_matching_finds_planted_correlation()
    test_ground_truth_matching_skips_when_too_few_valid()
    test_select_feature_exemplars_ranks_by_activation_descending()
    test_build_exemplar_table_matches_planted_feature_to_field_value()
    test_build_exemplar_table_skips_unmatched_features()
    test_window_broadcast_collapses_within_window_variation_token_level_does_not()
    test_feature_ablated_replacement_removes_exactly_that_features_contribution()
