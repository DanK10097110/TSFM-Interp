"""SAE evaluation harness (ROADMAP.md §6.2): reconstruction fidelity,
dead-feature rate, and forecast-preservation under reconstruction -- the
validity check Phase 3's planned feature-ablation work depends on, since
patching in a reconstruction that already breaks the forecast would make
any feature-ablation delta uninterpretable.
"""

from __future__ import annotations

import numpy as np
import torch

from ..analysis.l3_perturbation import _token_windows
from ..analysis.stats import mase
from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.alignment import align, pooling_matrix
from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..extraction.store import ActivationStore
from ..utils import batch_slices


@torch.no_grad()
def reconstruction_fidelity(sae, activations: np.ndarray, device,
                            batch_size: int = 8192) -> float:
    """Fraction of variance explained by the SAE's reconstruction over the given rows."""
    x = torch.from_numpy(activations)
    mean = x.mean(dim=0).to(device)
    total_resid, total_var = 0.0, 0.0
    for s, e in batch_slices(x.shape[0], batch_size):
        batch = x[s:e].to(device)
        recon, _ = sae(batch)
        total_resid += float(((batch - recon) ** 2).sum())
        total_var += float(((batch - mean) ** 2).sum())
    return 1.0 - total_resid / max(total_var, 1e-8)


@torch.no_grad()
def dead_feature_rate(sae, activations: np.ndarray, device, batch_size: int = 8192,
                      threshold: float = 1e-8) -> float:
    """Fraction of dictionary atoms that never fire above `threshold` over the given rows."""
    x = torch.from_numpy(activations)
    ever_fired = torch.zeros(sae.dict_size, dtype=torch.bool, device=device)
    for s, e in batch_slices(x.shape[0], batch_size):
        features = sae.encode(x[s:e].to(device))
        ever_fired |= (features.abs() > threshold).any(dim=0)
    return float((~ever_fired).float().mean())


@torch.no_grad()
def forecast_preservation(cfg: PipelineConfig, adapter, layer: str, sae,
                          store: ActivationStore, data: BenchmarkData, device) -> dict:
    """Patch the SAE's reconstruction of clean activations into a clean forward pass.

    Compares the resulting forecast's MASE against the model's own unpatched
    forecast, both against real targets. The SAE trains on window-pooled
    activations (`act/{model}/{layer}`), so its reconstruction is computed at
    that same pooled granularity, then broadcast to every raw token
    belonging to that window (via the same per-token-to-window assignment
    `analysis/l3_perturbation.py` already uses for its own patching) before
    being patched in as one whole-context replacement via `token_patch` --
    the same intervention primitive L3 uses. This means any degradation
    reported here reflects the SAE's reconstruction error *and* the
    window-pooling information loss together, not the SAE in isolation; the
    two are not decomposed in this pass.

    ⚠️ That window-pooling loss is **not the same size for every
    architecture, and the difference is large enough to dominate this
    check's result for some models.** When `alignment.window` equals a
    model's own token width (TimesFM's default: one 32-step patch per
    window), the broadcast is exact -- one token per window, nothing to
    average over -- so this check measures close to pure SAE reconstruction
    error. When a model tokenizes at a finer grain than the window
    (Chronos: one token per *timestep*, so a 32-step window covers 32
    distinct tokens), broadcasting one reconstructed vector across all of
    them destroys real within-window variation regardless of SAE quality.
    Confirmed empirically, not just argued (ROADMAP.md §6.2's Findings):
    after fixing this baseline's dead-feature collapse, TimesFM's check
    passed almost exactly (ΔMASE +0.047) while Chronos-T5-Base's did not
    (ΔMASE +2.37) despite a *similar* reconstruction-fidelity improvement
    for both models -- read Chronos's number here as upper-bounded by this
    granularity mismatch, not as a clean SAE-only validity failure.
    """
    adapter.ensure_loaded()
    take = min(data.n, cfg.sae.forecast_preservation_max_series, adapter.cfg.batch_size)
    contexts = data.contexts()[:take]
    targets = data.targets()[:take]
    seed = cfg.run.seed + 11

    torch.manual_seed(seed)
    f_clean = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_clean = mase(f_clean, targets, contexts)

    clean_tokens = capture_raw_tokens(adapter, contexts, [layer])[layer]  # [B, n_tokens, D] fp32
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len, cfg.alignment.window)
    n_windows = cfg.data.context_len // cfg.alignment.window
    win_of_token = _token_windows(adapter.token_time_spans(), cfg.alignment.window, n_windows)

    pooled = align(clean_tokens, pool)  # [B, W, D] -- exactly what the SAE trained on
    b, w, d = pooled.shape
    recon_pooled, _ = sae(pooled.reshape(-1, d).to(device))
    recon_pooled = recon_pooled.reshape(b, w, d).cpu()
    win_idx = torch.from_numpy(win_of_token)
    replacement = recon_pooled[:, win_idx, :]  # broadcast each token to its window's recon

    with token_patch(adapter.module, layer, adapter.token_slice, replacement):
        torch.manual_seed(seed)
        f_patch = adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"]
    mase_patch = mase(f_patch, targets, contexts)

    return {
        "n_series": int(take),
        "mase_clean": float(np.mean(mase_clean)),
        "mase_reconstructed": float(np.mean(mase_patch)),
        "mase_delta": float(np.mean(mase_patch) - np.mean(mase_clean)),
    }
