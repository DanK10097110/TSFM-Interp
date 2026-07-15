"""L2 — cross-model stitching probes.

For strided layer pairs and both directions, a closed-form ridge regression
maps one model's window states onto the other's, evaluated as R^2 on a
held-out series split. The critical control: both models saw the same
input, so predictability alone can reflect shared input structure rather
than shared computation. An input-feature baseline (raw window values, FFT
magnitudes, summary statistics) predicts the same targets, and only the
gain above that baseline is reported as evidence of shared learned
structure. Splits are by series, never by window, to prevent leakage
between highly correlated windows of one series.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..utils import log, save_json
from .stats import bootstrap_ci


def run_l2(cfg: PipelineConfig, store: ActivationStore, data: BenchmarkData,
           device: torch.device) -> None:
    """Fit stitching and baseline probes for both directions of the model pair."""
    out_dir = cfg.run_dir() / "l2"
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = cfg.comparison_pair()
    layers = {a.name: store.layers(a.name)[:: cfg.l2.layer_stride],
              b.name: store.layers(b.name)[:: cfg.l2.layer_stride]}
    n_windows = store.root.attrs["n_windows"]

    rng = np.random.default_rng(cfg.run.seed + 2)
    n_series = min(data.n, max(4, cfg.l2.max_rows // n_windows))
    series_idx = np.sort(rng.choice(data.n, size=n_series, replace=False))
    n_val = max(1, int(len(series_idx) * cfg.l2.val_frac))
    perm = rng.permutation(len(series_idx))
    val_series, train_series = series_idx[perm[:n_val]], series_idx[perm[n_val:]]

    baseline = _baseline_features(data.contexts(), store.root.attrs["window"]) \
        if cfg.l2.input_baseline else None

    results = {}
    for src, dst in ((a.name, b.name), (b.name, a.name)):
        results[f"{src}->{dst}"] = _direction(
            cfg, store, src, dst, layers, train_series, val_series, baseline, device)
    save_json(out_dir / "stitching.json", {
        "model_a": a.name, "model_b": b.name, "layers": layers,
        "directions": results,
        "n_train_series": int(len(train_series)), "n_val_series": int(len(val_series)),
    })
    log.info("L2 complete")


def _direction(cfg: PipelineConfig, store: ActivationStore, src: str, dst: str,
               layers: dict, train_series: np.ndarray, val_series: np.ndarray,
               baseline, device: torch.device) -> dict:
    """R^2 matrix for src-layer -> dst-layer probes plus the per-target baseline row."""
    r2 = np.zeros((len(layers[src]), len(layers[dst])), dtype=np.float32)
    base_r2 = np.zeros(len(layers[dst]), dtype=np.float32)
    dst_cache = {}
    for j, ld in enumerate(layers[dst]):
        ytr = _rows(store, dst, ld, train_series, device)
        yva = _rows(store, dst, ld, val_series, device)
        dst_cache[ld] = (ytr, yva)
        if baseline is not None:
            xtr = _flatten_windows(baseline, train_series, device)
            xva = _flatten_windows(baseline, val_series, device)
            base_r2[j] = ridge_r2(xtr, ytr, xva, yva, cfg.l2.lambdas)
    for i, ls in enumerate(layers[src]):
        xtr = _rows(store, src, ls, train_series, device)
        xva = _rows(store, src, ls, val_series, device)
        for j, ld in enumerate(layers[dst]):
            ytr, yva = dst_cache[ld]
            r2[i, j] = ridge_r2(xtr, ytr, xva, yva, cfg.l2.lambdas)
    gain = r2 - base_r2[None, :]
    out = {"r2": r2.tolist(), "baseline_r2": base_r2.tolist(), "gain": gain.tolist(),
           "best_gain": float(gain.max()), "best_r2": float(r2.max())}
    if cfg.stats.enabled and baseline is not None:
        out["best"] = _best_gain_ci(cfg, store, src, dst, layers, train_series,
                                    val_series, baseline, gain, device)
    return out


def _best_gain_ci(cfg: PipelineConfig, store: ActivationStore, src: str, dst: str,
                  layers: dict, train_series: np.ndarray, val_series: np.ndarray,
                  baseline, gain: np.ndarray, device: torch.device) -> dict:
    """Bootstrap the best pair's gain over validation series with fitted probes held fixed.

    This quantifies evaluation uncertainty (how stable is the number on new
    series from the same distribution), deliberately not refit variability.
    """
    i, j = np.unravel_index(int(gain.argmax()), gain.shape)
    ls, ld = layers[src][i], layers[dst][j]
    n_windows = store.root.attrs["n_windows"]
    ytr = _rows(store, dst, ld, train_series, device)
    yva = _rows(store, dst, ld, val_series, device)
    sse_st, sst = _val_decomposition(_rows(store, src, ls, train_series, device), ytr,
                                     _rows(store, src, ls, val_series, device), yva,
                                     cfg.l2.lambdas, n_windows)
    sse_b, _ = _val_decomposition(_flatten_windows(baseline, train_series, device), ytr,
                                  _flatten_windows(baseline, val_series, device), yva,
                                  cfg.l2.lambdas, n_windows)

    def gain_stat(idx: np.ndarray) -> float:
        s = sst[idx].sum()
        return float((1 - sse_st[idx].sum() / s) - (1 - sse_b[idx].sum() / s))

    def r2_stat(idx: np.ndarray) -> float:
        return float(1 - sse_st[idx].sum() / sst[idx].sum())

    n_boot = min(cfg.stats.n_boot, cfg.stats.n_boot_heavy)
    return {"src_layer": ls, "dst_layer": ld,
            "gain_ci": bootstrap_ci(gain_stat, len(val_series), n_boot,
                                    cfg.run.seed + 20, cfg.stats.ci),
            "r2_ci": bootstrap_ci(r2_stat, len(val_series), n_boot,
                                  cfg.run.seed + 21, cfg.stats.ci)}


def _val_decomposition(xtr: torch.Tensor, ytr: torch.Tensor, xva: torch.Tensor,
                       yva: torch.Tensor, lambdas: list, n_windows: int):
    """Per-validation-series SSE of the best-lambda probe and SST, as numpy arrays."""
    weights, stdz, _ = ridge_fit(xtr, ytr, xva, yva, lambdas)
    x_mean, x_std, y_mean = stdz
    err = ((yva - y_mean) - ((xva - x_mean) / x_std) @ weights) ** 2
    sse = err.reshape(-1, n_windows * err.shape[-1]).sum(dim=1).cpu().numpy()
    sst = ((yva - y_mean) ** 2).reshape(-1, n_windows * err.shape[-1]).sum(dim=1).cpu().numpy()
    return sse.astype(np.float64), sst.astype(np.float64)


def _rows(store: ActivationStore, model: str, layer: str, series: np.ndarray,
          device: torch.device) -> torch.Tensor:
    """Window-level rows for the given series as an fp32 device tensor."""
    arr = store.load(model, layer, level="window", rows=series)
    return torch.from_numpy(arr.reshape(-1, arr.shape[-1]).astype(np.float32)).to(device)


def _flatten_windows(features: np.ndarray, series: np.ndarray,
                     device: torch.device) -> torch.Tensor:
    """Baseline features [N, W, F] restricted to series and flattened to rows."""
    sub = features[series]
    return torch.from_numpy(sub.reshape(-1, sub.shape[-1]).astype(np.float32)).to(device)


def _baseline_features(contexts: np.ndarray, window: int) -> np.ndarray:
    """Hand-crafted per-window input features: raw values, rFFT magnitudes, statistics."""
    n, t = contexts.shape
    w = contexts.reshape(n, t // window, window).astype(np.float32)
    fft = np.abs(np.fft.rfft(w, axis=-1))
    mean = w.mean(axis=-1, keepdims=True)
    std = w.std(axis=-1, keepdims=True)
    x = np.arange(window, dtype=np.float32)
    xc = x - x.mean()
    slope = ((w - mean) * xc).sum(axis=-1, keepdims=True) / (xc ** 2).sum()
    centered = w - mean
    denom = (centered ** 2).sum(axis=-1, keepdims=True) + 1e-8
    lag1 = ((centered[..., 1:] * centered[..., :-1]).sum(axis=-1, keepdims=True)) / denom
    return np.concatenate([w, fft, mean, std, slope, lag1], axis=-1)


def ridge_fit(xtr: torch.Tensor, ytr: torch.Tensor, xva: torch.Tensor, yva: torch.Tensor,
              lambdas: list):
    """Closed-form ridge over a lambda grid; returns best weights, standardization, R^2."""
    x_mean, x_std = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    y_mean = ytr.mean(0, keepdim=True)
    xtr_s, xva_s = (xtr - x_mean) / x_std, (xva - x_mean) / x_std
    ytr_c, yva_c = ytr - y_mean, yva - y_mean
    xtx = xtr_s.T @ xtr_s
    xty = xtr_s.T @ ytr_c
    n, d = xtr_s.shape
    eye = torch.eye(d, device=xtr.device)
    sst = (yva_c ** 2).sum().clamp_min(1e-12)
    best_r2, best_w = -np.inf, None
    for lam in lambdas:
        weights = torch.linalg.solve(xtx + lam * n * eye, xty)
        r2 = float(1.0 - (((yva_c - xva_s @ weights) ** 2).sum() / sst).item())
        if r2 > best_r2:
            best_r2, best_w = r2, weights
    return best_w, (x_mean, x_std, y_mean), best_r2


def ridge_r2(xtr: torch.Tensor, ytr: torch.Tensor, xva: torch.Tensor, yva: torch.Tensor,
             lambdas: list) -> float:
    """Best variance-weighted validation R^2 over the lambda grid."""
    return ridge_fit(xtr, ytr, xva, yva, lambdas)[2]
