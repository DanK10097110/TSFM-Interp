"""L1 — representational geometry.

Linear CKA between every selected layer pair, computed in feature space on
the accelerator: with column-centered X and Y, CKA = ||Y'X||_F^2 /
(||X'X||_F ||Y'Y||_F), which equals Gram-based linear CKA exactly while
never materializing an N x N matrix. Global similarity uses window-level
rows for statistical power; family-conditioned similarity uses series-level
embeddings so "on which data regimes do these models agree" gets a direct
answer. RSA (Spearman-correlated representational dissimilarity matrices)
is an optional second opinion that is invariant to linear transforms in a
different way than CKA.

CKA is correlational: it evidences shared geometry, not shared mechanism.
L2 and L3 exist to push past that limitation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from ..config import PipelineConfig
from ..extraction.store import ActivationStore, load_meta
from ..utils import log, save_json
from .stats import bootstrap_ci


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Feature-space linear CKA between row-aligned representations [N, Dx], [N, Dy]."""
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    cross = (yc.T @ xc).norm() ** 2
    denom = (xc.T @ xc).norm() * (yc.T @ yc).norm()
    return float((cross / denom.clamp_min(1e-12)).item())


class _LayerBank:
    """Holds selected layers as fp16 cpu tensors, serving fp32 device views per pair."""

    def __init__(self, store: ActivationStore, model: str, layers: list,
                 level: str, rows: np.ndarray, device: torch.device):
        self.device = device
        self.tensors = {}
        for layer in layers:
            arr = store.load(model, layer, level=level, rows=rows)
            flat = arr.reshape(-1, arr.shape[-1])
            self.tensors[layer] = torch.from_numpy(np.ascontiguousarray(flat)).half()

    def get(self, layer: str) -> torch.Tensor:
        return self.tensors[layer].to(self.device).float()


def run_l1(cfg: PipelineConfig, store: ActivationStore, device: torch.device) -> None:
    """Compute CKA matrices, the depth-correspondence curve, and optional RSA."""
    out_dir = cfg.run_dir() / "l1"
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = cfg.comparison_pair()
    layers_a = store.layers(a.name)[:: cfg.l1.layer_stride]
    layers_b = store.layers(b.name)[:: cfg.l1.layer_stride]
    meta = load_meta(cfg.run_dir())
    n = len(meta)
    n_windows = store.root.attrs["n_windows"]

    rng = np.random.default_rng(cfg.run.seed)
    n_series = min(n, max(1, cfg.l1.max_rows // n_windows))
    rows = np.sort(rng.choice(n, size=n_series, replace=False))
    bank_a = _LayerBank(store, a.name, layers_a, "window", rows, device)
    bank_b = _LayerBank(store, b.name, layers_b, "window", rows, device)
    cka_window = np.zeros((len(layers_a), len(layers_b)), dtype=np.float32)
    for i, la in enumerate(layers_a):
        xa = bank_a.get(la)
        for j, lb in enumerate(layers_b):
            cka_window[i, j] = linear_cka(xa, bank_b.get(lb))
    log.info("L1: window-level CKA over %d rows, peak=%.3f", n_series * n_windows,
             cka_window.max())

    families, cka_family = [], []
    if cfg.l1.family_conditioned:
        pooled_a = _LayerBank(store, a.name, layers_a, "series", None, device)
        pooled_b = _LayerBank(store, b.name, layers_b, "series", None, device)
        for fam, group in meta.groupby("family"):
            if len(group) < cfg.l1.min_family_series:
                continue
            idx = torch.from_numpy(group.index.to_numpy().copy()).to(device)
            mat = np.zeros_like(cka_window)
            for i, la in enumerate(layers_a):
                xa = pooled_a.get(la)[idx]
                for j, lb in enumerate(layers_b):
                    mat[i, j] = linear_cka(xa, pooled_b.get(lb)[idx])
            families.append(fam)
            cka_family.append(mat)

    best = np.unravel_index(int(cka_window.argmax()), cka_window.shape)
    depth_curve = [{"layer_a": layers_a[i], "layer_b": layers_b[int(cka_window[i].argmax())],
                    "cka": float(cka_window[i].max())} for i in range(len(layers_a))]

    best_ci, families_ci = None, {}
    if cfg.stats.enabled:
        best_ci = _best_pair_ci(cfg, bank_a, bank_b, layers_a[best[0]], layers_b[best[1]],
                                n_series, n_windows)
        for fam, mat in zip(families, cka_family):
            fi, fj = np.unravel_index(int(mat.argmax()), mat.shape)
            fam_idx = meta.index[meta["family"] == fam].to_numpy()
            xa_f = pooled_a.get(layers_a[fi])[torch.from_numpy(fam_idx.copy()).to(device)]
            xb_f = pooled_b.get(layers_b[fj])[torch.from_numpy(fam_idx.copy()).to(device)]
            families_ci[str(fam)] = bootstrap_ci(
                lambda idx: linear_cka(xa_f[torch.from_numpy(idx).to(device)],
                                       xb_f[torch.from_numpy(idx).to(device)]),
                len(fam_idx), min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                cfg.run.seed + 11, cfg.stats.ci)

    rsa = _run_rsa(cfg, store, meta, layers_a, layers_b, cka_window, a.name, b.name) \
        if cfg.l1.rsa else []

    np.savez(out_dir / "cka.npz", cka_window=cka_window,
             cka_family=np.stack(cka_family) if cka_family else np.zeros((0,) + cka_window.shape))
    save_json(out_dir / "meta.json", {
        "model_a": a.name, "model_b": b.name,
        "layers_a": layers_a, "layers_b": layers_b, "families": families,
        "best_pair": {"layer_a": layers_a[best[0]], "layer_b": layers_b[best[1]],
                      "cka": float(cka_window[best]), "ci": best_ci},
        "families_ci": families_ci,
        "depth_curve": depth_curve, "rsa": rsa,
        "n_rows_window": int(n_series * n_windows),
    })


def _best_pair_ci(cfg: PipelineConfig, bank_a: _LayerBank, bank_b: _LayerBank,
                  layer_a: str, layer_b: str, n_series: int, n_windows: int) -> dict:
    """Cluster bootstrap of window-level CKA at the peak pair, resampling series."""
    xa = bank_a.get(layer_a).view(n_series, n_windows, -1)
    xb = bank_b.get(layer_b).view(n_series, n_windows, -1)

    def stat(idx: np.ndarray) -> float:
        sel = torch.from_numpy(idx).to(xa.device)
        return linear_cka(xa[sel].reshape(-1, xa.shape[-1]),
                          xb[sel].reshape(-1, xb.shape[-1]))

    return bootstrap_ci(stat, n_series, min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                        cfg.run.seed + 10, cfg.stats.ci)


def _run_rsa(cfg: PipelineConfig, store: ActivationStore, meta, layers_a: list,
             layers_b: list, cka_window: np.ndarray, name_a: str, name_b: str) -> list:
    """Spearman RDM agreement along the CKA-matched layer correspondence."""
    rng = np.random.default_rng(cfg.run.seed + 1)
    n = len(meta)
    idx = np.sort(rng.choice(n, size=min(n, cfg.l1.rsa_max_series), replace=False))
    results = []
    for i, la in enumerate(layers_a):
        lb = layers_b[int(cka_window[i].argmax())]
        rdm_a = _rdm(store.load(name_a, la, "series", idx))
        rdm_b = _rdm(store.load(name_b, lb, "series", idx))
        rho = spearmanr(rdm_a, rdm_b).statistic
        results.append({"layer_a": la, "layer_b": lb, "spearman": float(rho)})
    return results


def _rdm(x: np.ndarray) -> np.ndarray:
    """Upper-triangle correlation-distance RDM of series embeddings."""
    x = x.astype(np.float32)
    d = 1.0 - np.corrcoef(x)
    iu = np.triu_indices_from(d, k=1)
    return d[iu]
