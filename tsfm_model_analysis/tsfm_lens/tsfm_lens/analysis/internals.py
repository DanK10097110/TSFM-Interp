"""Per-model internals profile: what is in each model, layer by layer.

Three depth curves per model, all computed from the extraction store:

1. Effective dimensionality — participation ratio of the covariance
   spectrum, showing where the representation expands or compresses.
2. Family decodability — held-out accuracy of a linear probe predicting the
   benchmark family from window states, showing where data-type information
   becomes linearly accessible. Split by series, reported with a bootstrap
   CI over validation series against majority-class chance, plus (ROADMAP.md
   sec 16 E9's remaining scope) a label-permutation null: the identical probe
   refit `probe_permutation_repeats` times with the family label shuffled at
   the series level, so real accuracy can be read against how high a probe
   can score by chance on this exact split/architecture, not only against
   the (weaker, unconditional) majority-class chance line.
3. Input similarity — CKA between each layer and the hand-crafted input
   features from L2, showing how far each depth has moved from raw input
   statistics.

These are per-model diagnostics rather than comparisons, but plotting both
models on shared relative-depth axes is itself informative.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..utils import log, sample_rows, save_json
from .depth_axis import depth_axis_for_run
from .l1_geometry import linear_cka
from .l2_stitching import _baseline_features
from .stats import bootstrap_ci


def run_internals(cfg: PipelineConfig, store: ActivationStore, data: BenchmarkData,
                  device: torch.device) -> None:
    """Compute the three depth profiles for every extracted model."""
    out_dir = cfg.run_dir() / "internals"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_windows = store.root.attrs["n_windows"]
    n_series = min(data.n, max(8, cfg.internals.max_rows // n_windows))
    rows = sample_rows(data.n, n_series, cfg.run.seed + 6, strata=data.meta["family"].to_numpy())
    families = data.meta["family"].to_numpy()[rows]
    row_labels = np.repeat(families, n_windows)

    baseline = _baseline_features(data.contexts()[rows], store.root.attrs["window"])
    base_rows = torch.from_numpy(
        baseline.reshape(-1, baseline.shape[-1]).astype(np.float32)).to(device)

    perm = np.random.default_rng(cfg.run.seed + 61).permutation(n_series)
    n_val = max(2, int(0.25 * n_series))
    val_s, train_s = perm[:n_val], perm[n_val:]
    train_mask = np.zeros(n_series * n_windows, dtype=bool)
    train_mask[(train_s[:, None] * n_windows + np.arange(n_windows)).ravel()] = True

    n_families = int(np.unique(row_labels).size)
    family_comparisons = None
    if n_families < 2:
        # A logistic probe needs >=2 classes to fit at all (sklearn raises
        # otherwise) -- with one family there is nothing to decode, so skip
        # the probe rather than crash, and say so explicitly rather than
        # silently rendering an empty column (`ROADMAP.md` sec 15 A6). This
        # is the failure mode a real single-family checkpoint run actually
        # hit before this guard existed.
        family_comparisons = {
            "applicable": False,
            "reason": f"only {n_families} family present in this corpus; family "
                      f"decodability needs >=2 classes for a probe to fit",
        }

    profile = {}
    for model in store.models():
        layers = store.layers(model)[:: max(1, cfg.internals.layer_stride)]
        eff_dim, input_cka, probes, probe_nulls = [], [], [], []
        for layer in layers:
            arr = store.load(model, layer, level="window", rows=rows)
            x = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
            xt = torch.from_numpy(x).to(device)
            eff_dim.append(_participation_ratio(xt))
            input_cka.append(linear_cka(xt, base_rows))
            if n_families >= 2:
                probes.append(_family_probe(cfg, x, row_labels, train_mask, val_s, n_windows))
                probe_nulls.append(_family_probe_permutation_null(
                    cfg, x, families, train_mask, val_s, n_windows,
                    n_perm=cfg.internals.probe_permutation_repeats,
                    seed=cfg.run.seed + 63))
            else:
                probes.append({"value": None})
                probe_nulls.append({"n_perm": 0, "acc_null_mean": None,
                                    "acc_null_p95": None, "acc_null_values": []})
        counts = np.unique(row_labels[~train_mask], return_counts=True)[1]
        da = depth_axis_for_run(cfg.alignment.depth_axis, store, model, layers)
        profile[model] = {
            "layers": layers,
            "rel_depth": da.coords.tolist(),
            "depth_axis": da.axis, "depth_axis_degraded_from": da.fallback_from,
            "effective_dim": eff_dim,
            "input_cka": input_cka,
            "probe": probes,
            "probe_permutation_null": probe_nulls,
            "chance": float(counts.max() / counts.sum()),
            "n_classes": int(len(counts)),
            "family_comparisons": family_comparisons,
        }
        if n_families >= 2:
            peak = int(np.argmax([p["value"] for p in probes]))
            null_p95 = probe_nulls[peak]["acc_null_p95"]
            log.info("internals %s: probe peak %.2f (null p95 %s) at %s, eff-dim range %.0f-%.0f",
                     model, probes[peak]["value"],
                     f"{null_p95:.2f}" if null_p95 is not None else "n/a",
                     layers[peak], min(eff_dim), max(eff_dim))
        else:
            log.info("internals %s: family probe skipped (%s); eff-dim range %.0f-%.0f",
                     model, family_comparisons["reason"], min(eff_dim), max(eff_dim))
    save_json(out_dir / "profile.json", profile)


def _participation_ratio(x: torch.Tensor) -> float:
    """Effective rank of the covariance spectrum: (sum(ev))^2 / sum(ev^2)."""
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    return float((ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)).item())


def _family_probe(cfg: PipelineConfig, x: np.ndarray, labels: np.ndarray,
                  train_mask: np.ndarray, val_series: np.ndarray, n_windows: int) -> dict:
    """Series-split logistic probe accuracy with a bootstrap CI over validation series."""
    scaler = StandardScaler().fit(x[train_mask])
    pca = PCA(n_components=min(cfg.internals.probe_pca_dim, x.shape[1],
                               train_mask.sum() - 1),
              random_state=cfg.run.seed).fit(scaler.transform(x[train_mask]))
    clf = LogisticRegression(max_iter=300, random_state=cfg.run.seed)
    clf.fit(pca.transform(scaler.transform(x[train_mask])), labels[train_mask])
    correct = (clf.predict(pca.transform(scaler.transform(x[~train_mask])))
               == labels[~train_mask])
    per_series_acc = correct.reshape(len(val_series), n_windows).mean(axis=1)
    if not cfg.stats.enabled:
        return {"value": float(correct.mean())}
    return bootstrap_ci(lambda idx: float(per_series_acc[idx].mean()),
                        len(per_series_acc), cfg.stats.n_boot,
                        cfg.run.seed + 62, cfg.stats.ci)


def _family_probe_permutation_null(cfg: PipelineConfig, x: np.ndarray,
                                   series_families: np.ndarray, train_mask: np.ndarray,
                                   val_series: np.ndarray, n_windows: int,
                                   n_perm: int, seed: int) -> dict:
    """Label-permutation null for `_family_probe` (ROADMAP.md sec 16 E9).

    Reruns the identical probe fit (same scaler/PCA/LogisticRegression
    procedure, same train/val split) `n_perm` times with the family label
    shuffled at the SERIES level -- each series keeps one consistent
    (shuffled) label across all its windows, respecting `CLAUDE.md`
    invariant 2 (the series is the resampling/permutation unit, never a
    window) -- so the real probe's accuracy can be read against how high a
    probe can score on this exact architecture/split by chance alone, not
    only against the weaker, unconditional majority-class `chance` line.
    `n_perm=0` disables this entirely, mirroring
    `sae/ground_truth.py::permutation_null_alignment`'s convention.
    """
    if n_perm <= 0:
        return {"n_perm": 0, "acc_null_mean": None, "acc_null_p95": None,
                "acc_null_values": []}
    rng = np.random.default_rng(seed)
    n_series = len(series_families)
    accs = []
    for _ in range(n_perm):
        shuffled_series_labels = series_families[rng.permutation(n_series)]
        perm_row_labels = np.repeat(shuffled_series_labels, n_windows)
        result = _family_probe(cfg, x, perm_row_labels, train_mask, val_series, n_windows)
        accs.append(result["value"])
    accs = np.asarray(accs, dtype=np.float64)
    return {"n_perm": int(n_perm), "acc_null_mean": float(accs.mean()),
            "acc_null_p95": float(np.quantile(accs, 0.95)),
            "acc_null_values": [float(a) for a in accs]}
