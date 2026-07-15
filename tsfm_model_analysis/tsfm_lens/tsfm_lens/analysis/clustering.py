"""Activation clustering with approximate "what does this activate for" labels.

Series-level embeddings at a chosen layer (by default, each model's side of
the L1 peak-CKA pair) are clustered, and each cluster is labeled two ways:
by benchmark ground truth (majority family and purity) and by interpretable
signal statistics of its member series (dominant frequency, trend, noise
level, spectral entropy, autocorrelation) expressed as z-scores against the
corpus. The cross-model question "do these models carve the data the same
way" is answered by adjusted mutual information between the two partitions
plus a contingency matrix, which is robust to arbitrary cluster relabeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..utils import load_json, log, save_json
from .stats import bootstrap_ci

_FEATURE_WORDS = {
    "dominant_freq": ("high-frequency", "low-frequency"),
    "spectral_entropy": ("broadband/noisy spectrum", "peaked spectrum"),
    "trend_slope": ("rising trend", "falling trend"),
    "lag1_ac": ("smooth/persistent", "anti-persistent"),
    "noise_ratio": ("noisy", "clean"),
}


def run_clustering(cfg: PipelineConfig, store: ActivationStore, data: BenchmarkData) -> None:
    """Cluster both models' activations, label clusters, compare partitions."""
    out_dir = cfg.run_dir() / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = cfg.comparison_pair()
    layer_a, layer_b = _choose_layers(cfg, store, a.name, b.name)

    rng = np.random.default_rng(cfg.run.seed + 4)
    rows = np.sort(rng.choice(data.n, size=min(data.n, cfg.clustering.max_series),
                              replace=False))
    meta = data.meta.iloc[rows].reset_index(drop=True)
    features = _series_features(data.contexts()[rows])
    k = _resolve_k(cfg, meta)

    frames, cluster_meta, assignments = [], {}, {}
    for model, layer in ((a.name, layer_a), (b.name, layer_b)):
        x = store.load(model, layer, level="series", rows=rows).astype(np.float32)
        z = PCA(n_components=min(cfg.clustering.pca_dim, x.shape[1], len(rows) - 1),
                random_state=cfg.run.seed).fit_transform(StandardScaler().fit_transform(x))
        labels = KMeans(n_clusters=k, n_init=10, random_state=cfg.run.seed).fit_predict(z)
        coords = _embed_2d(cfg, z)
        sil = float(silhouette_score(z, labels)) if 1 < k < len(rows) else float("nan")
        info = _label_clusters(labels, meta, features)
        assignments[model] = labels
        cluster_meta[model] = {"layer": layer, "k": k, "silhouette": sil, "clusters": info}
        frames.append(pd.DataFrame({
            "model": model, "series_id": meta["series_id"], "family": meta["family"],
            "cluster": labels, "x": coords[:, 0], "y": coords[:, 1],
            "label": [info[c]["label"] for c in labels],
        }))
        log.info("clustering %s @ %s: k=%d silhouette=%.3f", model, layer, k, sil)

    pd.concat(frames, ignore_index=True).to_parquet(out_dir / "embedding.parquet")
    la, lb = assignments[a.name], assignments[b.name]
    ami = {"value": float(adjusted_mutual_info_score(la, lb))}
    if cfg.stats.enabled:
        ci = bootstrap_ci(lambda idx: float(adjusted_mutual_info_score(la[idx], lb[idx])),
                          len(la), min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                          cfg.run.seed + 5, cfg.stats.ci)
        ami.update({"lo": ci["lo"], "hi": ci["hi"]})
    contingency = pd.crosstab(la, lb, normalize="index")
    save_json(out_dir / "clusters.json", cluster_meta)
    save_json(out_dir / "comparison.json", {
        "model_a": a.name, "model_b": b.name, "ami": ami,
        "contingency": contingency.to_numpy().tolist(),
        "rows_a": [int(i) for i in contingency.index],
        "cols_b": [int(i) for i in contingency.columns],
    })
    log.info("clustering complete: AMI=%.3f", ami["value"])


def _choose_layers(cfg: PipelineConfig, store: ActivationStore,
                   name_a: str, name_b: str) -> tuple:
    """Resolve the clustering layer per model: L1 best pair, middle, or explicit index."""
    choice = cfg.clustering.layer
    la, lb = store.layers(name_a), store.layers(name_b)
    if choice == "auto":
        meta_path = cfg.run_dir() / "l1" / "meta.json"
        if meta_path.exists():
            best = load_json(meta_path)["best_pair"]
            return best["layer_a"], best["layer_b"]
        log.warning("clustering.layer=auto but no L1 artifact; using middle layers")
        choice = "middle"
    if choice == "middle":
        return la[len(la) // 2], lb[len(lb) // 2]
    idx = int(choice)
    return la[idx], lb[idx]


def _resolve_k(cfg: PipelineConfig, meta: pd.DataFrame) -> int:
    """Cluster count: the family count clipped to a sane range, or an explicit value."""
    if cfg.clustering.k == "auto":
        return int(np.clip(meta["family"].nunique(), 2, 20))
    return int(cfg.clustering.k)


def _embed_2d(cfg: PipelineConfig, z: np.ndarray) -> np.ndarray:
    """2-D coordinates for plotting: UMAP when available, PCA otherwise."""
    if cfg.clustering.use_umap:
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=cfg.run.seed).fit_transform(z)
        except ImportError:
            log.warning("umap-learn not installed; falling back to PCA coordinates")
    return z[:, :2]


def _series_features(contexts: np.ndarray) -> pd.DataFrame:
    """Interpretable per-series statistics used to describe what clusters activate for."""
    v = contexts.astype(np.float32)
    n, t = v.shape
    centered = v - v.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    band = power[:, 1:]
    total = band.sum(axis=1, keepdims=True) + 1e-12
    p = band / total
    dominant = (band.argmax(axis=1) + 1) / t
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(band.shape[1])
    tc = np.arange(t, dtype=np.float32) - (t - 1) / 2
    slope = (centered * tc).sum(axis=1) / (tc ** 2).sum() / (v.std(axis=1) + 1e-6)
    lag1 = (centered[:, 1:] * centered[:, :-1]).sum(axis=1) / ((centered ** 2).sum(axis=1) + 1e-8)
    resid = v - uniform_filter1d(v, size=9, axis=1, mode="nearest")
    noise = resid.std(axis=1) / (v.std(axis=1) + 1e-6)
    return pd.DataFrame({"dominant_freq": dominant, "spectral_entropy": entropy,
                         "trend_slope": slope, "lag1_ac": lag1, "noise_ratio": noise})


def _label_clusters(labels: np.ndarray, meta: pd.DataFrame,
                    features: pd.DataFrame) -> dict:
    """Approximate label per cluster: majority family plus salient feature descriptors."""
    mu, sd = features.mean(), features.std() + 1e-8
    info = {}
    for c in sorted(set(labels)):
        mask = labels == c
        fam_counts = meta.loc[mask, "family"].value_counts()
        top_family, purity = fam_counts.index[0], fam_counts.iloc[0] / mask.sum()
        z = ((features[mask].mean() - mu) / sd)
        salient = z.abs().sort_values(ascending=False).index[:2]
        words = [_FEATURE_WORDS[f][0 if z[f] > 0 else 1] for f in salient if abs(z[f]) > 0.4]
        desc = " · ".join(words) if words else "mixed characteristics"
        info[int(c)] = {
            "label": f"{top_family} ({purity:.0%}) · {desc}",
            "top_family": top_family, "purity": float(purity),
            "size": int(mask.sum()),
            "feature_z": {f: float(z[f]) for f in features.columns},
        }
    return info
