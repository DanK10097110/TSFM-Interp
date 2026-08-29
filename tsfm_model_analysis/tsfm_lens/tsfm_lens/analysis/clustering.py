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
from ..utils import load_json, log, sample_rows, save_json
from .stats import bootstrap_ci

_FEATURE_WORDS = {
    "dominant_freq": ("high-frequency", "low-frequency"),
    "spectral_entropy": ("broadband/noisy spectrum", "peaked spectrum"),
    "trend_slope": ("rising trend", "falling trend"),
    "lag1_ac": ("smooth/persistent", "anti-persistent"),
    "noise_ratio": ("noisy", "clean"),
}


def run_clustering(cfg: PipelineConfig, store: ActivationStore, data: BenchmarkData) -> None:
    """Cluster every configured model's activations, label clusters, compare partitions.

    Clustering itself was always per-model; only AMI and the contingency
    matrix are pairwise. So a panel run clusters each model once and reports
    AMI for every pair (`ROADMAP.md` sec 24.3 sub-item 3). `comparison.json`
    keeps its historical top-level `ami`/`contingency`/`rows_a`/`cols_b` keys
    describing pair 0 and gains a `pairs` list carrying the same fields per
    pair.
    """
    out_dir = cfg.run_dir() / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = cfg.comparison_pairs()
    if not pairs:
        raise ValueError(
            "clustering's cross-model product is AMI between two partitions and this "
            f"run has {len(cfg.models)} model(s); pipeline._apply_shape should have "
            "dropped the stage with a stated reason before reaching here "
            "(ROADMAP.md sec 24.3)")
    layer_of = _choose_layers(cfg, store, [m.name for m in cfg.models])

    rows = sample_rows(data.n, cfg.clustering.max_series, cfg.run.seed + 4,
                      strata=data.meta["family"].to_numpy())
    meta = data.meta.iloc[rows].reset_index(drop=True)
    features = _series_features(data.contexts()[rows])
    k = _resolve_k(cfg, meta)

    frames, cluster_meta, assignments = [], {}, {}
    for mcfg in cfg.models:
        model, layer = mcfg.name, layer_of[mcfg.name]
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

    pair_records = []
    for a, b in pairs:
        la, lb = assignments[a.name], assignments[b.name]
        ami = {"value": float(adjusted_mutual_info_score(la, lb))}
        if cfg.stats.enabled:
            ci = bootstrap_ci(lambda idx: float(adjusted_mutual_info_score(la[idx], lb[idx])),
                              len(la), min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                              cfg.run.seed + 5, cfg.stats.ci)
            ami.update({"lo": ci["lo"], "hi": ci["hi"]})
        contingency = pd.crosstab(la, lb, normalize="index")
        pair_records.append({
            "model_a": a.name, "model_b": b.name, "ami": ami,
            "contingency": contingency.to_numpy().tolist(),
            "rows_a": [int(i) for i in contingency.index],
            "cols_b": [int(i) for i in contingency.columns],
        })

    n_families = int(meta["family"].nunique())
    family_comparisons = None
    if n_families < 2:
        # Cluster *labels* (top_family/purity) are guarded in
        # `_label_clusters` itself; this is the corresponding note for the
        # report (`ROADMAP.md` sec 15 A6). AMI/contingency above are
        # unaffected -- they compare the two models' clusters to *each
        # other*, never to family labels, so they stay meaningful
        # regardless of how many families are present.
        family_comparisons = {
            "applicable": False,
            "reason": f"only {n_families} family present in this corpus; every "
                      f"cluster's family purity would be a trivial 100% "
                      f"regardless of clustering quality, so cluster labels above "
                      f"show only feature-based descriptors",
        }
    save_json(out_dir / "clusters.json", cluster_meta)
    save_json(out_dir / "comparison.json", {
        **pair_records[0],
        "pairs": pair_records, "run_shape": cfg.run_shape(),
        "family_comparisons": family_comparisons,
    })
    log.info("clustering complete: AMI=%s",
             ", ".join(f"{r['model_a']}/{r['model_b']}={r['ami']['value']:.3f}"
                       for r in pair_records))


def _choose_layers(cfg: PipelineConfig, store: ActivationStore, names: list) -> dict:
    """Resolve the clustering layer per model: L1 best pair, middle, or explicit index.

    `auto` reads L1's peak-CKA pair, which names two models. On a panel run
    that pair covers only two of the models, so the rest fall back to their
    middle layer -- stated in the log rather than silently, since "this model
    was clustered at its L1 peak" and "this model was clustered at its
    midpoint" are different claims about where the number came from.
    """
    choice = cfg.clustering.layer
    resolved: dict = {}
    if choice == "auto":
        meta_path = cfg.run_dir() / "l1" / "meta.json"
        if meta_path.exists():
            for rec in load_json(meta_path).get("pairs") or [load_json(meta_path)]:
                best = rec["best_pair"]
                resolved.setdefault(rec["model_a"], best["layer_a"])
                resolved.setdefault(rec["model_b"], best["layer_b"])
        else:
            log.warning("clustering.layer=auto but no L1 artifact; using middle layers")
        missing = [n for n in names if n not in resolved]
        if missing and meta_path.exists():
            log.warning("clustering.layer=auto: %s absent from L1's peak pairs; "
                        "using their middle layers", ", ".join(missing))
        for name in missing:
            layers = store.layers(name)
            resolved[name] = layers[len(layers) // 2]
        return {n: resolved[n] for n in names}
    for name in names:
        layers = store.layers(name)
        resolved[name] = layers[len(layers) // 2] if choice == "middle" else layers[int(choice)]
    return resolved


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
    """Approximate label per cluster: majority family plus salient feature descriptors.

    With a single family present, every cluster's family purity is
    trivially 100% regardless of how the clustering actually carved the
    data -- that's an artifact of there being nothing else a series could
    be, not a finding, so the family/purity framing is dropped in favor of
    the feature-based description alone (`ROADMAP.md` sec 15 A6).
    """
    mu, sd = features.mean(), features.std() + 1e-8
    n_families = int(meta["family"].nunique())
    info = {}
    for c in sorted(set(labels)):
        mask = labels == c
        z = ((features[mask].mean() - mu) / sd)
        salient = z.abs().sort_values(ascending=False).index[:2]
        words = [_FEATURE_WORDS[f][0 if z[f] > 0 else 1] for f in salient if abs(z[f]) > 0.4]
        desc = " · ".join(words) if words else "mixed characteristics"
        if n_families >= 2:
            fam_counts = meta.loc[mask, "family"].value_counts()
            top_family = fam_counts.index[0]
            purity = float(fam_counts.iloc[0] / mask.sum())
            label = f"{top_family} ({purity:.0%}) · {desc}"
        else:
            top_family, purity, label = None, None, desc
        info[int(c)] = {
            "label": label, "top_family": top_family, "purity": purity,
            "size": int(mask.sum()),
            "feature_z": {f: float(z[f]) for f in features.columns},
        }
    return info
