"""Project the catch22 feature space to 3D for visual inspection.

UMAP is the primary projector. The fallback (PCA, or t-SNE for a more clustered
look) exists only so the pipeline still produces a plot where UMAP cannot be
installed; it is flagged in the returned method name so a reader never mistakes
one for the other.

Important: this 3D embedding is for the eye only. UMAP preserves local
neighbourhood structure but distorts global distances and densities, so cluster
sizes and between-cluster gaps in the plot are not quantitative. All diversity
numbers are computed in the original feature space in ``diversity.py``, never on
these coordinates.
"""

from __future__ import annotations

import logging
import time

import numpy as np

try:
    import umap

    _HAVE_UMAP = True
except ImportError:
    _HAVE_UMAP = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger("tsfm_benchmark.benchmark_validation.embedding")


def embed_3d(features_scaled: np.ndarray, method: str = "umap", n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0) -> tuple[np.ndarray, str]:
    """Return (n_sequences, 3) coordinates and the method actually used."""
    n = len(features_scaled)
    t0 = time.perf_counter()
    if method == "umap" and _HAVE_UMAP:
        logger.info("embed_3d: fitting UMAP on %d points", n)
        reducer = umap.UMAP(n_components=3, n_neighbors=min(n_neighbors, max(2, n - 1)), min_dist=min_dist, random_state=seed)
        coords = reducer.fit_transform(features_scaled)
        logger.info("embed_3d: umap done in %.2fs", time.perf_counter() - t0)
        return coords, "umap"
    if method == "tsne" or (method == "umap" and not _HAVE_UMAP and n > 4):
        perplexity = min(30, max(5, n // 4))
        logger.info("embed_3d: fitting t-SNE on %d points (perplexity=%d)", n, perplexity)
        coords = TSNE(n_components=3, perplexity=perplexity, random_state=seed, init="pca").fit_transform(features_scaled)
        logger.info("embed_3d: t-SNE done in %.2fs", time.perf_counter() - t0)
        return coords, "tsne_fallback" if method == "umap" else "tsne"
    logger.info("embed_3d: falling back to PCA on %d points", n)
    return PCA(n_components=3, random_state=seed).fit_transform(features_scaled), "pca_fallback"
