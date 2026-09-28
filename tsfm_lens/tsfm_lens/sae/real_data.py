"""Optional real-world activation pool for SAE training (user request,
2026-08-05): the sealed benchmark corpus a run was built from is a few
hundred series -- SAEs benefit from much more data than that to avoid the
overcapacity/dead-feature problem `ROADMAP.md` §6.2 found on the first
baseline run. This repo already has a generic HF-dataset loader
(`tsfm_benchmark.build_pipeline.sources`) used elsewhere for realism
calibration and the real-derived generators; reused here for a different
purpose. These series never go through the leakage auditor, are never
sealed, and never get ground-truth labels -- `ground_truth_alignment` never
sees them -- so this is purely SAE-training-data augmentation, not a change
to the benchmark itself or anything invariant 10 ("real data never enters
the benchmark directly") governs.
"""

from __future__ import annotations

import numpy as np
import torch

from tsfm_benchmark.build_pipeline.sources import bootstrap_catalog

from ..extraction.alignment import align, pooling_matrix
from ..extraction.extract import capture_raw_tokens
from ..utils import batch_slices, log


def sample_real_context_windows(context_len: int, n_windows: int, seed: int = 0,
                                dataset_name: str = "Monash-University/monash_tsf",
                                total_limit: int = 2000) -> np.ndarray:
    """Slice `n_windows` random length-`context_len` windows out of a real HF time-series pool."""
    pool = bootstrap_catalog(dataset_name=dataset_name, total_limit=total_limit, seed=seed)
    long_enough = [v for _, v in pool if len(v) >= context_len]
    if not long_enough:
        raise ValueError(f"no series in '{dataset_name}' reach context_len={context_len} "
                         f"(pool had {len(pool)} series)")
    log.info(f"sae real-data pool: {len(long_enough)}/{len(pool)} series from "
             f"'{dataset_name}' reach context_len={context_len}")
    rng = np.random.default_rng(seed)
    out = np.empty((n_windows, context_len), dtype=np.float32)
    for i in range(n_windows):
        v = long_enough[rng.integers(len(long_enough))]
        start = rng.integers(0, len(v) - context_len + 1)
        out[i] = v[start: start + context_len]
    return out


@torch.no_grad()
def extract_real_activations(adapter, layer: str, contexts: np.ndarray, alignment_window: int,
                             batch_size: int, device) -> np.ndarray:
    """Run real (non-benchmark) contexts through the model; return window-pooled activations.

    Same shape and pooling convention as the store's own `act/{model}/{layer}`
    (`extraction/alignment.py`'s `pooling_matrix`/`align`), so the result
    concatenates directly with `train.py::load_all_windows`'s output.
    """
    adapter.ensure_loaded()
    pool = pooling_matrix(adapter.token_time_spans(), contexts.shape[1], alignment_window)
    take = min(batch_size, adapter.cfg.batch_size)
    out = []
    for s, e in batch_slices(len(contexts), take):
        # autocast=True (not the patching default): this cache is pooled and
        # concatenated with the STORE's own activations, which are written under
        # autocast -- so matching the store is what keeps the two halves of the
        # SAE's training set on one numerical footing (extract.py's docstring).
        tokens = capture_raw_tokens(adapter, contexts[s:e], [layer],
                                    autocast=True)[layer]
        out.append(align(tokens, pool).numpy())
    pooled = np.concatenate(out, axis=0)
    return pooled.reshape(-1, pooled.shape[-1]).astype(np.float32)
