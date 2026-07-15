"""Extraction stage: one pass per model over the benchmark, writing
time-aligned activations for every capture layer to the shared store.

GPU efficiency comes from batched forwards under autocast, alignment as a
single einsum on-device, and float16 storage. This stage is the single
funnel every representation-based analysis reads from, and the seam where a
trained SAE will later re-encode activations (see sae/interface.py).
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..utils import batch_slices, log
from .alignment import align, impulse_alignment_check, pooling_matrix
from .hooks import ActivationCatcher
from .store import ActivationStore, save_meta


def run_extraction(cfg: PipelineConfig, hub, data: BenchmarkData) -> ActivationStore:
    """Extract aligned activations for all configured models."""
    run_dir = cfg.run_dir()
    window = cfg.alignment.window
    n_windows = cfg.data.context_len // window
    store = ActivationStore.create(run_dir / "activations.zarr", data.n, n_windows,
                                   window, cfg.data.context_len)
    save_meta(run_dir, data.meta)
    store.write_targets(data.targets())

    layer_map = {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        layer_map[mcfg.name] = _extract_model(adapter, data, store, cfg)
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    store.set_layers(layer_map)
    log.info("extraction complete: %s", {m: len(ls) for m, ls in layer_map.items()})
    return store


def _extract_model(adapter, data: BenchmarkData, store: ActivationStore,
                   cfg: PipelineConfig) -> list:
    """Run one model over all contexts and persist its aligned activations."""
    adapter.ensure_loaded()
    layers = adapter.layer_names()
    if cfg.alignment.sanity_check:
        impulse_alignment_check(adapter, cfg.alignment.window, layers[:: max(1, len(layers) // 4)])
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len, cfg.alignment.window)
    contexts = data.contexts()
    initialized = set()

    with ActivationCatcher(adapter.module, layers) as catcher:
        for start, end in tqdm(list(batch_slices(data.n, adapter.cfg.batch_size)),
                               desc=f"extract {adapter.name}"):
            batch = contexts[start:end]
            with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                 dtype=adapter.dtype,
                                                 enabled=adapter.device.type == "cuda"):
                adapter.forward(adapter.prepare(batch))
            acts = catcher.collect()
            for name in layers:
                hidden = adapter.postprocess_tokens(name, acts[name]).float()
                aligned = align(hidden, pool).cpu().numpy().astype(np.float16)
                if name not in initialized:
                    store.init_layer(adapter.name, name, aligned.shape[-1],
                                     cfg.extraction.store_dtype)
                    initialized.add(name)
                store.write_batch(adapter.name, name, start, aligned)
    return layers


def capture_raw_tokens(adapter, contexts: np.ndarray, layers: list) -> dict:
    """Capture postprocessed token-level states for a small batch, kept in fp32 on cpu.

    Used by L3 activation patching, which needs token-granular clean caches
    rather than the window-pooled store contents.
    """
    adapter.ensure_loaded()
    out = {}
    with torch.no_grad(), ActivationCatcher(adapter.module, layers) as catcher, \
            torch.autocast(device_type=adapter.device.type, dtype=adapter.dtype,
                           enabled=adapter.device.type == "cuda"):
        adapter.forward(adapter.prepare(contexts))
        acts = catcher.collect()
    for name in layers:
        out[name] = adapter.postprocess_tokens(name, acts[name]).float().cpu()
    return out
