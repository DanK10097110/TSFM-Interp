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
from ..manifest import load_manifest, record_extra
from ..utils import batch_slices, log, save_json
from .alignment import align, pooling_matrix, run_alignment_gate
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
    if data.family_resolution:
        record_extra(run_dir, "family_resolution", data.family_resolution)
    if data.corpus_digest:
        # Merged into the `provenance` dict `run_pipeline` already recorded
        # at run start (ROADMAP.md sec 15 A7) -- the corpus is only actually
        # loaded once a data-consuming stage runs, so this is the earliest
        # point the digest is known.
        provenance = load_manifest(run_dir).get("provenance", {})
        provenance["corpus_digest"] = data.corpus_digest
        record_extra(run_dir, "provenance", provenance)

    layer_map = {}
    alignment_records = {}
    for mcfg in cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        layers = adapter.layer_names()
        if cfg.alignment.sanity_check:
            alignment_records[mcfg.name] = run_alignment_gate(
                adapter, cfg.alignment.window, layers,
                cfg.alignment.min_diagonal_frac, cfg.alignment.on_failure)
        layer_map[mcfg.name] = _extract_model(adapter, data, store, cfg, layers)
        if not cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    store.set_layers(layer_map)
    if alignment_records:
        save_json(run_dir / "alignment" / "alignment_check.json", alignment_records)
    log.info("extraction complete: %s", {m: len(ls) for m, ls in layer_map.items()})
    return store


def _extract_model(adapter, data: BenchmarkData, store: ActivationStore,
                   cfg: PipelineConfig, layers: list, rows: np.ndarray | None = None) -> list:
    """Run one model over all (or a row subset of) contexts and persist aligned activations.

    `layers` and `rows` are explicit rather than derived from `adapter`/`data`
    internally so `layer_screen`'s dedicated stride-1 screening pass
    (`ROADMAP.md` sec 15 A1) can reuse this exact extraction path -- same
    batching, alignment, and store-write logic -- against a different layer
    set and a different (smaller) row subset than the main run, instead of
    duplicating it.
    """
    adapter.ensure_loaded()
    pool = pooling_matrix(adapter.token_time_spans(), cfg.data.context_len, cfg.alignment.window)
    contexts = data.contexts()
    if rows is not None:
        contexts = contexts[rows]
    n = contexts.shape[0]
    initialized = set()

    with ActivationCatcher(adapter.module, layers) as catcher:
        for start, end in tqdm(list(batch_slices(n, adapter.cfg.batch_size)),
                               desc=f"extract {adapter.name}"):
            batch = contexts[start:end]
            with torch.no_grad(), torch.autocast(device_type=adapter.device.type,
                                                 dtype=adapter.dtype,
                                                 enabled=adapter.device.type == "cuda"):
                adapter.forward(adapter.prepare(batch))
            acts = catcher.collect()
            for name in layers:
                hidden = adapter.postprocess_tokens(name, acts[name]).float()
                # `cfg.extraction.store_dtype`, not a hardcoded `np.float16`
                # (ROADMAP.md sec 15 A19): a hardcoded cast here meant
                # `store_dtype: float32` only changed the zarr array's
                # declared dtype, never the actual values written into it --
                # the float16 precision loss (and any overflow-to-inf) had
                # already happened before `store.write_batch` ever saw them,
                # making the config knob a no-op for the one thing it was
                # for. Found by reading this exact line, not assumed working
                # because the config field existed.
                aligned = align(hidden, pool).cpu().numpy().astype(np.dtype(cfg.extraction.store_dtype))
                if name not in initialized:
                    store.init_layer(adapter.name, name, aligned.shape[-1],
                                     cfg.extraction.store_dtype)
                    initialized.add(name)
                store.write_batch(adapter.name, name, start, aligned)
    for name in layers:
        store.finalize_layer(adapter.name, name)
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
