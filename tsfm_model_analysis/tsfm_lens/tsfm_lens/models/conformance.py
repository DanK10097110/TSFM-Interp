"""Lightweight `ModelAdapter` conformance checks (ROADMAP.md §10, Phase 5 CI item).

Automates the per-model checklist Phase 4 already writes out by hand
(`CLAUDE.md` §6.2's "Adding a model" steps, ROADMAP.md §9's "Concrete
deliverables per model"): layer discovery works, the alignment machinery is
sane, `predict()` honors its contract, and every *optional* capability either
returns a well-formed value or degrades to `None` rather than raising
(`CLAUDE.md` invariant 8). This does not replace `--check-alignment`'s
judgment call on real checkpoints (near-1.0 diagonal dominance is a "trust
this" threshold a human should still read, not just an automated pass/fail)
-- it catches the mechanical break before that judgment call is even
possible, on any adapter, real or mock, without downloading a checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import DataConfig, ModelConfig
from .base import ModelAdapter


def check_adapter_conformance(adapter: ModelAdapter, window: int, n_series: int = 4,
                              horizon: int = 8, quantiles: list | None = None) -> dict:
    """Run every check; raise loudly on the first broken one (`CLAUDE.md` §2.5).

    Returns a report dict on success -- callers that want a full list of
    everything checked, not just the first failure, can inspect it; a test
    asserting specific keys is a stronger conformance signal than "did not
    raise" alone.
    """
    from ..extraction.alignment import impulse_alignment_check

    quantiles = quantiles or [0.1, 0.5, 0.9]
    report: dict = {"model": adapter.name}

    layers = adapter.all_layer_names()
    if not layers:
        raise AssertionError(f"'{adapter.name}': all_layer_names() returned nothing")
    report["n_layers"] = len(layers)
    report["final_block_name"] = adapter.final_block_name()
    if report["final_block_name"] not in layers:
        raise AssertionError(
            f"'{adapter.name}': final_block_name() '{report['final_block_name']}' "
            f"is not in all_layer_names()")

    discovered = adapter.discover_layers()
    if not discovered:
        raise AssertionError(f"'{adapter.name}': discover_layers() found no modules at all")
    report["n_discovered_modules"] = len(discovered)

    spans = adapter.token_time_spans()
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise AssertionError(f"'{adapter.name}': token_time_spans() shape {spans.shape}, "
                             f"expected [n_tokens, 2]")
    if not np.all(spans[:, 1] > spans[:, 0]):
        raise AssertionError(f"'{adapter.name}': token_time_spans() has a non-positive-width span")
    if not np.all(np.diff(spans[:, 0]) > 0):
        raise AssertionError(f"'{adapter.name}': token_time_spans() starts are not increasing")
    report["n_tokens"] = spans.shape[0]

    context_len = adapter.data_cfg.context_len
    if context_len % window:
        raise AssertionError(
            f"'{adapter.name}': context_len {context_len} is not a multiple of window {window}")
    alignment = impulse_alignment_check(adapter, window)
    if not alignment:
        raise AssertionError(f"'{adapter.name}': impulse_alignment_check() returned nothing")
    if not all(0.0 <= frac <= 1.0 for frac in alignment.values()):
        raise AssertionError(f"'{adapter.name}': impulse alignment fractions out of [0, 1]: "
                             f"{alignment}")
    report["alignment_first_layer_frac"] = next(iter(alignment.values()))

    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(n_series, context_len)).astype(np.float32)
    pred = adapter.predict(contexts, horizon, quantiles)
    if "point" not in pred:
        raise AssertionError(f"'{adapter.name}': predict() missing 'point' key")
    if pred["point"].shape != (n_series, horizon):
        raise AssertionError(f"'{adapter.name}': predict()['point'] shape "
                             f"{pred['point'].shape}, expected {(n_series, horizon)}")
    if not np.all(np.isfinite(pred["point"])):
        raise AssertionError(f"'{adapter.name}': predict()['point'] has non-finite values")
    report["predict_point_shape"] = list(pred["point"].shape)

    prepared = adapter.prepare(contexts)
    _check_optional_degrades_gracefully(adapter, prepared, report)
    return report


def _check_optional_degrades_gracefully(adapter: ModelAdapter, prepared, report: dict) -> None:
    """Every optional capability must return `None` or a well-formed value -- never raise.

    A capability that raises instead of returning `None` breaks every stage
    that calls it expecting a clean skip (`CLAUDE.md` invariant 8); this is
    the one check in the module that is specifically about *not* raising.
    """
    attn_info = adapter.attention_info()
    report["has_attention_info"] = attn_info is not None
    if attn_info is not None:
        for entry in attn_info:
            for key in ("block", "o_proj", "n_heads", "head_dim"):
                if key not in entry:
                    raise AssertionError(
                        f"'{adapter.name}': attention_info() entry missing '{key}': {entry}")

    mlp_info = adapter.mlp_info()
    report["has_mlp_info"] = mlp_info is not None

    adapter.forward(prepared)

    patterns = adapter.attention_patterns(prepared)
    report["has_attention_patterns"] = patterns is not None
    if patterns is not None:
        for name, pat in patterns.items():
            if not isinstance(pat, torch.Tensor) or pat.ndim != 4:
                raise AssertionError(
                    f"'{adapter.name}': attention_patterns()['{name}'] is not a [B,H,T,T] tensor")

    cross = adapter.cross_attention_patterns(prepared)
    report["has_cross_attention_patterns"] = cross is not None


def make_test_adapter(adapter_name: str, context_len: int = 128, horizon: int = 8) -> ModelAdapter:
    """Build a loaded adapter from a bare adapter registry name, for conformance tests only."""
    from . import build_adapter

    mcfg = ModelConfig(name=adapter_name, adapter=adapter_name)
    data_cfg = DataConfig(context_len=context_len, horizon=horizon)
    adapter = build_adapter(mcfg, data_cfg, torch.device("cpu"), torch.float32)
    adapter.ensure_loaded()
    return adapter
