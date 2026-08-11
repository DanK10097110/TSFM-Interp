"""ROADMAP.md sec 16 E17: patch-boundary phase-sensitivity probe.

Direct empirical test of the period=32 aliasing hypothesis `CLAUDE.md`
sec 12 bullet 5 names but explicitly says was never verified: "TimesFM's
finest resolvable lag is one patch-width (~32 steps) ... a real ceiling
on what that model's attention can express." This script asks a related
but distinct question at the *forecast* level, not the attention-lag
level: does trimming 0..patch_width-1 points off the front of an
otherwise-fixed context -- which only changes where patch boundaries land
relative to the underlying signal, not the context's own endpoint or the
forecast target right after it -- change the forecast's MASE? A
patch-tokenized model with no phase invariance built into its embedding
should show real MASE variance purely from this arbitrary trim; a model
whose tokenization has no patch boundaries to be sensitive to (Chronos's
per-timestep scalar quantization) should not -- and is in fact skipped
outright below (patch_width==1 means there is no second shift to compare),
which is the expected null control rather than a bug.

Reuses an already-extracted run's config/models/data -- no re-extraction,
one `adapter.predict()` call per (model, shift) -- far cheaper than any
patching/extraction stage.

Each model's own shift range is derived from `token_time_spans()`'s median
span width (the same local computation `analysis/attention.py` already
does for its own `token_width` -- there is no fixed adapter/config
attribute for "patch width" today, confirmed by grepping the adapters),
not a hardcoded 32: this probe should mean the same thing for a model
whose patch width isn't 32 too.

Example:
    python run_phase_sensitivity_sweep.py --run runs/medium_run_chronos_base \\
        --max-series 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.analysis.phase_sensitivity import phase_sensitivity_stats
from tsfm_lens.analysis.stats import mase as _mase
from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.models import ModelHub
from tsfm_lens.utils import log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def _predict_point_trimmed(adapter, contexts: np.ndarray, shift: int, horizon: int) -> np.ndarray:
    """Forecast from a context with `shift` points trimmed off the front,
    keeping the endpoint and total array length fixed.

    A naive `contexts[:, shift:]` slice shrinks the array to
    `context_len - shift`, which is fine for per-timestep tokenizers
    (Chronos) but crashes patch-tokenized models like TimesFM 2.5: its own
    `decode()` does `torch.reshape(inputs, (batch, -1, patch_len))`, which
    requires the context length to stay a multiple of the patch width
    (32) -- confirmed by reading `timesfm_2p5_torch.py::decode` directly,
    not guessed. There is no earlier history available to extend the
    front into instead (`BenchmarkData.contexts()` is exactly
    `context_len` long, per `data.py`), so the length can't be preserved
    by using more real data.

    The model's own `decode()` already supports variable-length effective
    context via its `masks` argument (True = padded/invalid, excluded from
    per-patch revin normalization and zeroed post-norm -- read directly off
    `decode()`'s use of `update_running_stats`/`torch.where(patched_masks,
    0.0, ...)`), which is exactly the semantics "trim s points off the
    front" needs: front-pad the removed region with masked placeholder
    values so the tensor stays at the original, patch-aligned length, with
    the real (trimmed) content right-aligned so the endpoint doesn't move.
    This calls the adapter's own `tfm.model.decode` directly (the same
    method `TimesFMAdapter.predict` calls) with an explicit mask, since the
    generic `ModelAdapter.predict(contexts, horizon, quantiles)` contract
    has no mask parameter and every other adapter's `predict` doesn't need
    one -- kept local to this script rather than widening that shared
    contract for a single probe's edge case.
    """
    if not hasattr(adapter, "tfm"):
        # Per-timestep tokenizers have no patch-alignment constraint --
        # the plain trimmed slice works as-is (this path is unreached today
        # since such models are skipped upstream as the null control, but
        # kept as the generic fallback rather than assuming TimesFM-only).
        pred = adapter.predict(contexts[:, shift:], horizon, [0.5])
        return pred["point"]
    n, context_len = contexts.shape
    values = np.zeros((n, context_len), dtype=np.float32)
    values[:, shift:] = contexts[:, shift:]
    mask = np.zeros((n, context_len), dtype=bool)
    mask[:, :shift] = True
    inputs = torch.from_numpy(values).to(adapter.device)
    masks = torch.from_numpy(mask).to(adapter.device)
    with torch.no_grad():
        renormed_outputs, _, ar_outputs = adapter.tfm.model.decode(horizon, inputs, masks)
    full = renormed_outputs[:, -1, ...]
    if ar_outputs is not None:
        extra = ar_outputs.reshape(inputs.shape[0], -1, adapter.tfm.model.q)
        full = torch.cat([full, extra], dim=1)
    full = full[:, :horizon, :].float().cpu().numpy()
    return full[:, :, adapter._point_idx]


def _patch_width(adapter) -> int:
    """Median token span width from `token_time_spans()` -- the model's own
    declared tokenization granularity, needing no forward pass to compute.
    """
    spans = adapter.token_time_spans()
    widths = spans[:, 1] - spans[:, 0]
    return max(1, int(round(float(np.median(widths)))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md sec 16 E17: patch-boundary phase-sensitivity probe")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--max-series", type=int, default=64)
    parser.add_argument("--max-shift", type=int, default=None,
                        help="cap the shift range (default: each model's own patch width)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    base_cfg = load_config(run_dir / "config_resolved.yaml")
    set_seed(base_cfg.run.seed)
    device = resolve_device(base_cfg.run.device)
    dtype = resolve_dtype(base_cfg.run.dtype, device)
    hub = ModelHub(base_cfg.models, base_cfg.data, device, dtype)
    data = load_benchmark(base_cfg.data, base_cfg.run.seed)
    model_a, model_b = base_cfg.comparison_pair()

    # A fresh rng offset (+5) not already claimed by run_l3 (+3) or the
    # quantization-churn probe's own draws, so this probe's row sample is
    # reproducible on its own terms without colliding with either.
    rng_rows = np.random.default_rng(base_cfg.run.seed + 5)
    n_rows = min(data.n, args.max_series)
    rows = np.sort(rng_rows.choice(data.n, size=n_rows, replace=False))
    contexts = data.contexts()[rows]
    targets = data.targets()[rows]
    horizon = base_cfg.data.horizon

    results = {}
    for mcfg in (model_a, model_b):
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        patch_width = _patch_width(adapter)
        max_shift = min(args.max_shift, patch_width) if args.max_shift else patch_width
        if max_shift < 2:
            # Per-timestep tokenization (e.g. Chronos-T5/Bolt: one token per
            # kept context step) has no patch boundary to be sensitive to --
            # this is the expected null control, not a bug. Skip with a log
            # rather than crash inside phase_sensitivity_stats's >=2-shifts
            # requirement (CLAUDE.md sec 2.5).
            log.info("phase-sensitivity probe: %s has patch_width=%d (no patch boundary to "
                     "test); skipping as the expected null control", mcfg.name, patch_width)
            if not base_cfg.run.keep_models_loaded:
                hub.release(mcfg.name)
            continue
        log.info("phase-sensitivity probe: %s patch_width=%d shifts=0..%d",
                 mcfg.name, patch_width, max_shift - 1)

        mase_by_shift = {}
        for shift in range(max_shift):
            if shift == 0:
                point = adapter.predict(contexts, horizon, [0.5])["point"]
            else:
                point = _predict_point_trimmed(adapter, contexts, shift, horizon)
            # MASE's own naive-forecast denominator is computed from the
            # REAL trimmed context (contexts[:, shift:]), not the
            # zero-padded array `_predict_point_trimmed` feeds the model --
            # the padding is a model-input device only, and must not also
            # rescale what MASE calls "the naive in-sample baseline".
            mase_by_shift[shift] = _mase(point, targets, contexts[:, shift:])

        stats = phase_sensitivity_stats(mase_by_shift, seed=base_cfg.run.seed)
        stats["patch_width"] = patch_width
        results[mcfg.name] = stats
        cv = stats["cv"]
        log.info("%s: cv=%s worst-best MASE=%.4f (shift %d vs %d)",
                 mcfg.name,
                 f"{cv['value']:.4f} [{cv['lo']:.4f},{cv['hi']:.4f}]" if cv else "n/a",
                 stats["worst_minus_best_mase"], stats["worst_shift"], stats["best_shift"])
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    out = Path(args.out) if args.out else run_dir.parent / "phase_sensitivity_sweep.json"
    save_json(out, {"run": str(run_dir), "n_series": int(n_rows), "models": results})

    print(f"\nwrote {out}\n")
    print(f"{'model':>20}  {'patch_width':>11}  {'cv [CI]':>26}  {'worst-best MASE':>16}")
    for name, s in results.items():
        cv = s["cv"]
        cv_str = f"{cv['value']:.4f} [{cv['lo']:.4f},{cv['hi']:.4f}]" if cv else "n/a"
        print(f"{name:>20}  {s['patch_width']:>11}  {cv_str:>26}  {s['worst_minus_best_mase']:>16.4f}")


if __name__ == "__main__":
    main()
