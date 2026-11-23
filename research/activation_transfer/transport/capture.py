"""Capture matched native activation contrasts and patch the original frozen models."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
from tsfm_lens.models.timesfm_adapter import TimesFMAdapter

from research.activation_transfer.transport.prepare import CONTEXT, DEFAULT_CACHE, HORIZON


REVISIONS = {"bolt": ("amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8"),
             "timesfm": ("google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3")}


def load_adapter(name, *, random_init=False):
    """Use pinned local checkpoint snapshots through the repository's prediction interface."""
    from huggingface_hub import snapshot_download
    repo, revision = REVISIONS[name]
    path = snapshot_download(repo, revision=revision, local_files_only=True)
    data = DataConfig(context_len=CONTEXT, horizon=HORIZON)
    cls = {"bolt": ChronosBoltAdapter, "timesfm": TimesFMAdapter}[name]
    config = ModelConfig(name=name, checkpoint=str(path), random_init=random_init)
    adapter = cls(config, data, torch.device("cpu"), torch.float32)
    adapter.ensure_loaded()
    return adapter


def site(adapter):
    """Return the middle block and contiguous patch slice covering the last 32 samples."""
    all_layers = adapter.all_layer_names()
    layer = all_layers[len(all_layers) // 2]
    spans = adapter.token_time_spans()
    selected = np.flatnonzero(spans[:, 0] >= CONTEXT - 32)
    if (len(selected) == 0 or spans[selected[0], 0] != CONTEXT - 32 or
            spans[selected[-1], 1] != CONTEXT or np.any(np.diff(selected) != 1) or
            not np.array_equal(spans[selected[:-1], 1], spans[selected[1:], 0])):
        raise ValueError("the chosen native tokens must exactly tile the last 32 input positions")
    return layer, slice(int(selected[0]), int(selected[-1]) + 1)


def run_pass(adapter, x, delta=None, batch_size=48):
    """Return flattened clean patch inputs and median forecasts under an optional additive patch."""
    layer, tokens = site(adapter)
    states, forecasts = [], []
    if delta is not None and (len(delta) != len(x) or not np.isfinite(delta).all()):
        raise ValueError("edits must have one finite row per series")
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size]
        captured = []

        def hook(module, args, kwargs):
            value = args[0] if args else kwargs["hidden_states"]
            activation = value[:, tokens, :]
            if captured or activation.ndim != 3 or len(activation) != len(batch):
                raise RuntimeError(f"unexpected {layer} invocation or geometry")
            captured.append(activation.detach().cpu().flatten(1).numpy().copy())
            if delta is None:
                return
            edited = value.clone()
            change = torch.as_tensor(delta[i:i + len(batch)], device=value.device, dtype=value.dtype)
            if tuple(change.shape) != tuple(captured[0].shape):
                raise ValueError("edit width must match the concatenated token dimensions")
            edited[:, tokens, :] += change.reshape_as(activation)
            if args:
                return (edited, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": edited}

        handle = adapter.module.get_submodule(layer).register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                forecast = adapter.predict(batch, HORIZON, [0.5])["quantiles"][..., 0]
        finally:
            handle.remove()
        if len(captured) != 1 or forecast.shape != (len(batch), HORIZON):
            raise RuntimeError("the forecast path must hit the chosen layer exactly once")
        states.append(captured[0])
        forecasts.append(forecast)
    return np.concatenate(states), np.concatenate(forecasts)


def signed_response(adapter, x, edits, batch_size=48):
    """Central forecast effect on each original history for each positive-minus-negative edit."""
    n, concepts, width = edits.shape
    doubled = np.repeat(x, concepts, axis=0)
    changes = edits.reshape(n * concepts, width)
    plus = run_pass(adapter, doubled, changes, batch_size)[1]
    minus = run_pass(adapter, doubled, -changes, batch_size)[1]
    return ((plus - minus) / 2).reshape(n, concepts, HORIZON)


def capture(name, cache, batch_size):
    """Collect clean and altered states, native patch effects and natural input effects."""
    with np.load(cache / "data.npz") as data:
        x, waves, dose = data["x"], data["patterns"], float(data["dose"])
    adapter = load_adapter(name)
    print(f"{name} loaded at {site(adapter)}", flush=True)
    clean, forecast = run_pass(adapter, x, batch_size=batch_size)
    single, baseline = run_pass(adapter, x[:3], batch_size=1)
    np.testing.assert_allclose(clean[:3], single, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(forecast[:3], baseline, rtol=5e-4, atol=5e-4)
    _, zero = run_pass(adapter, x[:3], np.zeros_like(single), batch_size=batch_size)
    np.testing.assert_allclose(forecast[:3], zero, rtol=5e-4, atol=5e-4)
    delta, input_effect = [], []
    for w in waves:
        high, f_high = run_pass(adapter, x + dose * w, batch_size=batch_size)
        low, f_low = run_pass(adapter, x - dose * w, batch_size=batch_size)
        delta.append((high - low) / 2)
        input_effect.append((f_high - f_low) / 2)
    delta = np.stack(delta, axis=1)
    print(f"{name}: {delta.shape} state contrasts captured", flush=True)
    native = signed_response(adapter, x, delta, batch_size=batch_size)
    layer, tokens = site(adapter)
    metadata = {"checkpoint": REVISIONS[name], "layer": layer, "tokens": [tokens.start, tokens.stop],
                "torch": str(torch.__version__), "batch_size": batch_size,
                "max_clean_single_batch_diff": float(np.max(abs(clean[:3] - single))),
                "max_zero_edit_forecast_diff": float(np.max(abs(forecast[:3] - zero)))}
    temp_path = cache / (name + ".partial.npz")
    np.savez_compressed(temp_path, clean=clean, delta=delta, input_effect=np.stack(input_effect, axis=1),
                        native=native, forecast=forecast, metadata=json.dumps(metadata))
    os.replace(temp_path, cache / (name + ".npz"))
    print(f"{name}: reference and capture saved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=REVISIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    capture(args.model, args.cache, args.batch_size)


if __name__ == "__main__":
    main()
