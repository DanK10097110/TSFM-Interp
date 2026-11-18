"""Offline paired TSFM activation extraction with preregistered seed partitions.

Run from repository root with PYTHONPATH=tsfm_lens and GPU visibility set by
the caller. This writes only new artifacts; it never reads benchmark corpora.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.extraction.hooks import _primary, _rebuild
from tsfm_lens.extraction.alignment import run_alignment_gate
from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
from tsfm_lens.models.timesfm_adapter import TimesFMAdapter


def paired_corpus():
    """Create192 generating seeds, balanced periods and shared pair noise/future."""
    rng = np.random.default_rng(12026)
    periods = np.concatenate([rng.permutation(np.tile([16, 24, 32, 48], n // 4))
                              for n in [96, 48, 48]])
    xs, futures, clean, metadata = [], [], [], []
    t = np.arange(160)
    for seed, period in enumerate(periods):
        rr = np.random.default_rng(771000 + seed)
        amplitude, phase = rr.uniform(.5, 2), rr.uniform(-np.pi, np.pi)
        slope, noise = rr.uniform(-1, 1), rr.uniform(.05, .25)
        eps = rr.normal(0, noise, len(t))
        base = slope * (t - 64) / 128
        for dose in [0, 1]:
            latent = base + dose * amplitude * np.sin(2 * np.pi * t / period + phase)
            observed = latent + eps
            xs.append(observed[:128]); futures.append(observed[128:]); clean.append(latent[128:])
            metadata.append(dict(seed=seed, split='train' if seed < 96 else 'validation' if seed < 144 else 'test',
                                 dose=dose, period=int(period), phase=phase, amplitude=amplitude,
                                 slope=slope, noise=noise, scale=amplitude ** 2))
    return np.asarray(xs, dtype=np.float32), np.asarray(futures), np.asarray(clean), metadata


def run(out: Path):
    """Capture three frozen layers, then verify self and counterfactual reach."""
    out.mkdir(parents=True, exist_ok=False)
    x, future, clean, metadata = paired_corpus()
    protocol = dict(evidence='correlational decoding; within-model natural donor patch causal only',
                    split='seed<96 train;96<=seed<144 validation;144<=seed<192 test',
                    candidate_layers='first,middle,last', pca_rank=16, ridge_alpha=[.01, 1, 100],
                    selection='validation only; freeze before test', context=128, horizon=32,
                    reach_gate='repeat0,selfpatch0,donor_delta>1e-7*forecast_RMS',
                    locality='geometry/token count only; no empirical lag-localization claim')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'corpus.npz', context=x, future=future, clean_future=clean)
    summary = {}
    for cls, name, checkpoint in [(ChronosBoltAdapter, 'bolt', 'amazon/chronos-bolt-small'),
                                  (TimesFMAdapter, 'timesfm', 'google/timesfm-2.5-200m-pytorch')]:
        started = time.perf_counter()
        adapter = cls(ModelConfig(name=name, checkpoint=checkpoint), DataConfig(context_len=128, horizon=32),
                      torch.device('cuda:0'), torch.float32)
        adapter.ensure_loaded()
        all_layers = adapter.all_layer_names()
        layers = [all_layers[i] for i in [0, len(all_layers) // 2, len(all_layers) - 1]]
        alignment = run_alignment_gate(adapter, 32, layers)
        modules = dict(adapter.module.named_modules())
        captured, predictions = {k: [] for k in layers}, []
        handles = []
        for layer in layers:
            def capture(module, inputs, output, key=layer):
                hidden = adapter.postprocess_tokens(key, _primary(output))
                if hidden.shape[1] != len(adapter.token_time_spans()):
                    raise ValueError('token geometry does not match captured context')
                captured[key].append(hidden.detach().float().cpu().numpy())
            handles.append(modules[layer].register_forward_hook(capture))
        for start in range(0, len(x), 8):
            predictions.append(adapter.predict(x[start:start + 8], 32, [.5])['point'])
        for h in handles:
            h.remove()
        prediction = np.concatenate(predictions)
        tokens = {k: np.concatenate(v) for k, v in captured.items()}
        repeat = adapter.predict(x[:8], 32, [.5])['point']
        repeat_error = float(np.max(np.abs(repeat - prediction[:8])))
        gates = {}
        for layer in layers:
            raw = []
            handle = modules[layer].register_forward_hook(lambda m, i, o: raw.append(_primary(o).detach().clone()))
            baseline = adapter.predict(x[:2], 32, [.5])['point']
            handle.remove()
            def patch(m, i, o, donor=False):
                value = raw[0].flip(0) if donor else raw[0]
                return _rebuild(o, value)
            handle = modules[layer].register_forward_hook(lambda m, i, o: patch(m, i, o))
            self_prediction = adapter.predict(x[:2], 32, [.5])['point']; handle.remove()
            handle = modules[layer].register_forward_hook(lambda m, i, o: patch(m, i, o, True))
            donor_prediction = adapter.predict(x[:2], 32, [.5])['point']; handle.remove()
            self_error = float(np.max(np.abs(self_prediction - baseline)))
            donor_delta = float(np.max(np.abs(donor_prediction - baseline)))
            floor = 1e-7 * max(float(np.sqrt(np.mean(baseline ** 2))), 1e-12)
            gates[layer] = dict(self_max_abs=self_error, donor_max_abs=donor_delta, floor=floor)
            if repeat_error != 0 or self_error != 0 or donor_delta <= floor:
                raise RuntimeError(f'failed reach gate {name} {layer}: {gates[layer]} repeat={repeat_error}')
        np.savez_compressed(out / f'{name}.npz', prediction=prediction, **tokens)
        summary[name] = dict(layers=layers, shapes={k: list(v.shape) for k, v in tokens.items()},
                             repeat_max_abs=repeat_error, gates=gates, alignment=alignment,
                             seconds=time.perf_counter()-started)
        print(json.dumps({name: summary[name]}), flush=True)
        del adapter, tokens, captured, raw, modules
        gc.collect(); torch.cuda.empty_cache()
    (out / 'extraction.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.out)
