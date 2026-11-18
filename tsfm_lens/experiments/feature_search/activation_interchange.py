"""Direction-specific distributed natural-donor interventions, with held-out tests.

Response SVD and covariance controls use training seeds only. Layer and rank
are chosen jointly on bidirectional validation fidelity error divided by the
whole-token swap response energy. The ceiling uses receiver normalization; its discrepancy
from the true donor prediction is retained. No feature-specific claim follows.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
from tsfm_lens.experiments.feature_search.statistics import compare_effects, paired_effect


def response_bases(clean, donor, *, max_rank=8, seed=41, covariance_samples=None):
    """Fit response, clean top-PC and covariance-shaped random channel frames."""
    delta = (donor-clean).reshape(-1, clean.shape[-1]).astype(np.float64)
    clean = clean if covariance_samples is None else covariance_samples
    clean = clean.reshape(-1, clean.shape[-1]).astype(np.float64)
    clean -= clean.mean(axis=0)
    rank = min(max_rank, min(delta.shape), min(clean.shape))
    response = randomized_svd(delta, n_components=rank, random_state=seed)[2].T
    top = randomized_svd(clean, n_components=rank, random_state=seed)[2].T
    rng = np.random.default_rng(seed)
    random = []
    for _ in range(16):
        frame, _ = np.linalg.qr(clean.T @ rng.normal(size=(len(clean), rank)))
        random.append(frame)
    return response, top, random


def projected_replacement(receiver, donor, basis, *, target_energy=None):
    """Project natural donor differences per token, optionally match edit norms."""
    delta = donor-receiver
    edit = (delta @ basis) @ basis.T
    if target_energy is not None:
        norm = np.sqrt(np.sum(edit**2, axis=-1, keepdims=True))
        if np.any((norm < 1e-12) & (target_energy > 1e-12)):
            raise ValueError('Cannot energy-match a zero control projection')
        edit *= np.divide(target_energy, norm, out=np.zeros_like(norm), where=norm > 1e-12)
    return (receiver+edit).astype(np.float32), np.sqrt(np.sum(edit**2, axis=-1, keepdims=True))


def patched_predict(adapter, contexts, layer=None, replacement=None):
    """Use the actual forecast path and adapter token positions, preserving REG."""
    outputs = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for start in range(0, len(contexts), 8):
            batch = contexts[start:start+8]
            if replacement is None:
                output = adapter.predict(batch, 32, [.5])['point']
            else:
                values = torch.as_tensor(replacement[start:start+8], device=adapter.device, dtype=torch.float32)
                with token_patch(adapter.module, layer, adapter.token_slice, values):
                    output = adapter.predict(batch, 32, [.5])['point']
            outputs.append(output)
    return np.concatenate(outputs)


def paired_rows(metadata, split, reverse=False):
    """Match absent/present rows by seed without assuming artifact row order."""
    seeds = sorted({row['seed'] for row in metadata if row['split'] == split})
    lookup = {(row['seed'], row['dose']): i for i, row in enumerate(metadata)}
    receiver = np.array([lookup[s, int(reverse)] for s in seeds])
    donor = np.array([lookup[s, 1-int(reverse)] for s in seeds])
    return receiver, donor, np.asarray(seeds)


def mse(a, b):
    """Per-series mean squared forecast discrepancy."""
    return np.mean((a-b)**2, axis=1)


def run(source, out):
    """Freeze validation choices, then evaluate two models and both directions."""
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    extraction = json.loads((source/'extraction.json').read_text(encoding='utf-8'))
    corpus = np.load(source/'corpus.npz')
    contexts, clean_future = corpus['context'], corpus['clean_future']
    protocol = dict(version=2, evidence='exploratory after aborted v1 evaluation; no confirmatory reinterpretation', covariance='union of absent and present training activations', ranks=[1, 4, 8], selection='bidirectional sum mean fidelity MSE / sum mean full-swap response MSE',
                    primary='equal-seed fidelity gain over sixteen covariance-shaped equal-edit-energy random frames',
                    family='two models by two directions, Holm signflip plus-one',
                    controls=['self', 'zero', 'full_swap', 'top_PC_equal_energy', 'covariance_random_equal_energy'],
                    caution='direction-specific distributed intervention; not a specific feature or receiver-future improvement',
                    precision='FP32, no autocast; batch8; training-only SVD and covariance', random_seed=41, random_draws=16, random_generator='single default_rng(41), 16 sequential frames per layer')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report, primary = {}, {}
    for name, cls, checkpoint in [('bolt', ChronosBoltAdapter, 'amazon/chronos-bolt-small'),
                                  ('timesfm', TimesFMAdapter, 'google/timesfm-2.5-200m-pytorch')]:
        adapter = cls(ModelConfig(name=name, checkpoint=checkpoint), DataConfig(context_len=128, horizon=32),
                      torch.device('cuda:0'), torch.float32)
        adapter.ensure_loaded()
        if not adapter.forecast_reads_patched_positions():
            report[name] = dict(skipped='adapter declares forecast does not read patched positions')
            del adapter
            continue
        cache = np.load(source/f'{name}.npz')
        train_receiver, train_donor, _ = paired_rows(metadata, 'train')
        candidates, frames, gates = [], {}, []
        for layer in extraction[name]['layers']:
            tokens = cache[layer]
            frames[layer] = response_bases(tokens[train_receiver], tokens[train_donor], covariance_samples=np.concatenate((tokens[train_receiver], tokens[train_donor])))
            direction_scores = {rank: [] for rank in [1,4,8]}
            reached = True
            for reverse in [False, True]:
                receiver, donor, _ = paired_rows(metadata, 'validation', reverse)
                base = patched_predict(adapter, contexts[receiver])
                self_prediction = patched_predict(adapter, contexts[receiver], layer, tokens[receiver])
                full = patched_predict(adapter, contexts[receiver], layer, tokens[donor])
                self_error = float(np.max(np.abs(base-self_prediction)))
                ceiling = float(np.mean(mse(full, base)))
                floor = 1e-7 * max(float(np.sqrt(np.mean(base**2))), 1e-12)
                gate = dict(layer=layer, direction='remove' if reverse else 'add', self_max_abs=self_error, ceiling_response_mse=ceiling, reach_max_abs=float(np.max(np.abs(full-base))), floor=floor)
                gates.append(gate)
                if self_error != 0 or np.max(np.abs(full-base)) <= floor:
                    reached = False
                    print(json.dumps(dict(model=name, layer=layer, direction=int(reverse), skipped='self or reach gate', self_error=self_error, ceiling=ceiling)), flush=True)
                    break
                for rank in [1,4,8]:
                    replacement, _ = projected_replacement(tokens[receiver], tokens[donor], frames[layer][0][:,:rank])
                    prediction = patched_predict(adapter, contexts[receiver], layer, replacement)
                    direction_scores[rank].append((float(np.mean(mse(prediction, full))), ceiling))
            if reached:
                candidates.extend(dict(layer=layer, rank=rank, validation_mse=float(np.mean([v[0] for v in values])), normalized_error=float(sum(v[0] for v in values)/sum(v[1] for v in values))) for rank,values in direction_scores.items())
        if not candidates:
            report[name] = dict(skipped='no layers passed actual forecast reach/self gates', gates=gates)
            del adapter
            torch.cuda.empty_cache()
            continue
        selected = min(candidates, key=lambda row: row['normalized_error'])
        layer, rank = selected['layer'], selected['rank']
        print(json.dumps(dict(model=name, frozen_selection=selected)), flush=True)
        tokens = cache[layer]
        learned_basis, top_basis, random_bases = frames[layer]
        model_result = dict(selection=selected, validation_candidates=candidates, gates=gates, directions={})
        saved = {}
        for reverse in [False, True]:
            direction = 'remove' if reverse else 'add'
            receiver, donor, seeds = paired_rows(metadata, 'test', reverse)
            predictions = dict(base=patched_predict(adapter, contexts[receiver]),
                               donor=patched_predict(adapter, contexts[donor]))
            predictions['self'] = patched_predict(adapter, contexts[receiver], layer, tokens[receiver])
            if np.max(np.abs(predictions['self']-predictions['base'])) != 0:
                raise RuntimeError('Evaluation self-patch failed exact same-batch identity')
            predictions['zero'] = patched_predict(adapter, contexts[receiver], layer, np.zeros_like(tokens[receiver]))
            predictions['full_swap'] = patched_predict(adapter, contexts[receiver], layer, tokens[donor])
            replacement, energy = projected_replacement(tokens[receiver], tokens[donor], learned_basis[:,:rank])
            predictions['learned'] = patched_predict(adapter, contexts[receiver], layer, replacement)
            top, _ = projected_replacement(tokens[receiver], tokens[donor], top_basis[:,:rank], target_energy=energy)
            predictions['top_PC'] = patched_predict(adapter, contexts[receiver], layer, top)
            for i, frame in enumerate(random_bases):
                control, _ = projected_replacement(tokens[receiver], tokens[donor], frame[:,:rank], target_energy=energy)
                predictions[f'random_{i}'] = patched_predict(adapter, contexts[receiver], layer, control)
            ceiling = predictions['full_swap']
            fidelity = {key: mse(pred, ceiling) for key,pred in predictions.items()}
            donor_loss = {key: mse(pred, clean_future[donor]) for key,pred in predictions.items()}
            random_loss = np.mean([fidelity[f'random_{i}'] for i in range(16)], axis=0)
            primary[f'{name}_{direction}'] = (random_loss, fidelity['learned'])
            ceiling_delta = float(np.mean(mse(ceiling, predictions['base'])))
            error = predictions['learned']-ceiling
            time = np.linspace(-1, 1, error.shape[1])
            model_result['directions'][direction] = dict(
                fidelity_level_offset_rms=float(np.sqrt(np.mean(error.mean(axis=1)**2))),
                fidelity_slope_error_rms=float(np.sqrt(np.mean((error@time/(time@time))**2))),
                n_seeds=len(seeds), top_PC_minus_learned_fidelity=paired_effect(fidelity['top_PC'], fidelity['learned'], seeds, n_boot=2000, n_permutations=1999, seed=41), fidelity_mse={k:float(v.mean()) for k,v in fidelity.items()},
                donor_clean_future_mse={k:float(v.mean()) for k,v in donor_loss.items()},
                donor_clean_future_loss_change={k:float((v-donor_loss['base']).mean()) for k,v in donor_loss.items()},
                normalized_full_response_fidelity=None if ceiling_delta <= 1e-14 else 1-float(fidelity['learned'].mean())/ceiling_delta,
                full_swap_to_true_donor_mse=float(mse(ceiling, predictions['donor']).mean()),
                full_swap_response_mse=ceiling_delta, self_max_abs=float(np.max(np.abs(predictions['self']-predictions['base']))),
                edit_energy_mean=float(np.mean(energy**2)),
                random_fidelity_mse_per_draw=[float(fidelity[f'random_{i}'].mean()) for i in range(16)],
                nuisance='no dedicated off-target intervention yet; no feature-specific claim')
            for key, value in predictions.items():
                saved[f'{direction}_{key}'] = value
            saved[f'{direction}_seeds'] = seeds
            saved[f'{direction}_donor_clean_future'] = clean_future[donor]
        report[name] = model_result
        np.savez_compressed(out/f'{name}_predictions.npz', **saved)
        (out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        del adapter, cache, tokens, frames
        gc.collect(); torch.cuda.empty_cache()
    seeds = paired_rows(metadata, 'test')[2]
    report['primary_comparisons'] = compare_effects(primary, seeds, n_boot=2000, n_permutations=1999, seed=41)
    (out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1., 6e9 / torch.cuda.get_device_properties(0).total_memory))
    run(args.source, args.out)
