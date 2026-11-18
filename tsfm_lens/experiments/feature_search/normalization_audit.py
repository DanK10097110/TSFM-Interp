"""Audit known native output-normalization side channels without fitting weights.

These architecture mechanisms are documented in the installed model code; this
is a pipeline coverage diagnostic, not a novelty claim. Supported surfaces are
ChronosBoltAdapter's native InstanceNorm and TimesFMAdapter's direct 2.5 decode
path for a single output patch. Other adapters require an explicit native audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def normalization_state(contexts, surface, *, horizon=32, bolt_context_length=2048):
    """Read native FP32 output location/scale without loading model parameters.

    TimesFM population variance is accumulated by its native running-statistics
    helper. The inverse uses true sigma, even when input normalization uses a
    denominator of one for sigma<1e-6. Bolt maps zero sigma to its native eps.
    Finite unmasked fixed-length contexts are supported here; other preparation
    paths are rejected rather than approximated.
    """
    contexts = np.asarray(contexts)
    if contexts.ndim != 2 or not np.isfinite(contexts).all() or contexts.shape[1] == 0:
        raise ValueError('Need finite nonempty two-dimensional contexts')
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Need a positive integer horizon')
    values = torch.as_tensor(contexts, dtype=torch.float32)
    if surface == 'chronos_bolt':
        from chronos.chronos_bolt import InstanceNorm
        if horizon > 64:
            raise ValueError('Bolt audit supports only a single native64-step output patch')
        values = values[:, -bolt_context_length:]
        _, (mu, sigma) = InstanceNorm()(values)
        return mu.numpy(), sigma.numpy()
    if surface == 'timesfm_2p5_decode':
        from timesfm.torch.util import update_running_stats
        from timesfm.timesfm_2p5.timesfm_2p5_base import TimesFM_2p5_200M_Definition
        native = TimesFM_2p5_200M_Definition()
        patch_len, output_len = native.input_patch_len, native.output_patch_len
        if values.shape[1] % patch_len or horizon > output_len:
            raise ValueError('TimesFM decode audit needs whole32-step patches and horizon<=128')
        n = torch.zeros(len(values))
        mu, sigma = n.clone(), n.clone()
        for patch in values.reshape(len(values), -1, patch_len).unbind(dim=1):
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, patch, torch.zeros_like(patch, dtype=torch.bool))
        return mu.numpy()[:, None], sigma.numpy()[:, None]
    raise ValueError(f'Unsupported adapter normalization surface: {surface}')


def transplant_output_normalizer(donor_prediction, donor_mu, donor_sigma, receiver_mu, receiver_sigma):
    """Move a donor forecast through receiver affine output units with no fit.

    Zero donor sigma destroys normalized-output information and cannot be
    inverted. This includes constant-context TimesFM outputs. Such cases are
    unidentifiable rather than evidence against a side channel.
    """
    prediction = np.asarray(donor_prediction, dtype=np.float64)
    states = [np.asarray(x, dtype=np.float64) for x in (donor_mu, donor_sigma, receiver_mu, receiver_sigma)]
    if prediction.ndim != 2 or any(x.shape != (len(prediction), 1) or not np.isfinite(x).all() for x in states) or not np.isfinite(prediction).all():
        raise ValueError('Need finite predictions and per-series column locations/scales')
    donor_mu, donor_sigma, receiver_mu, receiver_sigma = states
    if np.any(donor_sigma <= 0) or np.any(receiver_sigma < 0):
        raise ValueError('Donor output sigma must be positive; zero sigma is unidentifiable')
    return (prediction-donor_mu)/donor_sigma*receiver_sigma+receiver_mu


def compare_sidechannel(donor_prediction, full_swap, donor_mu, donor_sigma, receiver_mu, receiver_sigma):
    """Measure how much fixed affine correction explains full-swap discrepancy."""
    full_swap = np.asarray(full_swap)
    expected = transplant_output_normalizer(donor_prediction, donor_mu, donor_sigma, receiver_mu, receiver_sigma)
    if full_swap.shape != expected.shape or not np.isfinite(full_swap).all():
        raise ValueError('Need finite actual full-swap predictions matching donor shape')
    raw = float(np.mean((full_swap-donor_prediction)**2))
    residual = float(np.mean((full_swap-expected)**2))
    return dict(raw_donor_discrepancy_mse=raw, affine_corrected_residual_mse=residual,
                discrepancy_fraction_explained=None if raw == 0 else 1-residual/raw,
                corrected_max_abs=float(np.max(np.abs(full_swap-expected))),
                evidence_class='representation_plus_normalizer_coverage', fitted_parameters=0)


def run(source, interchange, out):
    """Audit both saved directions using actual adapter forecast-path surfaces."""
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    corpus = np.load(source/'corpus.npz')
    contexts = corpus['context']
    lookup = {(row['seed'], row['dose']): i for i,row in enumerate(metadata)}
    result = dict(caution='Known architecture mechanism, not novel; residual includes unpatched REG/downstream paths and numeric precision', models={})
    arrays = {}
    for name, surface in [('bolt', 'chronos_bolt'), ('timesfm', 'timesfm_2p5_decode')]:
        predictions = np.load(interchange/f'{name}_predictions.npz')
        mu, sigma = normalization_state(contexts, surface)
        entries = {}
        for direction, dose in [('add',0), ('remove',1)]:
            seeds = predictions[f'{direction}_seeds']
            receiver = np.array([lookup[int(seed),dose] for seed in seeds])
            donor = np.array([lookup[int(seed),1-dose] for seed in seeds])
            donor_prediction = predictions[f'{direction}_donor']
            full_swap = predictions[f'{direction}_full_swap']
            entries[direction] = compare_sidechannel(donor_prediction, full_swap, mu[donor], sigma[donor], mu[receiver], sigma[receiver])
            arrays[f'{name}_{direction}_expected'] = transplant_output_normalizer(donor_prediction, mu[donor], sigma[donor], mu[receiver], sigma[receiver])
        result['models'][name] = dict(surface=surface, directions=entries)
    np.savez_compressed(out/'affine_predictions.npz', **arrays)
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--interchange', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.interchange, args.out)
