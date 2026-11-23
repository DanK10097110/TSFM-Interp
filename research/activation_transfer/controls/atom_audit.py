"""Independently audit numerical replay and regularization in the literal atom study."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def spectral_support(states, directions, measured, relative_tolerance=1e-7):
    """Compare empirical row-space projection with the study's spectral ridge shrinkage."""
    x = states.astype(float) - states.mean(axis=0, dtype=float)
    w = directions.astype(float)
    _, singular, right = np.linalg.svd(x, full_matrices=False)
    keep = singular > relative_tolerance * singular[0]
    alpha = .01 * singular[0]**2
    coord = w @ right.T
    ridge = (coord * (singular**2 / (singular**2 + alpha))) @ right
    np.testing.assert_allclose(ridge, measured, rtol=2e-4, atol=2e-6)
    ideal = np.linalg.norm(coord[:, keep], axis=1) / np.linalg.norm(w, axis=1)
    actual = np.linalg.norm(ridge, axis=1) / np.linalg.norm(w, axis=1)
    return dict(matrix_shape=list(x.shape), rank=int(keep.sum()), relative_singular_threshold=relative_tolerance,
                regularizer=float(alpha), modes_above_half_response=int((singular**2 > alpha).sum()),
                row_span_norm_fraction=ideal.tolist(), ridge_norm_fraction=actual.tolist())


def numerical_summary(zero, real, row, atom, split, names, feature_count):
    ans = {}
    for name in names:
        subset = split[row] == name
        error = zero[subset].astype(float)
        signal = real[row[subset], atom[subset]].astype(float)
        norm = lambda array: np.sqrt(np.mean(array * array, axis=-1))
        noise_rms, signal_rms = norm(error), norm(signal)
        per = []
        for index in range(feature_count):
            take = atom[subset] == index
            if not take.any():
                per.append(None)
                continue
            per.append(dict(n=int(take.sum()), noise_rms=float(norm(error[take].reshape(-1))),
                            effect_rms=float(norm(signal[take].reshape(-1)))))
        ans[name] = dict(n_pairs=len(signal), max_abs_noise=float(np.abs(error).max()),
                         max_pair_noise_rms=float(noise_rms.max()),
                         p99_pair_noise_rms=float(np.quantile(noise_rms, .99)),
                         pair_effect_rms_within_10x_noise=int((signal_rms <= 10*noise_rms).sum()),
                         min_active_atom_effect_rms=min(a['effect_rms'] for a in per if a), per_atom=per)
    return ans


def audit(cache, model, batch=24):
    # Keep model-specific dependencies out of the numerical support / accounting unit tests.

    from research.activation_transfer.boundary.capture import MODELS, adapter_for
    from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
    with np.load(cache / 'data.npz') as d:
        train = d['split'] == 'train'
        ids = d['ids']; split = d['split'][~train]; x = d['x'][~train]
    with np.load(cache / (model+'.npz')) as d:
        np.testing.assert_array_equal(ids, d['ids'])
        all_tokens = d['tokens']; positions = d['positions']
    with np.load(cache / (model+'_vectors.npz')) as d:
        support = spectral_support(all_tokens[train].mean(axis=1), d['source_decoder'], d['source_projection'])
        support['atom_ids'] = d['source_atoms'].tolist()
    enc = adapter_for(model)
    selector = last_window_slice(enc, positions); layer = MODELS[model][-1]
    tokens = all_tokens[~train]
    delta = np.zeros((len(x), tokens.shape[-1]), dtype='f4')
    # Original arm subtracts the zero-dose forecast in the original item-order batches.
    clean = change_forecasts(enc, layer, x, tokens, delta, selector, batch)
    layouts = {}
    for source in MODELS:
        with np.load(cache / (source+'_vectors.npz')) as d:
            np.testing.assert_array_equal(ids, d['ids'])
            active = d['source_codes'][~train] > 0; features = d['source_atoms']
        row, atom = np.nonzero(active)
        # Preserve the literal row order, duplicates, and final partial batch from the real arm.
        replay = change_forecasts(enc, layer, x[row], tokens[row], delta[row], selector, batch)
        key = 'source_native' if model == source else 'target_dense'
        with np.load(cache / (source+'_'+key+'.npz')) as d:
            np.testing.assert_array_equal(ids[~train], d['ids'])
            np.testing.assert_array_equal(features, d['feature_ids'])
            observed = d['effect']
        layouts[source] = dict(scored_arm=key, atom_ids=features.tolist(),
            populations=numerical_summary(replay-clean[row], observed, row, atom, split,
                                          ('cal','daily','electricity'), len(features)))
    return dict(forecaster=model, batch=batch, geometry=support, layouts=layouts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=['bolt','timesfm'])
    p.add_argument('--cache', type=Path, default=Path('/tmp/tsfm_atoms_20261120'))
    args = p.parse_args()
    import torch
    torch.set_num_threads(3)
    out = audit(args.cache, args.model)
    (HERE / ('atom_audit_'+args.model+'.json')).write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
