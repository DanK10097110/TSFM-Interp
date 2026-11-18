"""Bounded, source-disjoint real-background pilots for ROADMAP 49.23 IDs 2–6.

All gates are exploratory effect-size screens. Real data stays outside sealed
benchmark corpora. CPU workers run serially; cached weights and Arrow files only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

DATASETS = ('monash_electricity_hourly', 'm4_hourly', 'monash_pedestrian_counts')
MODELS = {'bolt': 'amazon/chronos-bolt-small', 'timesfm': 'google/timesfm-2.5-200m-pytorch'}
TRANSFORMS = ((.1, 0.), (10., 0.), (1., -5.), (1., 5.))
LIMIT_BYTES = 3 * 2**30


def write(path, value):
    """Persist finite, human-readable records."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    """Read UTF-8 JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    """Stream artifact hashes."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signal(length, period=24, amplitude=.5, phase=.3):
    """Physical origin t=0 is always the first forecast step."""
    t = np.arange(-length, 32)
    return amplitude * np.sin(2 * np.pi * t / period + phase)


def mse(value, target):
    """One independent score per source, averaging dependent horizon positions."""
    return np.mean((np.asarray(value, dtype=float) - target)**2, axis=-1)


def response_metrics(response, expected, period=24):
    """Amplitude, phase, off-frequency energy and waveform error; no FFT-bin alias."""
    t = np.arange(32)
    basis = np.stack([np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period)], axis=1)
    coefficient = np.asarray(response) @ np.linalg.pinv(basis).T
    reference = np.asarray(expected) @ np.linalg.pinv(basis).T
    projection = coefficient @ basis.T
    energy = float(np.mean(np.asarray(expected)**2))
    return dict(error=mse(response, expected)/energy,
                amplitude=np.linalg.norm(coefficient, axis=-1)/np.linalg.norm(reference),
                phase=np.angle(np.exp(1j*(np.arctan2(coefficient[..., 1], coefficient[..., 0])-
                                          np.arctan2(reference[1], reference[0])))),
                off_target=mse(response, projection)/energy)


def descriptors(x):
    """Past-only fixed baseline features, including known-frequency detectability."""
    x = np.asarray(x, dtype=float)
    t = np.arange(x.shape[1])
    basis = np.stack([np.ones(len(t)), t/len(t), np.sin(2*np.pi*t/24),
                      np.cos(2*np.pi*t/24), np.sin(2*np.pi*t/40), np.cos(2*np.pi*t/40)], axis=1)
    coefficients = x @ np.linalg.pinv(basis).T
    return np.column_stack([coefficients, x.std(1), np.std(np.diff(x), axis=1),
                            np.mean(x[:, 24:]*x[:, :-24], axis=1),
                            np.max(abs(x), axis=1), np.mean(x**3, axis=1)])


def baseline(x, kind):
    """Classical alternatives use exactly the same observed history."""
    n = x.shape[1]
    t = np.arange(-n, 32)
    if kind == 'seasonal':
        return x[:, -24:][:, np.arange(32) % 24]
    basis = np.stack([np.ones(len(t)), t/n, np.sin(2*np.pi*t/24), np.cos(2*np.pi*t/24),
                      np.sin(2*np.pi*t/40), np.cos(2*np.pi*t/40)], axis=1)
    return (x @ np.linalg.pinv(basis[:n]).T) @ basis[n:].T


def dataset_means(values, metadata, indices):
    """Do not pretend dependent series in one dataset are independent datasets."""
    values = np.asarray(values)
    return {name: float(np.mean(values[[i for i in indices if metadata[i]['dataset'] == name]]))
            for name in DATASETS}


def screen(values, metadata, indices, threshold):
    """Require the effect-size gate in EVERY represented dataset; no p-values."""
    means = dataset_means(values, metadata, indices)
    return dict(dataset_means=means, threshold=threshold,
                passed=bool(all(np.isfinite(v) and v >= threshold for v in means.values())))


def history_control(x):
    """Permute old 32-hour patches; retain the final old patch and recent128 exactly."""
    if x.shape[1] != 256:
        raise ValueError('History control requires 256 hours')
    blocks = x[:, :128].reshape(len(x), 4, 32)
    result = x.copy()
    result[:, :128] = blocks[:, [1, 0, 2, 3]].reshape(len(x), 128)
    return result


def prepare(out, cache, per_dataset=8, offset=0):
    """Select by source IDs and past-only eligibility, then freeze all rows/gates."""
    import pyarrow as pa
    out.mkdir(parents=True, exist_ok=False)
    series, metadata, excluded = [], [], []
    for name in DATASETS:
        paths = sorted((cache/'datasets/autogluon___chronos_datasets'/name).rglob('*train.arrow'))
        if len(paths) != 1:
            raise ValueError(f'Expected one cached Arrow source for {name}: {paths}')
        with pa.memory_map(str(paths[0]), 'r') as source:
            table = pa.ipc.open_stream(source).read_all()
        order = np.random.default_rng(492306).permutation(len(table))
        eligible = []
        for row in order:
            record = table.slice(int(row), 1).to_pylist()[0]
            values = np.asarray(record['target'], dtype=float)
            if len(values) < 288:
                continue
            window = values[-288:]
            history = window[:256]
            scale = float(np.std(history[-128:]))
            if not np.isfinite(history).all() or scale <= 1e-8:
                excluded.append(dict(dataset=name, id=record['id'], reason='past nonfinite/constant'))
                continue
            stamps = np.asarray(record['timestamp'][-288:], dtype='datetime64[ms]')
            if not np.all(np.diff(stamps) == np.timedelta64(1, 'h')):
                excluded.append(dict(dataset=name, id=record['id'], reason='not hourly'))
                continue
            eligible.append((record, window, scale))
            if len(eligible) >= offset+per_dataset:
                break
        if len(eligible) < offset+per_dataset:
            raise ValueError(f'Insufficient sources in {name}')
        for j, (record, window, scale) in enumerate(eligible[offset:offset+per_dataset]):
            if not np.isfinite(window).all():
                raise ValueError('Selected future is nonfinite; do not silently select another source')
            center = float(np.mean(window[128:256]))
            series.append((window-center)/scale)
            metadata.append(dict(dataset=name, id=record['id'], row_rank=offset+j,
                                 split='fit' if j < per_dataset//2 else 'evaluation',
                                 center=center, scale=scale, sampling_hours=1,
                                 source_file=str(paths[0]), window_sha256=hashlib.sha256(window.tobytes()).hexdigest()))
    np.savez_compressed(out/'corpus.npz', series=np.asarray(series, dtype=np.float32))
    write(out/'metadata.json', metadata)
    checkpoints = {}
    for model, repo in MODELS.items():
        snapshots = sorted((cache/'hub'/('models--'+repo.replace('/', '--'))/'snapshots').iterdir())
        if len(snapshots) != 1:
            raise ValueError(f'Choose a single cached checkpoint explicitly: {repo}')
        weights = sorted(snapshots[0].glob('*.safetensors'))
        if not weights:
            raise ValueError('Missing cached safetensors')
        checkpoints[model] = dict(repo=repo, path=str(snapshots[0]), revision=snapshots[0].name,
                                  hashes={p.name: sha(p) for p in weights})
    write(out/'protocol.json', dict(
        version=1, experiments=[2, 3, 4, 5, 6], parked=[1, 7, 8], seed=492306,
        per_dataset=per_dataset, offset=offset, history=128, long_history=256, horizon=32,
        amplitude='0.5 times past128 source SD; relative-amplitude only; sources past-standardized',
        components=dict(a=dict(period=24, phase=.3), b=dict(period=40, phase=-.7)),
        polarities=[-1, 1], transforms=TRANSFORMS, site='last transformer block output before final normalization/head; all tokens including Bolt REG',
        fit='first half of each dataset, source IDs never overlap evaluation; no future fitting except ID2 supervised target on fit rows',
        scope='Exploratory effect-size screens, no p-values or model ranking; dependent datasets retained separately',
        gates={'2': 'heldout squared-error prediction >=20% better than input ridge, input RBF, and training mean in EVERY dataset; mean harmful interaction >=.05',
               '3': 'natural response error<=.5, oracle patch closure<=.05, transfer improves no-edit>=.1 and geometry random>=.05 in every dataset; all random geometry valid',
               '4': 'worst fixed transform forecast/response RMS>=.1 in every dataset; classical RMS<1e-5',
               '5': 'short and irrelevant response losses exceed informative long by>=.1 in every dataset; all prefix step-energy ratios<=2',
               '6': 'all three heldout SAE recon and delta NMSE<=.25; causal response error<=.1; aligned support overlap<=.5; pairwise decoded forecast disagreement<=.1'},
        sae=dict(seeds=[491, 492, 493], dict_size=64, k=8, steps=150, lr=.003, batch=64),
        budget=dict(cpu_threads=1, model_seconds=900, resident_bytes=LIMIT_BYTES, gpu=False, model_batch=4),
        source_sha256=sha(__file__), corpus_sha256=sha(out/'corpus.npz'), metadata_sha256=sha(out/'metadata.json'),
        checkpoints=checkpoints, exclusions=excluded))


class Native:
    """Two deterministic native paths, keeping model lifetimes in separate workers."""
    def __init__(self, name, checkpoint):
        import torch
        torch.set_num_threads(1)
        self.name = name
        if name == 'timesfm':
            from experiments.feature_search.homogeneous_normalizer import load_model
            self.model = load_model(next(Path(checkpoint).glob('*.safetensors')))
            self.layer = self.model.stacked_xf[-1]
            self.layer_name = f'stacked_xf.{len(self.model.stacked_xf)-1}'
            self.patch = 32
        else:
            from chronos import BaseChronosPipeline
            self.pipeline = BaseChronosPipeline.from_pretrained(checkpoint, device_map='cpu', torch_dtype=torch.float32)
            self.model = self.pipeline.model.eval()
            self.layer = self.model.encoder.block[-1]
            self.layer_name = f'encoder.block.{len(self.model.encoder.block)-1}'
            self.patch = int(self.model.config.chronos_config['input_patch_size'])

    def predict(self, x, replacement=None):
        """Capture raw states and replace exactly the declared interface when requested."""
        import torch
        from tsfm_lens.extraction.hooks import _primary, _rebuild
        predictions, states = [], []
        for start in range(0, len(x), 4):
            batch = torch.as_tensor(np.ascontiguousarray(x[start:start+4]), dtype=torch.float32)
            captured = []
            def hook(module, args, output):
                hidden = _primary(output)
                captured.append(hidden.detach().cpu().numpy().copy())
                if replacement is not None:
                    value = torch.as_tensor(replacement[start:start+4], dtype=hidden.dtype)
                    if value.shape != hidden.shape:
                        raise ValueError('Patch geometry mismatch')
                    return _rebuild(output, value)
            handle = self.layer.register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    if self.name == 'timesfm':
                        value, _, extra = self.model.decode(32, batch, torch.zeros_like(batch, dtype=torch.bool))
                        if extra is not None and extra.numel():
                            raise ValueError('Unexpected autoregressive rollout')
                        pred = value[:, -1, :32, self.model.aridx].numpy()
                    else:
                        _, mean = self.pipeline.predict_quantiles(batch, prediction_length=32, quantile_levels=[.5])
                        pred = mean.numpy()
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError('Expected exactly one hook firing per forecast')
            predictions.append(pred)
            states.append(captured[0])
        return np.concatenate(predictions), np.concatenate(states)


def supervised_screen(x, interaction, harm, train, test, metadata):
    """Fit fixed ridge/RBF input baselines and nested internal feature ridge on discovery."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge
    feature = descriptors(x)
    extra = np.column_stack([feature, interaction])
    predictors = {
        'input_ridge': (feature, make_pipeline(StandardScaler(), Ridge(alpha=10.))),
        'input_rbf': (feature, make_pipeline(StandardScaler(), KernelRidge(alpha=1., kernel='rbf', gamma=1/feature.shape[1]))),
        'internal_ridge': (extra, make_pipeline(StandardScaler(), Ridge(alpha=10.)))}
    errors, predictions = {}, {}
    for key, (matrix, estimator) in predictors.items():
        estimator.fit(matrix[train], harm[train])
        predictions[key] = estimator.predict(matrix)
        errors[key] = (predictions[key]-harm)**2
    predictions['training_mean'] = np.full(len(x), harm[train].mean())
    errors['training_mean'] = (predictions['training_mean']-harm)**2
    comparisons = {}
    for key in ['input_ridge', 'input_rbf', 'training_mean']:
        ratios = {}
        for dataset in DATASETS:
            ix = [i for i in test if metadata[i]['dataset'] == dataset]
            denominator = float(errors[key][ix].mean())
            ratios[dataset] = float(1-errors['internal_ridge'][ix].mean()/max(denominator, 1e-12))
        comparisons[key] = ratios
    passed = all(v >= .2 for row in comparisons.values() for v in row.values()) and np.mean(harm[test]) >= .05
    return dict(passed=bool(passed), improvements=comparisons, harmful_interaction=harm.tolist(),
                predictions={k: v.tolist() for k, v in predictions.items()},
                dataset_harm=dataset_means(harm, metadata, test),
                scope='Heldout pilot prediction of harmful interaction; not the closed previous failure-prediction task')


def donor_indices(x, train, test):
    """Choose similar/dissimilar sources from discovery using past-only descriptors."""
    d = descriptors(x)
    scale = d[train].std(0)
    z = (d-d[train].mean(0))/np.maximum(scale, 1e-6)
    distance = np.mean((z[test, None]-z[train][None])**2, axis=-1)
    return train[np.argmin(distance, axis=1)], train[np.argmax(distance, axis=1)]


def sae_audit(native, x, h0, h1, f0, f1, train, test, settings, arrays):
    """Matched refits use discovery activations only; function is evaluated by patches."""
    import torch
    from scipy.optimize import linear_sum_assignment
    from tsfm_lens.sae.models import TopKSAE
    fit = np.concatenate([h0[train], h1[train]], axis=0).reshape(-1, h0.shape[-1])
    center = fit.mean(0)
    scale = float(np.sqrt(np.mean((fit-center)**2)))
    tx = torch.tensor((fit-center)/scale, dtype=torch.float32)
    e0 = torch.tensor((h0[test]-center)/scale, dtype=torch.float32)
    e1 = torch.tensor((h1[test]-center)/scale, dtype=torch.float32)
    results, supports, decoders, effects = [], [], [], []
    for seed in settings['seeds']:
        generator = torch.Generator().manual_seed(seed)
        sae = TopKSAE(h0.shape[-1], settings['dict_size'], settings['k'], generator=generator)
        optimizer = torch.optim.Adam(sae.parameters(), lr=settings['lr'])
        for step in range(settings['steps']):
            ix = torch.randint(len(tx), (settings['batch'],), generator=generator)
            reconstruction, _ = sae(tx[ix])
            loss = torch.mean((reconstruction-tx[ix])**2)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); sae.normalize_decoder_()
        with torch.no_grad():
            r0, z0 = sae(e0); r1, z1 = sae(e1)
            delta = ((r1-r0)*scale).numpy()
            recon_error = float(((r0-e0)**2).mean()/e0.square().mean())
            delta_error = float(((r1-r0-(e1-e0))**2).mean()/(e1-e0).square().mean())
            decoder = sae.W_dec.numpy().copy()
            support = ((z1-z0).abs().numpy() > 1e-6)
            perm = np.random.default_rng(seed).permutation(settings['dict_size'])
            zp = z1.numpy()[..., perm]
            permutation_error = float(np.max(abs(zp@decoder[perm]+sae.b_dec.numpy()-r1.numpy())))
            duplicate_error = float(np.max(abs(np.concatenate([z1.numpy()/2, z1.numpy()/2], axis=-1)@
                                                 np.concatenate([decoder, decoder], axis=0)+sae.b_dec.numpy()-r1.numpy())))
        pred, _ = native.predict(x[test], h0[test]+delta)
        effects.append(pred-f0[test])
        function_error = float(np.mean(mse(pred, f1[test]))/np.mean((f1[test]-f0[test])**2))
        arrays[f'sae_{seed}_forecast'] = pred
        arrays[f'sae_{seed}_delta'] = delta
        arrays[f'sae_{seed}_decoder'] = decoder
        arrays[f'sae_{seed}_encoder'] = sae.W_enc.detach().numpy()
        arrays[f'sae_{seed}_b_enc'] = sae.b_enc.detach().numpy()
        arrays[f'sae_{seed}_b_dec'] = sae.b_dec.detach().numpy()
        results.append(dict(seed=seed, heldout_reconstruction_nmse=recon_error, heldout_delta_nmse=delta_error,
                            forecast_response_nmse=function_error, permutation_max_error=permutation_error,
                            duplicate_max_error=duplicate_error))
        supports.append(support); decoders.append(decoder)
    pairs = []
    for i in range(3):
        for j in range(i+1, 3):
            a, b = linear_sum_assignment(-(decoders[i]@decoders[j].T))
            first, second = supports[i][..., a], supports[j][..., b]
            overlap = float(np.logical_and(first, second).sum()/max(1, np.logical_or(first, second).sum()))
            disagreement = float(np.mean((effects[i]-effects[j])**2)/np.mean((f1[test]-f0[test])**2))
            pairs.append(dict(seeds=[settings['seeds'][i], settings['seeds'][j]], aligned_support_jaccard=overlap,
                              decoded_forecast_disagreement=disagreement))
    arrays['sae_fit_center'] = center
    arrays['sae_fit_scale'] = np.asarray(scale)
    valid = all(r['heldout_reconstruction_nmse'] <= .25 and r['heldout_delta_nmse'] <= .25 and
                r['forecast_response_nmse'] <= .1 and max(r['permutation_max_error'], r['duplicate_max_error']) < 1e-4 for r in results)
    passed = valid and all(r['aligned_support_jaccard'] <= .5 and r['decoded_forecast_disagreement'] <= .1 for r in pairs)
    return dict(passed=bool(passed), fidelity_valid=bool(valid), refits=results, pairs=pairs,
                scope='Tiny SAE measurement screen, no claim of model instability when dictionaries fail fidelity')


def worker(out, model):
    """Run all five frozen small pilots on one checkpoint, never widening the search."""
    started = time.monotonic()
    p = read(out/'protocol.json')
    if p['source_sha256'] != sha(__file__) or p['corpus_sha256'] != sha(out/'corpus.npz') or p['metadata_sha256'] != sha(out/'metadata.json'):
        raise ValueError('Frozen source or data changed')
    checkpoint = p['checkpoints'][model]
    if any(sha(Path(checkpoint['path'])/name) != value for name, value in checkpoint['hashes'].items()):
        raise ValueError('Checkpoint hash changed')
    metadata = read(out/'metadata.json')
    data = np.load(out/'corpus.npz')['series']
    x, y = data[:, 128:256], data[:, 256:]
    train = np.array([i for i, r in enumerate(metadata) if r['split'] == 'fit'])
    test = np.array([i for i, r in enumerate(metadata) if r['split'] == 'evaluation'])
    native = Native(model, checkpoint['path'])
    arrays, conditions, hidden = {}, {}, {}
    a, b = signal(128), signal(128, 40, .5, -.7)
    for name, value in [('base', x), ('plus', x+a[:128]), ('minus', x-a[:128]),
                        ('bplus', x+b[:128]), ('bminus', x-b[:128]),
                        ('bothplus', x+a[:128]+b[:128]), ('bothminus', x-a[:128]-b[:128])]:
        conditions[name], hidden[name] = native.predict(value)
        arrays['forecast_'+name] = conditions[name]
        arrays['state_'+name] = hidden[name]
    f, h = conditions['base'], hidden['base']
    repeat, _ = native.predict(x)
    identity, _ = native.predict(x, h)
    gates = dict(repeat_max_error=float(np.max(abs(repeat-f))), self_patch_max_error=float(np.max(abs(identity-f))))
    if max(gates.values()) != 0:
        raise ValueError(f'Identity controls fail: {gates}')
    internal, harmful = [], []
    for suffix, sign in [('plus', 1), ('minus', -1)]:
        interaction = conditions['both'+suffix]-conditions[suffix]-conditions['b'+suffix]+f
        hi = hidden['both'+suffix]-hidden[suffix]-hidden['b'+suffix]+h
        internal.append(np.log1p(np.mean(hi.astype(float)**2, axis=(1, 2))/np.maximum(np.mean(h.astype(float)**2, axis=(1, 2)), 1e-12)))
        additive = conditions[suffix]+conditions['b'+suffix]-f
        truth = y+sign*(a[128:]+b[128:])
        harmful.append(mse(conditions['both'+suffix], truth)-mse(additive, truth))
        arrays['interaction_'+suffix] = interaction
    e2 = supervised_screen(x, np.stack(internal, axis=1), np.mean(harmful, axis=0), train, test, metadata)
    print(model, 'interaction complete', flush=True)
    from experiments.feature_search.composition_restoration import random_edits
    near, far = donor_indices(x, train, test)
    transfer = {}
    for suffix, sign in [('plus', 1), ('minus', -1)]:
        delta = hidden[suffix]-h
        oracle, _ = native.predict(x[test], h[test]+delta[test])
        target = conditions[suffix][test]
        denominator = float(np.mean(a[128:]**2))
        oracle_error = mse(oracle, target)/denominator
        natural_error = response_metrics(target-f[test], sign*a[128:])['error']
        draws, invalid, diagnostics = random_edits(h[test], delta[near], np.random.default_rng(492303+sign), 2, True)
        random_pred = [native.predict(x[test], h[test]+draw)[0] for draw in draws]
        near_pred, _ = native.predict(x[test], h[test]+delta[near])
        far_pred, _ = native.predict(x[test], h[test]+delta[far])
        ratio = np.sqrt(np.sum(delta[test].astype(float)**2, axis=(1, 2))/np.maximum(np.sum(delta[near].astype(float)**2, axis=(1, 2)), 1e-12))
        matched, _ = native.predict(x[test], h[test]+delta[near]*ratio[:, None, None])
        loss = mse(near_pred, target)/denominator
        random_loss = np.mean([mse(value, target)/denominator for value in random_pred], axis=0)
        unedited = mse(f[test], target)/denominator
        checks = dict(natural=bool(np.all(natural_error <= .5)), oracle=bool(np.all(oracle_error <= .05)), geometry=not bool(invalid.any()))
        testmeta = [metadata[i] for i in test]
        improvement = screen(unedited-loss, testmeta, np.arange(len(test)), .1)
        specificity = screen(random_loss-loss, testmeta, np.arange(len(test)), .05)
        transfer[suffix] = dict(checks=checks, improvement=improvement, specificity=specificity,
                               oracle_error=oracle_error.tolist(), natural_error=natural_error.tolist(),
                               near_loss=loss.tolist(), far_loss=(mse(far_pred, target)/denominator).tolist(),
                               magnitude_matched_loss=(mse(matched, target)/denominator).tolist(),
                               random_geometry=diagnostics, invalid_geometry=invalid.tolist())
        for key, val in [('oracle', oracle), ('near', near_pred), ('far', far_pred), ('magnitude', matched), ('random0', random_pred[0]), ('random1', random_pred[1])]:
            arrays[f'transfer_{suffix}_{key}'] = val
    e3 = dict(passed=all(all(row['checks'].values()) and row['improvement']['passed'] and row['specificity']['passed'] for row in transfer.values()),
              polarities=transfer, near_donor_indices=near.tolist(), far_donor_indices=far.tolist(),
              scope='Raw normalized-state response transfer; oracle closure measures native normalization mismatch; random controls preserve Gram/radial geometry, not manifold validity')
    print(model, 'transfer complete', flush=True)
    transform_rows, rms = [], []
    for index, (scale, shift) in enumerate(TRANSFORMS):
        fp, hp = native.predict(scale*x+shift)
        fa, ha = native.predict(scale*(x+a[:128])+shift)
        fb, hb = native.predict(scale*(x-a[:128])+shift)
        fback = (fp-shift)/scale
        errors = [np.sqrt(mse(fback, f)), np.sqrt(mse((fa-fp)/scale, conditions['plus']-f)),
                  np.sqrt(mse((fb-fp)/scale, conditions['minus']-f))]
        rms.append(np.max(errors, axis=0))
        transform_rows.append(dict(scale=scale, shift=shift, mean_forecast_rms=float(errors[0][test].mean()),
                                   mean_plus_response_rms=float(errors[1][test].mean()), mean_minus_response_rms=float(errors[2][test].mean()),
                                   hidden_plus_relative_change=float(np.linalg.norm((ha-hp)[test]-(hidden['plus']-h)[test])/np.linalg.norm((hidden['plus']-h)[test]))))
        arrays[f'transform_{index}'] = np.stack([fp, fa, fb])
    classical_error = max(float(np.max(abs((baseline(scale*x+shift, kind)-shift)/scale-baseline(x, kind))))
                          for scale, shift in TRANSFORMS for kind in ['fourier', 'seasonal'])
    e4 = screen(np.max(rms, axis=0), metadata, test, .1)
    e4.update(transforms=transform_rows, classical_max_error=classical_error)
    e4['passed'] = e4['passed'] and classical_error < 1e-5
    long_x = data[:, :256]
    al = signal(256)
    longbase, _ = native.predict(long_x)
    control_x = history_control(long_x)
    controlbase, _ = native.predict(control_x)
    history_errors, history_checks = [], []
    for suffix, sign in [('plus', 1), ('minus', -1)]:
        longplus, _ = native.predict(long_x+sign*al[:256])
        controlled = history_control(long_x+sign*al[:256])
        controlplus, _ = native.predict(controlled)
        short_error = response_metrics(conditions[suffix]-f, sign*a[128:])['error']
        long_error = response_metrics(longplus-longbase, sign*a[128:])['error']
        control_error = response_metrics(controlplus-controlbase, sign*a[128:])['error']
        step_ratio = np.mean(np.diff(controlled[:, :129], axis=1)**2, axis=1)/np.maximum(np.mean(np.diff((long_x+sign*al[:256])[:, :129], axis=1)**2, axis=1), 1e-12)
        history_errors.append((short_error, long_error, control_error))
        history_checks.append(step_ratio)
        arrays[f'history_{suffix}'] = np.stack([longplus, controlplus])
    short_error, long_error, control_error = np.mean(history_errors, axis=0)
    e5 = dict(short_advantage=screen(short_error-long_error, metadata, test, .1),
              irrelevant_advantage=screen(control_error-long_error, metadata, test, .1),
              boundary_valid=bool(np.all(np.asarray(history_checks)[:, test] <= 2)),
              prefix_step_energy_ratios=np.asarray(history_checks).tolist(),
              errors=dict(short=short_error.tolist(), long=long_error.tolist(), irrelevant=control_error.tolist()),
              geometry=dict(history=[128, 256], patch=native.patch, physical_sampling_hours=1,
                            tokens=[128//native.patch, 256//native.patch], bolt_extra_reg=model == 'bolt'),
              scope='Old-patch permutation retains distribution and junction/recent128; short and long have identical future and physical component phase')
    e5['passed'] = e5['short_advantage']['passed'] and e5['irrelevant_advantage']['passed'] and e5['boundary_valid']
    print(model, 'scaling/history complete', flush=True)
    e6 = sae_audit(native, x, h, hidden['plus'], f, conditions['plus'], train, test, p['sae'], arrays)
    classical = {}
    for kind in ['fourier', 'seasonal']:
        clean = baseline(x, kind)
        response = baseline(x+a[:128], kind)-clean
        classical[kind] = dict(clean_mse=dataset_means(mse(clean, y), metadata, test),
                               response=dataset_means(response_metrics(response, a[128:])['error'], metadata, test))
    zero = np.zeros_like(x)
    pure, _ = native.predict(np.broadcast_to(a[:128], x.shape).copy())
    zero_forecast, _ = native.predict(zero)
    arrays['pure_component'] = pure
    arrays['zero_forecast'] = zero_forecast
    result = dict(model=model, checkpoint=checkpoint, site=native.layer_name, state_shape=list(h.shape),
                  identity=gates, experiments={'2': e2, '3': e3, '4': e4, '5': e5, '6': e6},
                  classical=classical, clean_mse=dataset_means(mse(f, y), metadata, test),
                  separate_forecast_sum_response_error=dataset_means(response_metrics(pure-zero_forecast, a[128:])['error'], metadata, test),
                  component_response={suffix: {k: v.tolist() for k, v in response_metrics(conditions[suffix]-f, sign*a[128:]).items()}
                                      for suffix, sign in [('plus', 1), ('minus', -1)]},
                  elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  evidence='Exploratory behavioral and within-model intervention screens; no independent dataset-population inference',
                  installed_sources={type(native.model).__module__: sha(sys.modules[type(native.model).__module__].__file__)})
    np.savez_compressed(out/f'{model}_arrays.npz', **arrays)
    result['arrays_sha256'] = sha(out/f'{model}_arrays.npz')
    write(out/f'{model}.json', result)
    print(model, 'finished', result['elapsed_seconds'], flush=True)


def run(out):
    """Monitor serial workers; failed execution cannot nominate expansion."""
    p = read(out/'protocol.json')
    execution = {}
    for model in MODELS:
        if (out/f'{model}.json').exists():
            raise ValueError('Refuse to overwrite a completed model result')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        started = time.monotonic()
        max_rss = 0
        failure = None
        with (out/f'{model}.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.real_background_poc',
                                      'worker', '--out', str(out), '--model', model], env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:
                    status = Path(f'/proc/{child.pid}/status').read_text(encoding='utf-8')
                    rss = int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))*1024
                    max_rss = max(max_rss, rss)
                except (FileNotFoundError, StopIteration):
                    pass
                if max_rss > p['budget']['resident_bytes'] or time.monotonic()-started > p['budget']['model_seconds']:
                    failure = 'resource limit exceeded'
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    break
                time.sleep(.2)
        execution[model] = dict(returncode=child.returncode, peak_observed_rss=max_rss,
                                elapsed_seconds=time.monotonic()-started, failure=failure,
                                valid=child.returncode == 0 and failure is None)
        write(out/'execution.json', execution)
    summary(out)


def summary(out):
    """Record all five outcomes and paired source contrasts without significance claims."""
    execution = read(out/'execution.json')
    models = {m: read(out/f'{m}.json') for m in MODELS if execution[m]['valid']}
    decisions = {}
    for experiment in ['2', '3', '4', '5', '6']:
        passes = [m for m, r in models.items() if r['experiments'][experiment]['passed']]
        decisions[experiment] = dict(qualifying_models=passes,
                                    decision='eligible for fresh-source replication' if passes else 'stop this screen',
                                    all_models_completed=len(models) == len(MODELS))
    contrast = {}
    if len(models) == 2:
        meta = read(out/'metadata.json')
        test = [i for i, r in enumerate(meta) if r['split'] == 'evaluation']
        for sign in ['plus', 'minus']:
            difference = np.asarray(models['timesfm']['component_response'][sign]['error'])-np.asarray(models['bolt']['component_response'][sign]['error'])
            contrast[sign] = dataset_means(difference, meta, test)
    write(out/'summary.json', dict(decisions=decisions, execution=execution,
          paired_timesfm_minus_bolt_response_error=contrast,
          inference='No p-values. Full family 5 experiments x2 models retained; dataset means screen allocation only, not confirmation.'))


def main():
    """Prepare freezes data and protocol before any model call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'summary'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--model', choices=list(MODELS))
    parser.add_argument('--per-dataset', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.per_dataset < 4 or args.per_dataset % 2:
            raise ValueError('Need even per-dataset count >=4')
        prepare(args.out, args.cache, args.per_dataset, args.offset)
    elif args.command == 'worker':
        worker(args.out, args.model)
    elif args.command == 'run':
        run(args.out)
    else:
        summary(args.out)


if __name__ == '__main__':
    main()
