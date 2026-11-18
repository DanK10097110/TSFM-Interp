"""Bounded discovery screen of native trend/seasonality composition.

Known-period input OLS is an exact control for this controlled synthetic task.
Native mixed forecast loss and actual component-sum gains are paired by seed.
All12signed model comparisons use one fixed Holm family; no uniqueness claim.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import time

import numpy as np

from tsfm_lens.analysis.stats import mean_ci
from tsfm_lens.experiments.feature_search.statistics import compare_effects

PROTOCOL_PATH = Path(__file__).with_name('trend_seasonality_protocol.json')


def load_protocol():
    """Read the fixed two-sided discovery-screen protocol."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def make_corpus(protocol=None):
    """Generate nine matched trend/phase/component cells per period and seed."""
    protocol = load_protocol() if protocol is None else protocol
    horizon, context = protocol['horizon'], protocol['context']
    t = np.arange(-context, horizon)
    contexts, futures, metadata = [], [], []
    for seed in range(protocol['n_bundles']):
        rng = np.random.default_rng(protocol['seed_base'] + seed)
        amplitude = rng.uniform(.5, 2.)
        phase = rng.uniform(0., 2 * np.pi)
        offset = rng.uniform(-2., 2.)
        slope_magnitude = amplitude * rng.uniform(.5, 2.) / horizon
        variance = amplitude**2 / 2 + slope_magnitude**2 * (horizon**2 - 1) / 12
        for period in protocol['periods']:
            cells = [('mixed', sign, cell) for sign in (-1, 1) for cell in (0, 1)]
            cells += [('pure_trend', sign, 0) for sign in (-1, 1)]
            cells += [('pure_season', 0, cell) for cell in (0, 1)]
            cells += [('background_only', 0, 0)]
            for condition, sign, cell in cells:
                angle = phase + cell * protocol['phase_shift']
                slope = sign * slope_magnitude
                values = np.full(len(t), offset)
                if condition in ('mixed', 'pure_trend'):
                    values += slope * t
                if condition in ('mixed', 'pure_season'):
                    values += amplitude * np.sin(2 * np.pi * t / period + angle)
                contexts.append(values[:context])
                futures.append(values[context:])
                metadata.append(dict(seed=seed, generator_seed=protocol['seed_base'] + seed,
                    period=period, condition=condition, trend_sign=sign, phase_cell=cell,
                    amplitude=amplitude, phase=angle, offset=offset, slope=slope,
                    slope_magnitude=slope_magnitude, variance_scale=variance))
    return np.array(contexts, dtype=np.float32), np.array(futures), metadata


def design_matrix(length, period, start, horizon):
    """Use explicit physical time and retain the full trend/harmonic Gram."""
    if length < 4 or period <= 2 or horizon <= 0:
        raise ValueError('Need identifiable trend/harmonic design and positive horizon')
    t = np.arange(start, start + length)
    design = np.column_stack((t / horizon, np.ones(length),
                              np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)))
    if np.linalg.matrix_rank(design) != 4 or np.linalg.cond(design) > 1e6:
        raise ValueError('Trend/harmonic design is rank-deficient or poorly conditioned')
    return design


def fit_components(values, period, *, start, horizon):
    """Fit slope, intercept and a seasonal sine/cosine pair by joint OLS."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Need finite two-dimensional series')
    design = design_matrix(values.shape[1], period, start, horizon)
    coefficients = np.linalg.lstsq(design, values.T, rcond=None)[0].T
    return dict(coefficients=coefficients, residual=values - coefficients @ design.T,
                gram=design.T @ design / len(design), condition_number=float(np.linalg.cond(design)))


def input_forecast_baseline(contexts, period, horizon):
    """Extrapolate a known-period harmonic-plus-linear input fit without internals."""
    fit = fit_components(contexts, period, start=-contexts.shape[1], horizon=horizon)
    future_design = design_matrix(horizon, period, 0, horizon)
    return fit['coefficients'] @ future_design.T


def error_decomposition(prediction, truth, period, horizon):
    """Verify exact full-Gram-plus-residual reconstruction without orthogonal claims."""
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and truth must have identical shapes')
    p = fit_components(prediction, period, start=0, horizon=horizon)
    t = fit_components(truth, period, start=0, horizon=horizon)
    delta = p['coefficients'] - t['coefficients']
    projected = np.einsum('ni,ij,nj->n', delta, p['gram'], delta)
    diagonal = np.sum(delta**2 * np.diag(p['gram']), axis=1)
    residual = np.mean((p['residual'] - t['residual'])**2, axis=1)
    direct = np.mean((prediction - truth)**2, axis=1)
    if not np.allclose(projected + residual, direct, atol=1e-9, rtol=1e-9):
        raise ValueError('Full trend/harmonic Gram failed exact MSE reconstruction')
    return dict(total_mse=direct, reconstructed_mse=projected + residual,
                gram_cross_terms=projected - diagonal, gram=p['gram'],
                prediction_coefficients=p['coefficients'], condition_number=p['condition_number'])


def matched_rows(metadata, period):
    """Match complete nine-cell component blocks without assuming artifact order."""
    lookup = {}
    expected = {('mixed', sign, cell) for sign in (-1, 1) for cell in (0, 1)}
    expected |= {('pure_trend', sign, 0) for sign in (-1, 1)}
    expected |= {('pure_season', 0, cell) for cell in (0, 1)} | {('background_only', 0, 0)}
    for i, row in enumerate(metadata):
        if row['period'] != period:
            continue
        key = (row['seed'], row['condition'], row['trend_sign'], row['phase_cell'])
        if key in lookup:
            raise ValueError('Duplicate trend/seasonality factorial cell')
        lookup[key] = i
    seeds = np.unique([key[0] for key in lookup])
    if len(seeds) < 3:
        raise ValueError('Need at least three generating bundles')
    mixed, trend, seasonal, background, groups = [], [], [], [], []
    for seed in seeds:
        cells = {(key[1], key[2], key[3]) for key in lookup if key[0] == seed}
        if cells != expected:
            raise ValueError('Every seed/period needs all nine factorial cells')
        for sign in (-1, 1):
            for cell in (0, 1):
                mixed.append(lookup[seed, 'mixed', sign, cell])
                trend.append(lookup[seed, 'pure_trend', sign, 0])
                seasonal.append(lookup[seed, 'pure_season', 0, cell])
                background.append(lookup[seed, 'background_only', 0, 0])
                groups.append(seed)
    return dict(seeds=seeds, mixed=np.array(mixed), trend=np.array(trend),
                seasonal=np.array(seasonal), background=np.array(background), groups=np.array(groups))


def validate_corpus(context, truth, metadata, protocol=None):
    """Validate common scales, physical components and the exact input control."""
    protocol = load_protocol() if protocol is None else protocol
    horizon = protocol['horizon']
    if context.shape != (len(metadata), protocol['context']) or truth.shape != (len(metadata), horizon):
        raise ValueError('Corpus arrays and metadata must align')
    maximum_input_mse = 0.
    for period in protocol['periods']:
        rows = matched_rows(metadata, period)
        selected = np.array([i for i, row in enumerate(metadata) if row['period'] == period])
        coefficients = fit_components(truth[selected], period, start=0, horizon=horizon)['coefficients']
        for coefficient, i in zip(coefficients, selected):
            row = metadata[i]
            variance = row['amplitude']**2 / 2 + row['slope_magnitude']**2 * (horizon**2 - 1) / 12
            if not np.isfinite(variance) or variance <= 0 or not np.isclose(row['variance_scale'], variance):
                raise ValueError('Scale must be the declared sum of component future variances')
            amplitude = row['amplitude'] if row['condition'] in ('mixed', 'pure_season') else 0.
            slope = row['slope'] if row['condition'] in ('mixed', 'pure_trend') else 0.
            expected = [slope * horizon, row['offset'], amplitude * np.cos(row['phase']), amplitude * np.sin(row['phase'])]
            if not np.allclose(coefficient, expected, atol=1e-9):
                raise ValueError('Future truth does not match its physical trend/seasonal metadata')
        baseline = input_forecast_baseline(context[selected], period, horizon)
        maximum_input_mse = max(maximum_input_mse, float(np.max(np.mean((baseline - truth[selected])**2, axis=1))))
        m, t, s, b = (rows[key] for key in ('mixed', 'trend', 'seasonal', 'background'))
        if not np.allclose(context[m], context[t] + context[s] - context[b], atol=3e-6):
            raise ValueError('Context component decomposition is not additive')
        if not np.allclose(truth[m], truth[t] + truth[s] - truth[b], atol=1e-10):
            raise ValueError('True future component decomposition is not additive')
    for seed in np.unique([row['seed'] for row in metadata]):
        bundle = [row for row in metadata if row['seed'] == seed]
        for field in ('amplitude', 'slope_magnitude', 'offset', 'variance_scale'):
            if not np.allclose([row[field] for row in bundle], bundle[0][field], atol=1e-12):
                raise ValueError('Both periods and all cells must share bundle nuisance values')
    if maximum_input_mse > 1e-10:
        raise ValueError('Known-period harmonic-plus-linear input baseline failed exact forecast control')
    return dict(max_input_ols_forecast_mse_raw=maximum_input_mse,
                baseline='Known-period harmonic-plus-linear context OLS; no activations or learned model',
                variance_units='Sum of component future variances, not realized mixture variance')


def component_sum(prediction, rows):
    """Combine actual native pure-component and constant-background forecasts."""
    return prediction[rows['trend']] + prediction[rows['seasonal']] - prediction[rows['background']]


def analyze(predictions, truth, metadata, protocol=None):
    """Apply the complete two-sided Holm12 discovery family with fixed signs."""
    protocol = load_protocol() if protocol is None else protocol
    settings = protocol['inference']
    truth = np.asarray(truth, dtype=float)
    if set(predictions) != set(protocol['models']):
        raise ValueError('All three prespecified models are required')
    mean_settings = dict(n_boot=settings['n_boot'], seed=settings['seed'], unit='generating_seed_bundle')
    claims, descriptive = {}, {}
    common_seeds = None
    for period in protocol['periods']:
        rows = matched_rows(metadata, period)
        seeds = rows['seeds']
        if common_seeds is None:
            common_seeds = seeds
        elif not np.array_equal(common_seeds, seeds):
            raise ValueError('Both periods must share generating bundles')
        def average(values):
            return np.array([np.mean(np.asarray(values)[rows['groups'] == seed]) for seed in seeds])
        losses, gains, model_entries = {}, {}, {}
        period_rows = np.array([i for i, row in enumerate(metadata) if row['period'] == period])
        for model in protocol['models']:
            prediction = np.asarray(predictions[model], dtype=float)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise ValueError('Stop: material nonfinite or misaligned native predictions')
            mixed = rows['mixed']
            variance = np.array([metadata[i]['variance_scale'] for i in mixed])
            combined = component_sum(prediction, rows)
            native_raw = np.mean((prediction[mixed] - truth[mixed])**2, axis=1)
            combined_raw = np.mean((combined - truth[mixed])**2, axis=1)
            losses[model] = average(native_raw / variance)
            gains[model] = average((native_raw - combined_raw) / variance)
            decomposition = error_decomposition(prediction[period_rows], truth[period_rows], period, protocol['horizon'])
            coefficients = decomposition['prediction_coefficients']
            predicted_slope = coefficients[:, 0] / protocol['horizon']
            predicted_amplitude = np.linalg.norm(coefficients[:, 2:4], axis=1)
            predicted_phase = np.arctan2(coefficients[:, 3], coefficients[:, 2])
            pure_losses, behavior = {}, {}
            for condition in ('mixed', 'pure_trend', 'pure_season', 'background_only'):
                local = np.array([metadata[i]['condition'] == condition for i in period_rows])
                indices = period_rows[local]
                condition_seeds = np.array([metadata[i]['seed'] for i in indices])
                scaled_error = np.mean((prediction[indices] - truth[indices])**2, axis=1) / np.array([metadata[i]['variance_scale'] for i in indices])
                grouped_error = np.array([scaled_error[condition_seeds == seed].mean() for seed in seeds])
                pure_losses[condition] = mean_ci(grouped_error, **mean_settings)
                true_slope = np.array([metadata[i]['slope'] for i in indices])
                nominal = np.array([metadata[i]['slope_magnitude'] for i in indices])
                target_phase = np.array([metadata[i]['phase'] for i in indices])
                amplitude = np.array([metadata[i]['amplitude'] for i in indices])
                present = condition in ('mixed', 'pure_season')
                valid = predicted_amplitude[local] > 1e-6 * amplitude if present else np.zeros(len(indices), dtype=bool)
                behavior[condition] = dict(
                    absolute_slope_error_over_nominal=float(np.mean(np.abs(predicted_slope[local] - true_slope) / nominal)),
                    seasonal_amplitude_over_nominal=float(np.mean(predicted_amplitude[local] / amplitude)),
                    seasonal_phase_error_radians=None if not valid.any() else float(np.mean(np.abs(np.angle(np.exp(1j * (predicted_phase[local][valid] - target_phase[valid])))))),
                    phase_unidentifiable_rows=int(np.sum(~valid)) if present else 0,
                    seasonal_component_present=present)
            model_entries[model] = dict(
                native_mixed_mse_scaled=mean_ci(losses[model], **mean_settings),
                component_sum_mse_scaled=mean_ci(average(combined_raw / variance), **mean_settings),
                component_sum_gain_scaled=mean_ci(gains[model], **mean_settings),
                native_mixed_mse_raw=float(native_raw.mean()), component_sum_mse_raw=float(combined_raw.mean()),
                condition_losses_scaled=pure_losses, behavior=behavior, gram=decomposition['gram'].tolist(),
                gram_reconstruction_max_abs=float(np.max(np.abs(decomposition['total_mse'] - decomposition['reconstructed_mse']))),
                gram_cross_terms_raw_mse=float(decomposition['gram_cross_terms'].mean()),
                design_condition_number=decomposition['condition_number'])
        for earlier, later in itertools.combinations(protocol['models'], 2):
            claims[f'native_gap_{later}_minus_{earlier}_period{period}'] = (losses[later], losses[earlier])
            claims[f'component_sum_gain_{later}_minus_{earlier}_period{period}'] = (gains[later], gains[earlier])
        descriptive[str(period)] = model_entries
    if len(claims) != 12:
        raise ValueError('The fixed Holm12 family must remain complete')
    results = compare_effects(claims, common_seeds, n_boot=settings['n_boot'],
        n_permutations=settings['n_permutations'], seed=settings['seed'], ci=settings['ci'])
    for value in results.values():
        excludes_zero = value['lo'] > 0 or value['hi'] < 0
        value.update(unit='MSE divided by common sum of component future variances',
            direction='positive' if value['value'] > 0 else 'negative' if value['value'] < 0 else 'zero',
            minimum_absolute_practical=protocol['minimum_absolute_practical'],
            passed=bool(abs(value['value']) >= protocol['minimum_absolute_practical'] and excludes_zero and value['p_holm'] < .05))
    return dict(primary=results, family_size=12, n_seeds=len(common_seeds), n_contexts=len(metadata),
        descriptive=descriptive, discovery_only=True,
        contrast_sign='Later-listed minus earlier-listed in model order timesfm,bolt,chronos2',
        units='Scaled MSE uses a²/2+slope_magnitude²*(H²−1)/12 across ALL cells/periods; not realized mixture variance',
        caution='Discovery screen needs fresh replication of promising effects; exact input OLS is a control,not an activation result;no unique/missing-feature claim')


def native_predict(adapter, contexts, horizon, batch=8):
    """Use native median forecasts at the requested48-step horizon without hooks."""
    import torch
    outputs = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for start in range(0, len(contexts), batch):
            prediction = adapter.predict(contexts[start:start + batch], horizon, [.5])['point']
            prediction = np.asarray(prediction)
            if prediction.shape != (len(contexts[start:start + batch]), horizon) or not np.isfinite(prediction).all():
                raise RuntimeError('Stop: native output shape or finiteness gate failed')
            outputs.append(prediction)
    return np.concatenate(outputs)


def run(out):
    """Run a single bounded offline worker with exact repeats and source hashes."""
    import torch
    from tsfm_lens.config import ModelConfig, DataConfig
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    protocol = load_protocol()
    out.mkdir(parents=True, exist_ok=False)
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    (out / 'protocol.json').write_bytes(protocol_bytes)
    contexts, truth, metadata = make_corpus(protocol)
    validation = validate_corpus(contexts, truth, metadata, protocol)
    metadata_bytes = (json.dumps(metadata, sort_keys=True, indent=2) + '\n').encode('utf-8')
    (out / 'metadata.json').write_bytes(metadata_bytes)
    (out / 'input_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'corpus.npz', context=contexts, clean_future=truth)
    provenance = dict(protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), models={})
    predictions = {}
    classes = dict(timesfm=TimesFMAdapter, bolt=ChronosBoltAdapter, chronos2=Chronos2Adapter)
    started = time.perf_counter()
    print(json.dumps(dict(total_forecasts=3456, estimated_seconds=[30, 120], runtime_cap=600)), flush=True)
    for model in protocol['models']:
        if time.perf_counter() - started > 600:
            (out / 'paused.json').write_text(json.dumps(dict(reason='Ten-minute cap before next model', model=model)), encoding='utf-8')
            return
        block = time.perf_counter()
        adapter = classes[model](ModelConfig(name=model, checkpoint=protocol['checkpoints'][model]),
            DataConfig(context_len=protocol['context'], horizon=protocol['horizon']), torch.device('cuda:0'), torch.float32)
        adapter.ensure_loaded()
        prediction = native_predict(adapter, contexts, protocol['horizon'])
        repeat = native_predict(adapter, contexts[:8], protocol['horizon'])
        repeat_error = float(np.max(np.abs(repeat - prediction[:8])))
        if repeat_error != 0:
            raise RuntimeError('Native same-batch repeat gate failed')
        permutation_error = None
        if model == 'chronos2':
            permuted = native_predict(adapter, contexts[:8][::-1].copy(), protocol['horizon'])[::-1]
            permutation_error = float(np.max(np.abs(permuted - repeat)))
            if not np.allclose(permuted, repeat, atol=1e-6, rtol=1e-6):
                raise RuntimeError('Chronos2 independent-group batch permutation gate failed')
        predictions[model] = prediction
        np.savez_compressed(out / f'{model}_predictions.npz', prediction=prediction)
        source = Path(inspect.getsourcefile(classes[model]))
        provenance['models'][model] = dict(checkpoint=protocol['checkpoints'][model],
            adapter_source_path=str(source), adapter_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            repeat_max_abs=repeat_error, batch_permutation_max_abs=permutation_error,
            seconds=time.perf_counter() - block, horizon=protocol['horizon'], dtype='float32', batch=8)
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(json.dumps(dict(model=model, completed=len(contexts), seconds=time.perf_counter() - block,
            repeat_max_abs=repeat_error)), flush=True)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    result = analyze(predictions, truth, metadata, protocol)
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(primary=result['primary']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1., 6e9 / torch.cuda.get_device_properties(0).total_memory))
    run(args.out)
