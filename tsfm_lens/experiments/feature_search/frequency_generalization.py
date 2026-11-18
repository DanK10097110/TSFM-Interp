"""Stage49 forecast-only spectral generalization screen, separate from tiled CLI.

Joint full-Gram fitting handles non-tiling frequencies. Input perturbations are
behavioral interventions; no hidden states, probes or learned parameters change.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from tsfm_lens.analysis.feature_interference import sign_comparison
from tsfm_lens.analysis.stats import holm

PAIRS = [(32, 16), (31, 16), (33, 16), (24, 12), (48, 24), (40, 20)]
ENDPOINTS = ('normalized_mse', 'off_target_energy', 'target_derivative_error')


def protocol():
    """Return the prospective settings; no outcome-dependent tuning is supported."""
    return dict(seed_base=3200000, n_seeds=16, contexts=[128, 512], pairs=PAIRS,
                horizon=64, delta=.02, amplitudes=[.5, 2.], offset=[-2., 2.],
                models=[dict(name='bolt', adapter='chronos_bolt', checkpoint='amazon/chronos-bolt-small'),
                        dict(name='timesfm', adapter='timesfm', checkpoint='google/timesfm-2.5-200m-pytorch'),
                        dict(name='chronos2', adapter='chronos2', checkpoint='amazon/chronos-2')],
                literature_primary_urls=['https://arxiv.org/abs/2510.19236','https://arxiv.org/abs/2511.15324','https://arxiv.org/abs/2502.06037','https://arxiv.org/abs/2511.05619','https://arxiv.org/abs/2610.28997','https://arxiv.org/abs/2611.02750'],
                novelty_boundary='Prior work covers composition interference, spectral shift and patch/tokenization effects; this is a controlled comparative generalization diagnostic, not a new aliasing mechanism',
                primary_estimand='sign_probability', alpha=.05, n_boot=2000, bootstrap_seed=3200017,
                minimum_absolute_mean=dict(normalized_mse=.05, off_target_energy=.02,
                                           target_derivative_error=.05),
                family_size=108, maximum_basis_condition=100.,
                candidate_gate='same model-pair endpoint passes in at least two distinct period pairs, including one outside the entire31/16,32/16,33/16 neighborhood; fresh replication required',
                scope='forecast-only discovery screen; exact q concerns conditional non-tie majority, not mean superiority; no hidden-feature cause or aliasing diagnosis',
                time_origin='context t=-C..-1, future t=0..H-1; identical future across contexts',
                normalization='native mean/scale recorded, never frozen; empirical context statistics separately labeled',
                resource=dict(cpu_threads=2, batch_size=8, gpu_gib=6, runtime_seconds=600))


def basis(periods, times):
    """Return the full rank joint harmonic design, refusing aliases or instability."""
    if len(periods) != 2 or len(set(periods)) != 2 or min(periods) <= 2:
        raise ValueError('Need distinct periods above Nyquist')
    matrix = np.column_stack([np.ones(len(times))] +
                             [fn(2*np.pi*np.asarray(times)/period)
                              for period in periods for fn in (np.sin, np.cos)])
    condition = float(np.linalg.cond(matrix))
    if np.linalg.matrix_rank(matrix) != 5 or condition > 100:
        raise ValueError('Joint harmonic basis is rank deficient or ill conditioned')
    return matrix, condition


def make_case(config, periods, context):
    """Share latent seed draws across every pair/context and three phase cells."""
    horizon, delta = config['horizon'], config['delta']
    inputs, futures, metadata = [], [], []
    for seed in range(config['n_seeds']):
        rng = np.random.default_rng(config['seed_base'] + seed)
        a, b = rng.uniform(*config['amplitudes'], size=2)
        phi, psi = rng.uniform(-np.pi, np.pi, size=2)
        offset = rng.uniform(*config['offset'])
        for cell, shift in [('baseline', 0.), ('minus', -delta), ('plus', delta)]:
            def signal(t):
                return offset + a*np.sin(2*np.pi*t/periods[0]+phi) + b*np.sin(2*np.pi*t/periods[1]+psi+shift)
            inputs.append(signal(np.arange(-context, 0)))
            futures.append(signal(np.arange(horizon)))
            metadata.append(dict(seed=seed, generator_seed=config['seed_base']+seed,
                                 condition=cell, a=a, b=b, phi=phi, psi=psi,
                                 offset=offset, phase_shift=shift))
    ids = np.asarray([f"{m['generator_seed']}:{periods[0]}:{periods[1]}:{context}:{m['condition']}" for m in metadata])
    return dict(context=np.asarray(inputs, dtype=np.float32), clean_future=np.asarray(futures), row_ids=ids), metadata


def metrics(prediction, truth, metadata, periods, delta):
    """Separate joint-fit cross-response from target fidelity without FFT leakage."""
    pred, truth = np.asarray(prediction), np.asarray(truth)
    if pred.shape != truth.shape or pred.ndim != 2 or not np.isfinite(pred).all() or len(pred) != len(metadata):
        raise ValueError('Finite predictions must match ordered corpus rows and horizon')
    design, condition = basis(periods, np.arange(pred.shape[1]))
    lookup = {(m['seed'], m['condition']): i for i, m in enumerate(metadata)}
    seeds = sorted({m['seed'] for m in metadata})
    if len(lookup) != len(metadata) or set(lookup) != {(s, c) for s in seeds for c in ('baseline', 'minus', 'plus')}:
        raise ValueError('Need one complete three-cell bundle per seed')
    values = {key: [] for key in ENDPOINTS + ('total_derivative_error', 'derivative_offset_energy', 'derivative_residual_energy', 'derivative_cross_energy', 'derivative_energy_closure_error')}
    for seed in seeds:
        i, minus, plus = [lookup[seed, c] for c in ('baseline', 'minus', 'plus')]
        m = metadata[i]
        for index in (i, minus, plus):
            cell = metadata[index]
            if any(cell[k] != m[k] for k in ('a','b','phi','psi','offset','generator_seed')):
                raise ValueError('Matched cells changed latent nuisance parameters')
            t = np.arange(pred.shape[1])
            expected = cell['offset']+cell['a']*np.sin(2*np.pi*t/periods[0]+cell['phi'])+cell['b']*np.sin(2*np.pi*t/periods[1]+cell['psi']+cell['phase_shift'])
            if not np.allclose(truth[index],expected,atol=1e-10,rtol=1e-10):
                raise ValueError('Groundtruth does not match row metadata and physical time')
        response = (pred[plus]-pred[minus])/(2*delta)
        target = m['b']*np.cos(2*np.pi*np.arange(pred.shape[1])/periods[1]+m['psi'])*np.sin(delta)/delta
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        off = design[:, 1:3] @ coefficients[1:3]
        fitted_target = design[:, 3:5] @ coefficients[3:5]
        strength = m['b']**2/2
        values['normalized_mse'].append(np.mean((pred[i]-truth[i])**2)/((m['a']**2+m['b']**2)/2))
        values['off_target_energy'].append(np.mean(off**2)/strength)
        values['target_derivative_error'].append(np.mean((fitted_target-target)**2)/strength)
        values['total_derivative_error'].append(np.mean((response-target)**2)/strength)
        values['derivative_offset_energy'].append(coefficients[0]**2/strength)
        residual = response-design@coefficients
        values['derivative_residual_energy'].append(np.mean(residual**2)/strength)
        components = [off, fitted_target, np.full_like(off, coefficients[0]), residual]
        component_energy = sum(np.mean(v*v) for v in components)
        cross_energy = sum(2*np.mean(a*b) for a,b in combinations(components,2))
        values['derivative_cross_energy'].append(cross_energy/strength)
        values['derivative_energy_closure_error'].append((np.mean(response**2)-component_energy-cross_energy)/strength)
    return {key: np.asarray(value) for key, value in values.items()}, dict(basis_condition=condition,
        gram_condition=float(np.linalg.cond(design.T@design)),
        groundtruth_fit_residual=float(np.max(np.abs(truth-truth@np.linalg.pinv(design).T@design.T))),
        warning='Full Gram retained: separate component energies are descriptive and need not sum to total energy on non-orthogonal windows')


def summarize(values, config):
    """Pointwise approximate bootstrap of means; no assumption-free mean test."""
    rng = np.random.default_rng(config['bootstrap_seed'])
    draws = rng.integers(0, len(values), size=(config['n_boot'], len(values)))
    lo, hi = np.quantile(np.asarray(values)[draws].mean(1), [.025, .975])
    return dict(value=float(np.mean(values)), lo=float(lo), hi=float(hi),
                interval='approximate pointwise whole-seed mean bootstrap')


def analyze(config, cases, predictions):
    """Retain all108 claims, global exact sign correction and fixed size gates."""
    model_names = [m['name'] for m in config['models']]
    claims, descriptive, vectors = {}, {}, {}
    for case_id, (data, metadata, periods) in cases.items():
        model_values = {}
        descriptive[case_id] = {}
        for model in model_names:
            artifact = predictions[case_id][model]
            if not np.array_equal(artifact['row_ids'], data['row_ids']):
                raise ValueError('Forecast row identity mismatch')
            value, diagnostics = metrics(artifact['prediction'], data['clean_future'], metadata, periods, config['delta'])
            model_values[model] = value
            descriptive[case_id][model] = dict(metrics={key: summarize(v, config) for key, v in value.items()}, diagnostics=diagnostics)
            vectors.update({f'{case_id}/{model}/{key}': v for key, v in value.items()})
        for first, second in combinations(model_names, 2):
            for endpoint in ENDPOINTS:
                difference = model_values[second][endpoint]-model_values[first][endpoint]
                key = f'{case_id}/{second}-minus-{first}/{endpoint}'
                effect = sign_comparison(difference)
                effect.update(mean=summarize(difference, config), minimum_absolute_mean=config['minimum_absolute_mean'][endpoint],
                              periods=list(periods), endpoint=endpoint, model_pair=f'{second}-minus-{first}')
                claims[key] = effect
    if len(claims) != config['family_size']:
        raise ValueError('Frozen global case/model-pair/endpoint family changed')
    adjusted = holm({key: effect['p'] for key, effect in claims.items()})
    for key, effect in claims.items():
        direction = 0 if effect['value'] is None else np.sign(effect['value']-.5)
        mean = effect['mean']['value']
        effect.update(p_holm=adjusted[key], passed=bool(adjusted[key] < config['alpha'] and
            abs(mean) >= effect['minimum_absolute_mean'] and np.sign(mean) == direction))
    candidates = candidate_patterns(claims)
    correspondence = {}
    for model in model_names:
        loss = [entry[model]['metrics']['normalized_mse']['value'] for entry in descriptive.values()]
        leak = [entry[model]['metrics']['off_target_energy']['value'] for entry in descriptive.values()]
        rho = np.nan if np.ptp(loss)==0 or np.ptp(leak)==0 else spearmanr(loss, leak).statistic
        correspondence[model] = dict(spearman=None if not np.isfinite(rho) else float(rho),
            scope='descriptive across12 dependent cases; no p-value, prediction or causal claim')
    return dict(protocol=config, claims=claims, descriptive=descriptive, correspondence=correspondence,
                candidates_for_fresh_replication=candidates, primary_family_size=len(claims),
                minimum_exact_sign_p=2**(1-config['n_seeds']),
                minimum_possible_holm_p=min(1.,len(claims)*2**(1-config['n_seeds'])),
                caveat='Statistical evidence and point-mean size gates do not establish expected-loss superiority; no automatic expansion'), vectors


def candidate_patterns(claims):
    """Require two distinct pairs and evidence outside the full32-step neighborhood."""
    candidates = []
    for pair in {v['model_pair'] for v in claims.values()}:
        for endpoint in ENDPOINTS:
            pairs = {tuple(v['periods']) for v in claims.values() if v['model_pair'] == pair and v['endpoint'] == endpoint and v['passed']}
            if len(pairs) >= 2 and any(p not in ((31,16),(32,16),(33,16)) for p in pairs):
                candidates.append(dict(model_pair=pair, endpoint=endpoint, period_pairs=sorted(pairs)))
    return candidates


def digest(path):
    """Bind exact immutable corpus and producer source bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    """Write protocol/corpus hashes before inference into a new directory."""
    config = protocol()
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for p in config['pairs']:
        for context in config['contexts']:
            name = f'p{p[0]}_{p[1]}_c{context}'
            data, metadata = make_case(config, p, context)
            _, diagnostic = metrics(data['clean_future'], data['clean_future'], metadata, p, config['delta'])
            np.savez_compressed(out/f'{name}.npz', **data)
            (out/f'{name}.json').write_text(json.dumps(metadata, indent=2))
            sources[name] = dict(corpus_sha256=digest(out/f'{name}.npz'), metadata_sha256=digest(out/f'{name}.json'), periods=p, diagnostics=diagnostic)
    payload = dict(config=config, sources=sources, driver_sha256=digest(__file__), prepared_before_inference=True)
    (out/'protocol.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    return payload


def load_prepared(prepared):
    """Verify declared source bytes and exact frozen generator before any loading."""
    payload = json.loads((prepared/'protocol.json').read_text())
    config = payload['config']
    if payload['driver_sha256'] != digest(__file__) or json.loads(json.dumps(protocol())) != config:
        raise ValueError('Prepared driver or frozen protocol changed')
    cases = {}
    for name, source in payload['sources'].items():
        if digest(prepared/f'{name}.npz') != source['corpus_sha256'] or digest(prepared/f'{name}.json') != source['metadata_sha256']:
            raise ValueError('Prepared corpus binding changed')
        with np.load(prepared/f'{name}.npz') as data:
            cases[name] = ({k:data[k] for k in data.files}, json.loads((prepared/f'{name}.json').read_text()), source['periods'])
    expected_names = {f'p{p[0]}_{p[1]}_c{c}' for p in config['pairs'] for c in config['contexts']}
    if set(cases) != expected_names:
        raise ValueError('Prepared case membership changed')
    for p in config['pairs']:
        for context in config['contexts']:
            name = f'p{p[0]}_{p[1]}_c{context}'
            expected, metadata = make_case(config,p,context)
            stored, rows, periods = cases[name]
            if rows != metadata or list(periods) != list(p) or any(not np.array_equal(stored[key],value) for key,value in expected.items()):
                raise ValueError('Prepared input differs from frozen physical generator')
    return payload, config, cases


def run(prepared, out, device):
    """Serial offline registry inference only after a separately authorized slot."""
    import torch
    from tsfm_lens.config import ModelConfig, DataConfig
    from tsfm_lens.models import build_adapter
    from experiments.feature_search.normalization_audit import normalization_state
    payload, config, cases = load_prepared(prepared)
    out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes((prepared/'protocol.json').read_bytes())
    torch.set_num_threads(2)
    target = torch.device(device)
    if target.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(min(1.,6*1024**3/torch.cuda.get_device_properties(target).total_memory), target)
    predictions, provenance = {name:{} for name in cases}, {}
    started = time.monotonic()
    for model in config['models']:
        model_started = time.monotonic()
        if target.type=='cuda':torch.cuda.reset_peak_memory_stats(target)
        adapter = build_adapter(ModelConfig(**model), DataConfig(context_len=128,horizon=config['horizon']), target, torch.float32)
        adapter.ensure_loaded()
        provenance[model['name']] = dict(model_config=asdict(ModelConfig(**model)), adapter_source_sha256=digest(__import__('inspect').getfile(type(adapter))), cases={})
        for name, (data, metadata, periods) in cases.items():
            if time.monotonic()-started > config['resource']['runtime_seconds']:
                raise RuntimeError('Ten-minute cap reached; retain partial outputs without analysis')
            adapter.data_cfg = DataConfig(context_len=data['context'].shape[1],horizon=config['horizon'])
            outputs = []
            for start in range(0,len(data['context']),8):
                if time.monotonic()-started > config['resource']['runtime_seconds']:
                    raise RuntimeError('Ten-minute cap within native batch loop')
                with torch.inference_mode():
                    outputs.append(adapter.predict(data['context'][start:start+8],config['horizon'],[.5])['point'])
            pred = np.concatenate(outputs)
            repeat = adapter.predict(data['context'][:8],config['horizon'],[.5])['point']
            permutation = adapter.predict(data['context'][:8][::-1].copy(),config['horizon'],[.5])['point'][::-1]
            if not np.array_equal(repeat,pred[:8]) or not np.allclose(repeat,permutation,atol=1e-6,rtol=1e-6):
                raise RuntimeError('Native repeat/batch-independent permutation control failed')
            if model['name'] == 'chronos2':
                isolated = np.concatenate([adapter.predict(x[None],config['horizon'],[.5])['point'] for x in data['context'][:8]])
                if not np.allclose(isolated,repeat,atol=1e-6,rtol=1e-6):
                    raise RuntimeError('Chronos-2 singleton independence failed')
                _, (mu,sigma) = adapter._inner.instance_norm(adapter.prepare(data['context']))
                mu,sigma = mu.cpu().numpy(),sigma.cpu().numpy()
                head = dict(output_patches=adapter._num_output_patches(config['horizon']), forward_passes=1, autoregressive_rollouts=0, use_arcsinh=bool(adapter._inner.chronos_config.use_arcsinh), normalization_source_sha256=digest(__import__('inspect').getfile(type(adapter._inner.instance_norm))))
            else:
                surface = 'chronos_bolt' if model['name']=='bolt' else 'timesfm_2p5_decode'
                mu,sigma = normalization_state(data['context'],surface,horizon=config['horizon'])
                if model['name']=='bolt':
                    native_steps = int(adapter.pipeline.model_prediction_length)
                else:
                    from timesfm.timesfm_2p5.timesfm_2p5_base import TimesFM_2p5_200M_Definition
                    native_steps = int(TimesFM_2p5_200M_Definition().output_patch_len)
                if config['horizon'] > native_steps:
                    raise RuntimeError('Frozen single native head horizon no longer supported')
                head = dict(native_output_steps=native_steps,forward_passes=1,autoregressive_rollouts=0)
            artifact=dict(prediction=pred,row_ids=data['row_ids'],corpus_sha256=np.asarray(payload['sources'][name]['corpus_sha256']))
            np.savez_compressed(out/f"{model['name']}_{name}.npz",**artifact,native_mu=mu,native_sigma=sigma,
                                empirical_mean=data['context'].mean(1),empirical_std=data['context'].std(1))
            predictions[name][model['name']] = artifact
            provenance[model['name']]['cases'][name] = dict(head=head,repeat_max_abs=float(np.max(abs(repeat-pred[:8]))),
                permutation_max_abs=float(np.max(abs(repeat-permutation))), native_normalization='installed native helper; not frozen',
                forecast_sha256=digest(out/f"{model['name']}_{name}.npz"))
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
            print(json.dumps(dict(model=model['name'],case=name,elapsed=time.monotonic()-started)),flush=True)
        provenance[model['name']]['wall_seconds'] = time.monotonic()-model_started
        if target.type=='cuda':
            provenance[model['name']]['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(target)
            provenance[model['name']]['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(target)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        adapter._release()
        del adapter
        gc.collect()
        if target.type=='cuda':torch.cuda.empty_cache()
    result,vectors=analyze(config,cases,predictions)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(out/'seed_metrics.npz',**vectors)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare_only == bool(args.prepared):parser.error('choose prepare-only or prepared')
    if args.prepare_only:prepare(args.out)
    else:run(args.prepared,args.out,args.device)
