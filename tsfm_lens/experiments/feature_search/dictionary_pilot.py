"""Prepare dictionary_pilot4951000; run ONE bounded stage only after a slot grant.

Use ``prepare`` now. ``run --slot-granted ...`` is a coordinator action after
allocation. Each method/k/seed/cut is a separate <=600s, <=3GiB CPU process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from experiments.feature_search import dictionary_core as core

SEEDS = (4951000, 4951001, 4951002)
METHODS = ('native', 'rotation', 'pca', 'topk', 'nested')


def read(path):
    """Read one UTF-8 record."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Atomic finite JSON; stages are immutable but knowledge gets a final status."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    """Stream source/artifact provenance without copying activation archives."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def knowledge(out, question, results, controls, limitations, next_step, status):
    """Persist a self-contained record after every stage, including failures."""
    write(out/'knowledge.json', dict(question=question, exact_results=results,
          controls=controls, limitations=limitations, next_step=next_step, status=status))


def prepare(out, source):
    """Inventory saved states and freeze all fitting/evaluation choices, no training."""
    rows = read(source/'rows.json')
    fit, test = core.split_indices(rows)
    inventory = {}
    for model in ('bolt', 'timesfm'):
        with np.load(source/f'{model}_states.npz') as states:
            inventory[model] = dict(full_shape=list(states['0_full'].shape),
                                    cuts=read(source/f'{model}_worker.json')['cuts'],
                                    archive_entries=len(states.files))
    files = ['rows.json', 'inputs.npz', 'protocol.json'] + [f'{m}_{s}' for m in inventory
            for s in ('states.npz', 'worker.json')]
    local = Path(__file__).parent
    from tsfm_lens.sae import models
    from experiments.feature_search import selective_study, real_background_poc, composition_restoration, coherence_study, homogeneous_normalizer
    code = list(local.glob('dictionary_*.py')) + [Path(m.__file__) for m in
            (models, selective_study, real_background_poc, composition_restoration, coherence_study, homogeneous_normalizer)]
    protocol = dict(source=str(source.resolve()), hashes={f: sha(source/f) for f in files},
        code_hashes={str(p.resolve()): sha(p) for p in code}, inventory=inventory,
        seeds=SEEDS, methods=METHODS, k=[8, 16], width_multiplier=1, maximum_width_multiplier=2,
        steps=500, batch=64, lr=.001, seconds=600, rss_bytes=3*2**30, threads=1,
        first_cut='0', fit_rows=fit.tolist(), test_rows=test.tolist(),
        native_subset='four TEST sources per domain, one first-recorded case per source; 20 receivers, including four of16 synthetic TEST sources; never selected by scores',
        execution_plan='both models, cut0; k8/16; topk/nested/rotation three seeds; deterministic native/PCA once per k; fit then native preservation per artifact; source-heldout diagnostics remain exploratory',
        split_counts={s: {d: len({r['source'] for r in rows if r['split']==s and r['domain']==d})
                          for d in sorted({r['domain'] for r in rows})} for s in ('fit', 'test')},
        fitting='FIT full-token states only; equal source mass then equal case/token mass; vector mean and one RMS scalar',
        comparison='same SAE parameters, initialization, batches and steps; nested has two reconstruction terms (not equal FLOPs); direct fits have zero optimizer steps',
        activity='actual nonzero counts measured FIT/TEST; each SAE also gets native/PCA/rotation controls at its exact per-token L0, without TEST tuning',
        probes='FIT correlation-screened ridge alpha .01, budgets1/4; mean/RMS separately; normalized and physical coefficients; input, normalizer metadata-only and same-metadata-plus-feature controls',
        normalization='Bolt full-history location/scale; TimesFM cumulative32-step-patch locations/scales; analytic FP64 metadata from history, no native inference; both per-history-SD and model-normalized-input spectral targets; metadata partialling FIT only',
        semantics='fixed .1 locked-normalizer sinusoid vs noise central responses; FIT selects features; TEST evaluates background/phase slices; phase of intervention itself is fixed, so no crossed-phase claim',
        causal_gate='required fresh singleton native calls at same cut: exact repeat/self patch/saved replay; nonzero reach; reconstruction and selected-feature ablation vs Gram-matched random controls',
        usefulness='NOT ESTABLISHED until native preservation and perturbation-specific effects pass; no automatic monosemanticity verdict',
        limitations=['spectral cache only; four independent TEST real sources/domain',
                     'residual coordinates are not post-activation FFN expansion neurons',
                     'random initialization can retain decodable tones (PM35); trained-vs-init causal control missing',
                     'Matryoshka-inspired global TopK two-prefix loss; no full paper reproduction',
                     'TSAE main controlled comparison PatchTST/ETT; univariate variable routing inapplicable; no TSAE implemented',
                     'BatchTopK deferred; if added, threshold must be FIT-frozen and batch-independent',
                     'all claims exploratory; no TEST-best selection or token-level confidence intervals'],
        literature=['https://arxiv.org/html/2605.05151v1', 'https://arxiv.org/html/2503.17547v1',
                    'https://arxiv.org/html/2412.06410v1', 'https://arxiv.org/html/2610.04925v1'])
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', protocol)
    knowledge(out, 'Can compact features outperform signed native and rotated coordinates on held-out spectral diagnostics?',
              dict(inventory=inventory, split_counts=protocol['split_counts'], training_launched=False, native_launched=False),
              ['source split audited; frozen provenance; no targets read'], protocol['limitations'],
              'Coordinator grants one stage slot; first cut0, width=d; all three seeds and k8/16 required before comparison.', 'ready_not_run')


def verify(out, model):
    """Refuse changed source/code/checkpoint metadata before loading any states."""
    p = read(out/'protocol.json')
    source = Path(p['source'])
    for name, expected in p['hashes'].items():
        if name.startswith(('bolt_', 'timesfm_')) and not name.startswith(model+'_'):
            continue
        if sha(source/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    for path, expected in p['code_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen implementation changed: '+path)
    return p, source


def load_states(source, model, cut, rows):
    """Read only full-token states at one declared block-output cut."""
    cuts = read(source/f'{model}_worker.json')['cuts']
    c = cuts.index(cut)
    with np.load(source/f'{model}_states.npz') as archive:
        values = np.stack([archive[f'{i}_full'][c] for i in range(len(rows))])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite states')
    return values


def normalizer_metadata(history, model):
    """Analytic metadata for finite native inputs, without checkpoint loading.

    TimesFM uses cumulative patch statistics. These FP64 analytical values
    approximate native FP32 arithmetic; they are not a native hook measurement.
    """
    history = np.asarray(history, float)
    if model == 'bolt':
        mu = history.mean(1, keepdims=True)
        sd = history.std(1, keepdims=True)
        sd = np.where(sd == 0, 1e-5, sd)
        return np.column_stack([mu, sd]), (history-mu)/sd
    if model != 'timesfm' or history.shape[1] % 32:
        raise ValueError('Expected native TimesFM 32-step patches')
    stops = range(32, history.shape[1]+1, 32)
    mu = np.column_stack([history[:, :end].mean(1) for end in stops])
    sd = np.column_stack([history[:, :end].std(1) for end in stops])
    transformed = (history.reshape(len(history), -1, 32)-mu[..., None])/np.where(sd[..., None] < 1e-6, 1., sd[..., None])
    return np.column_stack([mu, sd]), transformed.reshape(history.shape)


def concept_targets(history, model):
    """Fixed history-only spectral descriptors plus nuisance controls, not case IDs."""
    t = np.arange(-history.shape[1], 0)
    columns = [np.ones(len(t)), t/len(t)]
    names = ['level', 'trend']
    for f in (5.35, 6.35, 17.35):
        columns.extend([np.sin(2*np.pi*f*t/128), np.cos(2*np.pi*f*t/128)])
        names.extend([f'sin_{f}', f'cos_{f}'])
    coefficients = history @ np.linalg.pinv(np.stack(columns, axis=1)).T
    metadata, normalized_input = normalizer_metadata(history, model)
    normalized_coefficients = normalized_input @ np.linalg.pinv(np.stack(columns, axis=1)).T
    sd = history.std(1).clip(1e-8)
    targets = np.column_stack([coefficients[:, 2:]/sd[:, None], coefficients[:, 2:], normalized_coefficients[:, 2:], history.mean(1), sd,
                               coefficients[:, 1]/sd])
    target_names = ([n+'_history_sd' for n in names[2:]]+[n+'_physical' for n in names[2:]]+
                    [n+'_model_normalized' for n in names[2:]]+['level_nuisance', 'scale_nuisance', 'trend_nuisance'])
    generic = np.column_stack([history.mean(1), sd, np.std(np.diff(history), axis=1),
                              history[:, -1], np.mean(history[:, 1:]*history[:, :-1], axis=1)])
    phase = np.arctan2(coefficients[:, 3], coefficients[:, 2])
    return targets, target_names, generic, phase, metadata


def probe_metrics(codes, history, rows, fit, test, model):
    """Single/few-feature selectivity with input controls and phase/domain slices."""
    y, names, generic, phase, metadata = concept_targets(history.astype(float), model)
    weights = core.source_weights([rows[i]['source'] for i in fit])
    test_rows = [rows[i] for i in test]
    views = {'code_mean': codes.mean(1), 'code_rms': np.sqrt(np.mean(codes**2, axis=1)),
             'input_generic': generic, 'input_spectral_oracle': y[:, :18], 'metadata_only': metadata}
    result = {}
    for view, features in views.items():
        result[view] = []
        for target, name in enumerate(names):
            ym = weights @ y[fit, target]
            variance = float(weights @ (y[fit, target]-ym)**2)
            if variance < 1e-12:
                result[view].append(dict(target=name, skipped='FIT constant target'))
                continue
            for budget in (1, 4):
                pred, selected = core.sparse_probe(features[fit], y[fit, target], weights, features[test], budget)
                error = (pred-y[test, target])**2/variance
                bins = np.minimum(3, ((phase[test]+np.pi)/(2*np.pi)*4).astype(int))
                result[view].append(dict(target=name, budget=budget, selected=selected,
                    test=core.source_summary(error, test_rows),
                    phase_slices={str(b): core.source_summary(error[bins==b], [r for r, take in zip(test_rows, bins==b) if take])
                                  for b in range(4) if np.any(bins==b)}))
                if view in ('code_mean', 'code_rms', 'input_generic'):
                    conditional, metadata_pred, ids = core.conditional_probe(features[fit], y[fit, target], weights,
                        features[test], budget, metadata[fit], metadata[test])
                    result[view][-1]['metadata_conditioned'] = dict(selected=ids, metadata_dimensions=metadata.shape[1],
                        test=core.source_summary((conditional-y[test, target])**2/variance, test_rows),
                        metadata_only_test=core.source_summary((metadata_pred-y[test, target])**2/variance, test_rows))
    return result


def perturbation_metrics(source, model, cut, artifact, rows, fit, test):
    """FIT-only feature selection by sinusoid/noise contrast; report TEST nuisance leakage."""
    c = read(source/f'{model}_worker.json')['cuts'].index(cut)
    responses = {}
    with np.load(source/f'{model}_states.npz') as archive:
        for direction in ('sinusoid', 'noise'):
            responses[direction] = []
            for i in range(len(rows)):
                h = archive[f'{i}_{direction}_0.1_locked'][c]
                z = core.encode((h-artifact['center'])/float(artifact['scale']), artifact)
                responses[direction].append(np.mean(((z[0]-z[1])/.2)**2, axis=0))
            responses[direction] = np.asarray(responses[direction])
    w = core.source_weights([rows[i]['source'] for i in fit])
    signal = w @ responses['sinusoid'][fit]
    nuisance = w @ responses['noise'][fit]
    score = signal/(signal+nuisance+1e-12)
    eligible = signal > max(float(signal.max())*.01, 1e-12)
    order = np.argsort(-np.where(eligible, score, -1), kind='stable')
    result = {}
    for budget in (1, 4):
        selected = order[eligible[order]][:budget]
        a = responses['sinusoid'][test][:, selected].sum(1)
        b = responses['noise'][test][:, selected].sum(1)
        result[str(budget)] = dict(selected=selected.tolist(),
            signal_energy=core.source_summary(a, [rows[i] for i in test]),
            nuisance_energy=core.source_summary(b, [rows[i] for i in test]),
            signal_share=core.source_summary(a/(a+b+1e-12), [rows[i] for i in test]))
    return result


def stage_name(args):
    """Keep each bounded work item immutable and independently resumable."""
    return f'{args.stage}_{args.model}_cut{args.cut}_{args.method}_k{args.k}_s{args.seed}'


def fit_stage(args, dest):
    """Train one dictionary and evaluate once; never optimize on TEST states."""
    p, source = verify(args.out, args.model)
    rows = read(source/'rows.json')
    fit, test = core.split_indices(rows)
    h = load_states(source, args.model, args.cut, rows)
    nt, d = h.shape[1:]
    w = core.source_weights(np.repeat([rows[i]['source'] for i in fit], nt))
    center, scale = core.fit_scaler(h[fit].reshape(-1, d), w)
    x = (h-center)/scale
    train = x[fit].reshape(-1, d)
    artifact = dict(center=center, scale=np.array(scale), k=np.array(args.k))
    if args.method in ('topk', 'nested'):
        arrays, budget = core.train_sae(train, w, args.method, args.seed, args.k, d,
                                        steps=p['steps'], batch=p['batch'])
        artifact.update(arrays)
        budget['expected_sampled_tokens_per_source'] = budget['sampled_tokens']/len({rows[i]['source'] for i in fit})
    else:
        artifact['basis'] = core.fit_basis(train, w, args.method, args.seed)
        budget = dict(steps=0, trainable_parameters=0, stored_basis_parameters=d*d,
                      fit_token_count=len(train), fit_passes='PCA covariance/eigensolve or data-independent basis')
    z = core.encode(x, artifact)
    rec = core.reconstruct(z, artifact)
    metrics = {}
    for name, indices in [('fit', fit), ('test', test)]:
        ws = core.source_weights(np.repeat([rows[i]['source'] for i in indices], nt))
        metrics[name] = core.health(x[indices].reshape(-1, d), z[indices].reshape(-1, z.shape[-1]),
                                   rec[indices].reshape(-1, d), ws)
        row_error = np.mean((rec[indices]-x[indices])**2, axis=(1, 2))
        metrics[name]['source_mse_scaled'] = core.source_summary(row_error, [rows[i] for i in indices])
    history = np.load(source/'inputs.npz')['history']
    probes = probe_metrics(z, history, rows, fit, test, args.model)
    perturbations = perturbation_metrics(source, args.model, args.cut, artifact, rows, fit, test)
    matched = {}
    if args.method in ('topk', 'nested'):
        counts = np.count_nonzero(z, axis=-1)
        for method in ('native', 'rotation', 'pca'):
            control = dict(k=np.array(args.k), basis=core.fit_basis(train, w, method, args.seed))
            cz = core.encode(x, control, counts=counts)
            cr = core.reconstruct(cz, control)
            matched[method] = dict(requested_l0_exact=True,
                actual_l0_equal=bool(np.array_equal(np.count_nonzero(cz, axis=-1), counts)),
                test=core.health(x[test].reshape(-1, d), cz[test].reshape(-1, d), cr[test].reshape(-1, d),
                                 core.source_weights(np.repeat([rows[i]['source'] for i in test], nt))),
                probes=probe_metrics(cz, history, rows, fit, test, args.model))
    np.savez_compressed(dest/'dictionary.npz', **artifact)
    np.savez_compressed(dest/'fit_activity.npz', active=np.any(z[fit] != 0, axis=(0, 1)))
    result = dict(model=args.model, cut=args.cut, method=args.method, k=args.k, seed=args.seed,
                  width=z.shape[-1], budget=budget, metrics=metrics, probes=probes,
                  perturbations=perturbations, activity_matched_controls=matched,
                  native_forecast_preservation='NOT RUN: usefulness not established',
                  protocol_sha256=sha(args.out/'protocol.json'), dictionary_sha256=sha(dest/'dictionary.npz'))
    write(dest/'results.json', result)
    knowledge(dest, 'Does this bounded dictionary preserve and selectively expose spectral information?',
              dict(results_file='results.json', sha256=sha(dest/'results.json'), budget=budget,
                   test={key: metrics['test'][key] for key in ('nmse', 'mean_l0', 'dead_fraction')},
                   native_forecast_preservation='not_run'),
              ['FIT-only scaling/selection; source weights; input/metadata controls; nuisance response; activity matched bases'],
              p['limitations'], 'Finish three seeds; compare stability; request native patch slot.', 'descriptive_complete_native_pending')


def compare(out, model, cut, k, dest=None):
    """Summarize all three completed seeds; never choose a TEST winner."""
    result = {}
    for method in METHODS:
        seeds = (SEEDS[0],) if method in ('native', 'pca') else SEEDS
        paths = [out/f'fit_{model}_cut{cut}_{method}_k{k}_s{s}' for s in seeds]
        if any(not (path/'results.json').exists() for path in paths):
            result[method] = dict(status='incomplete_three_seed_requirement')
            continue
        if len(paths) == 1:
            result[method] = dict(status='deterministic_basis_single_fit', seed_variation='not_applicable')
            continue
        artifacts = [dict(np.load(path/'dictionary.npz')) for path in paths]
        active = [np.load(path/'fit_activity.npz')['active'] for path in paths]
        result[method] = {f'{i}_{j}': core.stability(core.decoder(artifacts[i]), core.decoder(artifacts[j]), active[i], active[j])
                          for i, j in ((0, 1), (0, 2), (1, 2))}
    if dest is None:
        dest = out/f'compare_{model}_cut{cut}_k{k}'
        dest.mkdir(exist_ok=False)
    knowledge(dest, 'Are active feature directions stable across the three dictionary seeds?', result,
              ['one-to-one matching; dead atoms excluded; deterministic native/PCA ceilings; random rotation floor'],
              ['direction stability does not establish semantic or causal stability'],
              'Native preservation and crossed-phase targeted interventions remain required.', 'comparison_only')


def rss_bytes(lines):
    """A just-exited/zombie /proc status legitimately lacks VmRSS."""
    return int(next((line.split()[1] for line in lines if line.startswith('VmRSS:')), '0'))*1024


def launch(args):
    """External monitor enforces caps even during SVD, imports or model loading."""
    if not args.slot_granted:
        raise ValueError('No slot granted: prepare only; coordinator must authorize run')
    dest = args.out/stage_name(args)
    dest.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               DICTIONARY_WORKER_GRANTED='1')
    command = [sys.executable, '-m', 'experiments.feature_search.dictionary_pilot', '_worker',
               '--out', str(args.out), '--stage', args.stage, '--model', args.model, '--cut', args.cut,
               '--method', args.method, '--k', str(args.k), '--seed', str(args.seed)]
    start, peak, reason = time.monotonic(), 0, None
    with (dest/'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                parent = Path(f'/proc/{os.getpid()}/status').read_text().splitlines()
                total = rss_bytes(lines)+rss_bytes(parent)
                peak = max(peak, total)
            except (FileNotFoundError, StopIteration):
                pass
            if peak > 3*2**30 or time.monotonic()-start > 600:
                reason = 'resource cap exceeded'
                os.killpg(process.pid, 9)
                process.wait()
                break
            time.sleep(.1)
    execution = dict(returncode=process.returncode, reason=reason, peak_rss=peak, seconds=time.monotonic()-start)
    write(dest/'execution.json', execution)
    if process.returncode != 0 or not (dest/'knowledge.json').exists():
        knowledge(dest, 'Bounded dictionary stage', execution, ['external 1-thread/3GiB/600s envelope'],
                  ['stage failed; no scientific result'], 'Inspect worker.log; no automatic retry.', 'failed')
        raise RuntimeError(execution)


def main():
    """Explicit prepare/run boundary; direct worker invocation is refused."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', '_worker', 'compare'))
    p.add_argument('--out', type=Path, default=Path('runs/dictionary_pilot4951000'))
    p.add_argument('--source', type=Path, default=Path('runs/selective_replication4942000'))
    p.add_argument('--slot-granted', action='store_true')
    p.add_argument('--stage', choices=('fit', 'native', 'ffn', 'compare'), default='fit')
    p.add_argument('--model', choices=('bolt', 'timesfm'), default='bolt')
    p.add_argument('--method', choices=METHODS, default='native')
    p.add_argument('--cut', default='0')
    p.add_argument('--k', type=int, choices=(8, 16), default=8)
    p.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.out, a.source)
    elif a.command == 'run':
        launch(a)
    elif a.command == 'compare':
        a.stage = 'compare'
        launch(a)
    else:
        if os.environ.get('DICTIONARY_WORKER_GRANTED') != '1':
            p.error('Use monitored run after a slot grant')
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        dest = a.out/stage_name(a)
        if a.stage == 'fit':
            fit_stage(a, dest)
        elif a.stage == 'compare':
            compare(a.out, a.model, a.cut, a.k, dest=dest)
        else:
            from experiments.feature_search.dictionary_native import native_stage, ffn_stage
            (native_stage if a.stage == 'native' else ffn_stage)(a, dest)


if __name__ == '__main__':
    main()
