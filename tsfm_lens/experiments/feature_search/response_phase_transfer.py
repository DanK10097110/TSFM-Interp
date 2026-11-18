"""Prospective fresh-source phase transfer with old-only response baselines.

The previous four-phase study supplies all predictive coefficients. Fresh
forecasts test those fixed maps; per-source fitted maps are descriptive oracles.
Preparation never loads a model. Native entry points require a protocol-bound
coordinator slot grant and enforce serial CPU/time/RSS limits.
"""
from __future__ import annotations
import argparse
import contextlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha, normalization
from experiments.feature_search.response_phase_study import (
    CENTERS, PHASE_COEFFICIENTS, J, basis, prediction, waveform_loss,
    response_matrix, split_matrix, validate as validate_old)
from experiments.feature_search.response_surface_focused import require_grant, validate
from experiments.feature_search.response_surface_crossed import wave

SEEDS = tuple(range(4954300, 4954316))
OFFSETS = (-.12, 0., .12)
BASELINES = ('zero', 'identity', 'scalar_bounded', 'scalar_unconstrained', 'complex_linear', 'full_H')


def fit_training(deltas, amplitudes, frequency):
    """Fit equal-source phase-pooled NMSE, using observed old responses only.

Each source is divided by its known stimulus amplitude, so normalization by
total ideal energy gives equal source weights. Complex coefficients are fitted
in waveform space too, rather than Frobenius-projecting an anisotropic H.
"""
    y = np.asarray(deltas, float) / np.asarray(amplitudes, float)[:, None, None]
    if y.ndim != 3 or y.shape[1:] != (4, 32) or not np.isfinite(y).all():
        raise ValueError('Need finite source x four-phase x horizon training responses')
    ideal = prediction(np.eye(2), 1., frequency)
    target = y.mean(axis=0)
    scalar = float(np.sum(ideal * target) / np.sum(ideal**2))
    complex_design = np.column_stack([ideal.ravel(), prediction(J, 1., frequency).ravel()])
    qr = np.linalg.lstsq(complex_design, target.ravel(), rcond=None)[0]
    design = np.column_stack([prediction(e.reshape(2, 2), 1., frequency).ravel() for e in np.eye(4)])
    full = np.linalg.lstsq(design, target.ravel(), rcond=None)[0].reshape(2, 2)
    maps = dict(zero=np.zeros((2, 2)), identity=np.eye(2),
                scalar_bounded=np.clip(scalar, 0., 1.) * np.eye(2),
                scalar_unconstrained=scalar * np.eye(2),
                complex_linear=qr[0] * np.eye(2) + qr[1] * J, full_H=full)
    losses = {name: float(np.mean([waveform_loss(prediction(h, 1., frequency), z, ideal) for z in y]))
              for name, h in maps.items()}
    return dict(matrices={k: v.tolist() for k, v in maps.items()},
                scalar_unconstrained=scalar, scalar_bounded=float(np.clip(scalar, 0., 1.)),
                complex_qr=qr.tolist(), training_nmse=losses)


def delayed_map(h, center, frequency):
    """Frozen principal-delay transport; center equality is exact, no fresh fit."""
    h = np.asarray(h, float)
    phase = np.arctan2(h[1, 0] - h[0, 1], np.trace(h))
    tau = -phase / (2 * np.pi * center / 128)
    angle = -(frequency - center) * 2 * np.pi / 128 * tau
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation @ h


def window_decomposition(delta, amplitude, frequency, start, stop):
    """Reuse response windows with absolute forecast timestamps and DC sensitivity."""
    y = np.asarray(delta, float)[:, start:stop]
    b = basis(frequency)[start:stop]
    odd = np.column_stack([(y[0] - y[2])/2, (y[1] - y[3])/2]) / amplitude
    even = np.column_stack([y[0] + y[2], y[1] + y[3]])
    h = np.linalg.lstsq(b, odd, rcond=None)[0]
    dc = np.linalg.lstsq(np.column_stack([b, np.ones(stop-start)]), odd, rcond=None)[0]
    residual = odd - b @ h
    return dict(start=start, stop=stop, H=h.tolist(), **split_matrix(h),
                H_dc_adjusted=dc[:2].tolist(), dc=dc[2].tolist(),
                offband_nmse=float(np.sum(residual**2)/np.sum(b**2)),
                even_nmse=float(np.sum(even**2)/(4*amplitude**2*np.sum(b**2))))


def source_draws(types, nboot=16384):
    """Common stratified whole-source draws for every prospective comparison."""
    types = np.asarray(types)
    rng = np.random.default_rng(4954300)
    return np.concatenate([rng.choice(np.flatnonzero(types == k), (nboot, np.sum(types == k)))
                           for k in sorted(set(types))], axis=1)


def paired_interval(differences, draws, family=4):
    """Source-paired mean and Bonferroni percentile intervals; no threshold verdict."""
    values = np.asarray(differences, float)
    boot = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()), wins=int(np.sum(values > 0)), sources=len(values),
                source_differences=values.tolist(), percentile95=np.quantile(boot, [.025, .975]).tolist(),
                family=family, family95_percentile=np.quantile(boot, [.025/family, 1-.025/family]).tolist())


def fresh_design():
    """400 saved inputs: sixteen independent backgrounds crossed with six tones/four phases."""
    inputs, nodes, sources, groups = [], [], [], []
    for sid, seed in enumerate(SEEDS):
        kind = sid % 4
        f, weight = [(17.35, .5), (17.35, 8.), (6.35, .5), (6.35, 8.)][kind]
        rng = np.random.default_rng(seed)
        phi, psi = rng.uniform(-np.pi, np.pi, 2)
        raw = .5*wave(5.35, phi) + weight*wave(f, psi) + rng.normal(0, .05, 128)
        x = ((raw-raw.mean())/raw.std()).astype(np.float32)
        scale = float(x.astype(float).std()); amplitude = .1*scale
        source = f'syn:{seed}'; base = len(inputs)
        sources.append(dict(source=source, seed=seed, background_type=kind,
                            target_phase=float(phi), distractor_phase=float(psi),
                            distractor_frequency=f, distractor_amplitude=weight))
        inputs.append(x); nodes.append(dict(index=base, source_id=sid, source=source, kind='background'))
        for center in CENTERS:
            for offset in OFFSETS:
                frequency = center + offset; indices = []
                for phase_id in range(4):
                    index = len(inputs); indices.append(index)
                    inputs.append((x.astype(float)+amplitude*wave(frequency, phase_id*np.pi/2)).astype(np.float32))
                    nodes.append(dict(index=index, source_id=sid, source=source, kind='signal', root=base,
                                      center=center, offset=offset, frequency=frequency, phase_id=phase_id))
                groups.append(dict(source=source, source_id=sid, background_type=kind, base=base,
                                   indices=indices, center=center, offset=offset, frequency=frequency,
                                   amplitude=amplitude, input_scale=scale))
    bank = np.stack(inputs)
    gates = input_gates(bank, groups)
    return bank, nodes, sources, groups, gates


def input_gates(bank, groups):
    """Check phase order, physical scale and antipodal closure before any native launch."""
    worst = 0.; closure = 0.
    for g in groups:
        observed = bank[g['indices']].astype(float)-bank[g['base']].astype(float)
        nominal = g['amplitude']*np.stack([wave(g['frequency'], k*np.pi/2) for k in range(4)])
        worst = max(worst, float(np.sqrt(np.mean((observed-nominal)**2))/g['amplitude']))
        closure = max(closure, float(np.sqrt(np.mean((observed[:2]+observed[2:])**2))/g['amplitude']))
    if worst > 1e-5 or closure > 2e-5:
        raise ValueError('Nominal phase/antipodal input gate failed')
    return dict(nominal_rms_over_peak_max=worst, antipodal_rms_over_peak_max=closure)


def prepare(out, old, proposal):
    """Bind old-only fitted predictors, fresh bank, code, inference family and resource caps."""
    old_protocol = validate_old(old); focused = Path(old_protocol['prior'])
    if not read(old/'audit.json')['passed']:
        raise ValueError('Old response arithmetic must be audited')
    groups_old = read(old/'groups.json'); raw = np.load(old/'raw_arithmetic.npz')
    old_rows = read(old/'matrix_rows.json'); predictors = []
    for model in ('bolt', 'timesfm'):
        for policy in ('native', 'locked'):
            for fi, frequency in enumerate(CENTERS):
                groups = [g for g in groups_old if g['frequency'] == frequency]
                if len(groups) != 16 or {g['source_id'] for g in groups} != set(range(16)):
                    raise ValueError('Training cohort must be all sixteen old sources')
                deltas = np.stack([raw[f'{model}_{policy}_{i}_{fi}_delta'] for i in range(16)])
                fitted = fit_training(deltas, [g['amplitude'] for g in groups], frequency)
                old_h = np.mean([r['H'] for r in old_rows if r['model'] == model and r['policy'] == policy and r['frequency'] == frequency], axis=0)
                np.testing.assert_allclose(fitted['matrices']['full_H'], old_h, atol=1e-12, rtol=0)
                predictors.append(dict(model=model, policy=policy, center=frequency, **fitted))
    bank, nodes, sources, groups, gates = fresh_design()
    old_sources = {g['source'] for g in groups_old}
    if old_sources & {s['source'] for s in sources}:
        raise ValueError('Fresh and training sources overlap')
    out.mkdir(parents=True, exist_ok=False)
    np.save(out/'inputs.npy', bank)
    np.save(out/'bootstrap_source_indices.npy', source_draws([s['background_type'] for s in sources]))
    for name, value in [('nodes', nodes), ('sources', sources), ('groups', groups), ('frozen_predictors', predictors)]:
        write(out/f'{name}.json', value)
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'tests/test_response_phase_transfer.py',
             root/'experiments/feature_search/response_phase_transfer_plants.py',
             root/'experiments/feature_search/response_phase_transfer_audit.py']
    files += [root/'experiments/feature_search'/f'{name}.py' for name in
              ['response_phase_study', 'response_surface_focused', 'response_surface_crossed',
               'selective_study', 'coherence_study', 'real_background_poc', 'homogeneous_normalizer']]
    files.append(root/'tsfm_lens/extraction/hooks.py')
    names = ['inputs.npy', 'nodes.json', 'sources.json', 'groups.json', 'frozen_predictors.json', 'bootstrap_source_indices.npy']
    primary = [dict(model=m, policy='native', period=19.2, baseline=b)
               for m in ('bolt', 'timesfm') for b in ('scalar_bounded', 'identity')]
    protocol = dict(question='Does a frozen old-source full phase map predict fresh weak-tone responses beyond attenuation and identity?',
        stage='prepared_capture_held', nodes=400, source_count=16, phases=4, context=128, horizon=32,
        seeds=list(SEEDS), offsets=list(OFFSETS), amplitude_peak_background_sd=.1,
        policies=['native', 'locked'], training_sources=sorted(old_sources),
        predictors=list(BASELINES),
        fit_objective='Equal-source mean of phase-pooled waveform NMSE on all16 OLD sources, separately model/policy/center. Scalar is waveform least-squares, bounded[0,1] primary, unconstrained diagnostic. Complex qI+rJ also waveform-fitted. FullH equals frozen old meanH. Zero is explicitly scored.',
        primary_family=primary,
        inference='Native19.2 fullH vs bounded scalar and vs identity, two models = family4. Paired source differences (baseline NMSE minus fullH NMSE),16384 common type-stratified source bootstrap draws seed4954300. Report ordinary95% and Bonferroni98.75% percentile CIs; approximate family coverage, no p-values or automatic threshold claims. Fresh synthetic generalization, not sealed confirmation.',
        descriptive='Locked normalization,24h,unconstrained scalar,zero,complexlinear,all nearby metrics and all horizon decompositions are descriptive. Source-specific fullH is same-data oracle only, never predictive.',
        nearby='All fixed center maps evaluated at exact nearby frequency. Compare fullH constant versus principal-delay transported fullH ONLY at offsets+/-0.12, equally pool the two offsets within source. No fresh fitting/selection/interpolation.',
        horizon_analysis='After primary scoring, reuse each phase delta on windows[0:16],[16:32],[0:32], preserving absolute forecast timestamps; H and joint-DC H, odd/even/offband. Apply also to old arrays after primary. No causal patch or LTI claim.',
        normalization='Frozen native normalization and background-locked normalization/inverse on identical inputs, batches<=4.',
        controls='Exact background repeat and native/locked identity in forecasts and transient endpoint hidden state, self-patch, same-batch forecast replay; finite checks, input phase/closure gates, hashes. Save forecasts only; endpoint hidden used for identity gates and discarded.',
        cpu_threads=1, worker_cap=3*2**30, seconds_per_model=600,
        checkpoints=read(focused/'protocol.json')['checkpoints'],
        budget=read(proposal/'proposal.json')['resource_estimate']['with_nearby_frequency_control'],
        execution='One native worker at a time. Both run and worker require external protocol-bound coordinator slot grant. Parent monitors RSS/wall time. Failure retained, no in-place retry or rescue claims.',
        capture_hold=True, old=str(old.resolve()), proposal=str(proposal.resolve()),
        old_hashes={n:sha(old/n) for n in ['protocol.json','audit.json','raw_arithmetic.npz','matrix_rows.json','groups.json']},
        source_hashes={str(f.resolve()):sha(f) for f in files}, hashes={n:sha(out/n) for n in names})
    write(out/'protocol.json', protocol)
    write(out/'knowledge.json', dict(status='prepared_native_capture_held', question=protocol['question'], design=protocol,
        exact_findings=dict(nodes=400, sources=16, gates=gates, old_training_predictors=predictors, fresh_results=None),
        controls=protocol['controls'], limitations='Preparation only. No new native evidence; four phases and finite windows cannot establish time invariance or patch causality.',
        evidence_class='Frozen prospective fresh-source response prediction; old-only fitted baselines',
        failed_instrumentation_runs=[], next_decision='HOLD native until coordinator slot grant after weather.'))
    print(dict(run=str(out.resolve()), protocol_sha256=sha(out/'protocol.json'), nodes=len(bank), native='HELD'), flush=True)


def capture_bank(out, native, p, inputs, nodes):
    """Forecast-only capture with endpoint state parity gates and exact batch replay."""
    started = time.monotonic(); model = native.name
    gates = dict(repeat_forecast=0., repeat_hidden=0., normalizer_forecast=0., normalizer_hidden=0., self_patch=0.)
    arrays = {policy: np.lib.format.open_memmap(out/f'{model}_{policy}_forecast.npy', mode='w+', dtype=np.float32, shape=(len(nodes), 32))
              for policy in ('native', 'locked')}
    for source in sorted({n['source_id'] for n in nodes}):
        ids = [n['index'] for n in nodes if n['source_id'] == source]
        base = next(n['index'] for n in nodes if n['source_id'] == source and n['kind'] == 'background')
        x = inputs[base:base+1].copy(); expected, hx = native.predict(x); repeat, hr = native.predict(x)
        with normalization(native, x, 'background'):
            locked, hl = native.predict(x)
        patched, _ = native.predict(x, replacement=hx)
        for key, a, b in [('repeat_forecast', expected, repeat), ('repeat_hidden', hx, hr),
                          ('normalizer_forecast', expected, locked), ('normalizer_hidden', hx, hl), ('self_patch', expected, patched)]:
            gates[key] = max(gates[key], float(np.max(np.abs(a-b))))
        if any(gates.values()) or not all(np.isfinite(a).all() for a in (expected,hx,repeat,hr,locked,hl,patched)):
            raise ValueError(f'Exact native identity gate failed: {gates}')
        for start in range(0, len(ids), 4):
            batch = ids[start:start+4]; anchor = np.repeat(x, len(batch), axis=0)
            for policy in ('native', 'locked'):
                ctx = contextlib.nullcontext() if policy == 'native' else normalization(native, anchor, 'background')
                with ctx:
                    pred, _ = native.predict(inputs[batch].copy())
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite forecast')
                arrays[policy][batch] = pred
            if time.monotonic()-started > p['seconds_per_model'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['worker_cap']:
                raise RuntimeError('Frozen resource cap exceeded')
        batch = ids[:4]; anchor = np.repeat(x, len(batch), axis=0)
        for policy in ('native', 'locked'):
            ctx = contextlib.nullcontext() if policy == 'native' else normalization(native, anchor, 'background')
            with ctx:
                pred, _ = native.predict(inputs[batch].copy())
            if not np.array_equal(pred, arrays[policy][batch]):
                raise ValueError('Same-batch replay failed')
        print(dict(model=model, source=source, seconds=time.monotonic()-started), flush=True)
    for a in arrays.values(): a.flush()
    return dict(gates=gates, batch_replay_exact=True, seconds=time.monotonic()-started,
                peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                hashes={f'{model}_{policy}_forecast.npy':sha(out/f'{model}_{policy}_forecast.npy') for policy in arrays})


def worker(out, model, grant):
    """Refuse all native loading until externally granted and hash-validated."""
    require_grant(out, grant, model); p = validate(out, model)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise ValueError('CPU one-thread environment required')
    if list(out.glob(f'{model}_*.npy')): raise ValueError('Refuse partial overwrite')
    from experiments.feature_search.real_background_poc import Native
    native = Native(model, p['checkpoints'][model]['path'])
    result = capture_bank(out, native, p, np.load(out/'inputs.npy', mmap_mode='r'), read(out/'nodes.json'))
    validate(out); write(out/f'{model}_worker.json', result)


def run(out, model, grant):
    """Monitor exactly one granted worker; preserve every execution including failures."""
    import fcntl
    release = require_grant(out, grant, model); p = validate(out)
    with (out/'native.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = read(out/'execution.json') if (out/'execution.json').exists() else {}
        if model in execution or list(out.glob(f'{model}_*.npy')): raise ValueError('Refuse repeat/partial overwrite')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        knowledge = read(out/'knowledge.json'); knowledge['status'] = 'native_running'; write(out/'knowledge.json', knowledge)
        started = time.monotonic(); peak = 0; failure = None
        with (out/f'{model}.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.response_phase_transfer', 'worker',
                                     '--out', str(out), '--model', model, '--grant', str(grant)], env=env, stdout=log, stderr=subprocess.STDOUT)
            print(dict(event='native_launch', model=model, pid=proc.pid, slot=release['slot'], run=str(out.resolve())), flush=True)
            while proc.poll() is None:
                try:
                    status = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8')
                    peak = max(peak, int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))*1024)
                except (FileNotFoundError, StopIteration): pass
                if peak > p['worker_cap'] or time.monotonic()-started > p['seconds_per_model']:
                    failure = 'Frozen resource cap exceeded'; proc.kill(); proc.wait(); break
                time.sleep(.2)
        wf = out/f'{model}_worker.json'
        if wf.exists() and read(wf)['peak_rss'] > p['worker_cap']: failure = 'Worker HWM exceeds cap'
        if proc.returncode != 0 and failure is None: failure = f'Native worker exit {proc.returncode}; see log'
        execution[model] = dict(valid=proc.returncode == 0 and failure is None and wf.exists(), returncode=proc.returncode,
            seconds=time.monotonic()-started, peak_rss=peak, failure=failure, grant_sha256=sha(grant), slot=release['slot'])
        write(out/'execution.json', execution)
        knowledge = read(out/'knowledge.json'); knowledge['execution'] = execution
        knowledge['status'] = 'captured_pending_analysis' if execution[model]['valid'] else 'failed_native_instrumentation'
        if not execution[model]['valid']: knowledge['failed_instrumentation_runs'].append(dict(model=model, **execution[model]))
        knowledge['next_decision'] = 'Release slot after serial captures; analyze only valid saved arrays. Preserve failures without rescue claims.'
        write(out/'knowledge.json', knowledge)
        print(dict(event='native_finish', model=model, **execution[model]), flush=True)


def analyze(out):
    """Score frozen maps, then descriptive oracle/horizon fits from saved arrays only."""
    p = validate(out); execution = read(out/'execution.json')
    if (out/'results.json').exists(): raise ValueError('Refuse overwrite analysis')
    groups = read(out/'groups.json'); predictors = read(out/'frozen_predictors.json')
    draws = np.load(out/'bootstrap_source_indices.npy'); rows = []; raw_saved = {}; horizons = []
    for model in ('bolt', 'timesfm'):
        if not execution[model]['valid']: raise ValueError('Invalid capture')
        worker_record = read(out/f'{model}_worker.json')
        if any(worker_record['gates'].values()) or not worker_record['batch_replay_exact']: raise ValueError('Invalid native gates')
        for policy in ('native', 'locked'):
            file = f'{model}_{policy}_forecast.npy'
            if sha(out/file) != worker_record['hashes'][file]: raise ValueError('Forecast changed')
            forecast = np.load(out/file, mmap_mode='r')
            for gi, g in enumerate(groups):
                fitted = next(r for r in predictors if r['model'] == model and r['policy'] == policy and r['center'] == g['center'])
                metrics, raw = response_matrix(forecast[g['indices']], forecast[g['base']], g['amplitude'], g['frequency'])
                maps = {name:np.asarray(h) for name,h in fitted['matrices'].items()}
                maps['oracle_same_source'] = raw['H']
                maps['delay_full_H'] = delayed_map(maps['full_H'], g['center'], g['frequency'])
                pred = {name:prediction(h, g['amplitude'], g['frequency']) for name,h in maps.items()}
                losses = {name:waveform_loss(v, raw['delta'], raw['ideal']) for name,v in pred.items()}
                rows.append(dict(g, model=model, policy=policy, losses=losses, decomposition=metrics))
                key = f'{model}_{policy}_{gi}'
                raw_saved.update({f'{key}_{name}':v for name,v in raw.items()})
                raw_saved.update({f'{key}_prediction_{name}':v for name,v in pred.items()})
                for start, stop in [(0,16),(16,32),(0,32)]:
                    horizons.append(dict(cohort='fresh', model=model, policy=policy, source=g['source'], center=g['center'],
                        offset=g['offset'], **window_decomposition(raw['delta'], g['amplitude'], g['frequency'], start, stop)))
    primary = []
    for comparison in p['primary_family']:
        selected = [r for r in rows if r['model'] == comparison['model'] and r['policy'] == comparison['policy'] and r['center'] == CENTERS[1] and r['offset'] == 0]
        delta = [r['losses'][comparison['baseline']] - r['losses']['full_H'] for r in selected]
        primary.append(dict(comparison, **paired_interval(delta, draws)))
    descriptive = []; nearby = []
    for model in ('bolt','timesfm'):
        for policy in ('native','locked'):
            for center in CENTERS:
                for offset in OFFSETS:
                    selected = [r for r in rows if r['model']==model and r['policy']==policy and r['center']==center and r['offset']==offset]
                    descriptive.append(dict(model=model, policy=policy, center=center, offset=offset,
                        mean_nmse={name:float(np.mean([r['losses'][name] for r in selected])) for name in selected[0]['losses']}))
                differences = [np.mean([r['losses']['full_H']-r['losses']['delay_full_H'] for r in rows
                    if r['model']==model and r['policy']==policy and r['center']==center and r['offset']!=0 and r['source_id']==sid]) for sid in range(16)]
                nearby.append(dict(model=model, policy=policy, center=center, **paired_interval(differences, draws, family=1)))
    old = Path(p['old'])
    for file,digest in p['old_hashes'].items():
        if sha(old/file)!=digest: raise ValueError('Old arithmetic binding changed')
    old_raw = np.load(old/'raw_arithmetic.npz'); old_groups = read(old/'groups.json')
    for model in ('bolt','timesfm'):
        for policy in ('native','locked'):
            for fi, frequency in enumerate(CENTERS):
                for g in [g for g in old_groups if g['frequency']==frequency]:
                    delta=old_raw[f'{model}_{policy}_{g["source_id"]}_{fi}_delta']
                    for start,stop in [(0,16),(16,32),(0,32)]:
                        horizons.append(dict(cohort='old',model=model,policy=policy,source=g['source'],center=frequency,offset=0.,
                            **window_decomposition(delta,g['amplitude'],frequency,start,stop)))
    np.savez_compressed(out/'raw_arithmetic.npz', **raw_saved)
    write(out/'source_rows.json', rows); write(out/'horizon_decomposition.json', horizons)
    results = dict(primary=primary, descriptive=descriptive, nearby_constant_minus_delay=nearby)
    write(out/'results.json', results)
    knowledge = read(out/'knowledge.json'); knowledge.update(status='computed_pending_audit', exact_findings=results,
        evidence_class='Fresh synthetic source transfer of frozen old-only response predictors; conditional behavioral evidence',
        next_decision='Independently audit raw arithmetic, training-only fits, source pairing and interval family before interpretation.')
    write(out/'knowledge.json', knowledge)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker','analyze'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--old',type=Path,default=Path('runs/response_phase4954200'))
    parser.add_argument('--proposal',type=Path,default=Path('runs/response_phase_transfer_proposal4954300'))
    parser.add_argument('--model',choices=['bolt','timesfm']);parser.add_argument('--grant',type=Path)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.out,args.old,args.proposal)
    elif args.command=='analyze': analyze(args.out)
    else:
        if args.model is None: parser.error('--model required')
        (run if args.command=='run' else worker)(args.out,args.model,args.grant)
