"""Fresh, unmodified real sources with two retrospective selector windows.

Prepare is model-free. Native execution requires a coordinator-granted slot and
uses serial monitored CPU workers. Only analysis opens any target archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from experiments.feature_search.coherence_study import DOMAINS, Bank, load_bank, read, real_data, sha, write
from experiments.feature_search.selective_tree import continuation, decompose

ORIGINS = (192, 224, 256)
DEPTHS = (1, 2, 3)
MODELS = ('bolt', 'timesfm')
METHODS = ('native', 'hybrid1', 'hybrid2', 'hybrid3',
           'output_free1', 'output_free2', 'output_free3', 'seasonal_naive24',
           'tone_last1', 'tone_last2', 'tone_last3')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def record_knowledge(out, phase, status, findings, next_decision):
    """Persist exact outcomes, including failed instrumentation, for later sessions."""
    out.mkdir(parents=True, exist_ok=True)
    path = out/'knowledge.json'
    events = read(path).get('events', []) if path.exists() else []
    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), phase=phase, status=status,
        question='Do selective hybrids improve unmodified real forecasts, and can two past hindcasts identify continuation failure?',
        design=dict(requested_sources=32, domains=list(DOMAINS), ranks=list(range(64, 72)),
            split='4 FIT / 4 TEST per domain', origins=list(ORIGINS), history=128, horizon=32,
            selector='Strict lower mean of two own-history-normalized hindcast losses; ties native',
            methods=list(METHODS), native_budget='Sequential CPU, one compute thread, 3GiB, 600 seconds/model'),
        exact_findings=findings,
        controls=['Independent origin normalization', 'Future poisoning and scale invariance',
                  'Target-blind worker', 'Fixed depths and native ties', 'Source freshness and FIT/TEST separation',
                  'Hash, finite, repeat, reconstruction and resource gates'],
        limitations=['Pilot is exploratory; hindcasts are overlapping past selector features, not independent prospective tests',
                     'Stability/abstention are descriptive cues', 'Cached-source pretraining overlap unknown',
                     'No scientific improvement claim before valid native forecasts and held-out analysis'],
        evidence_class='instrumentation' if phase != 'analysis' or status != 'complete' else 'exploratory behavioral/descriptive',
        next_decision=next_decision)
    events.append(entry)
    write(path, dict(schema_version=1, **entry, events=events))


def normalize_history(history):
    """Undo any inherited affine normalization using ONLY this origin's past."""
    x = np.asarray(history, dtype=float)
    if x.shape != (128,) or not np.isfinite(x).all():
        raise ValueError('Expected finite 128-point history')
    center, scale = float(x.mean()), float(x.std())
    valid = bool(np.ptp(x) > 0 and scale > 0)
    scale = scale if valid else 1.
    if not valid:
        return np.zeros_like(x), center, scale, valid
    return (x-center)/scale, center, scale, valid


def build(sources):
    """Construct a target-free bank and separate analysis targets in origin units."""
    bank = Bank()
    rows, plans, hindcasts, futures = [], [], [], []
    for meta, value in sources:
        value = np.asarray(value, float)
        if value.shape != (288,) or not np.isfinite(value).all():
            raise ValueError('Selected source invalid; never replace using its outcome')
        row = {k: meta[k] for k in ('source', 'domain', 'rank', 'split')}
        row['split'] = row['split'].upper()
        row['past_sha256'] = hashlib.sha256(value[:256].tobytes()).hexdigest()
        row['index'] = len(rows)
        rows.append(row)
        source_plans, targets = [], []
        for origin in ORIGINS:
            x, center, scale, valid = normalize_history(value[origin-128:origin])
            levels, details = decompose(x, depth=3)
            ids = [[bank.add(part) for part in level] for level in levels]
            closure = max(float(np.max(abs(np.sum(level, axis=0)-x))) for level in levels)
            if closure > 1e-12:
                raise ValueError('FP64 reconstruction gate failed')
            source_plans.append(dict(origin=origin, history=[origin-128, origin],
                target=[origin, origin+32], center=center, scale=scale,
                selector_usable=valid, levels=ids, zero=bank.add(np.zeros(128)),
                details=details, extracted_depth=len(details), abstained=not details,
                capped_depth={str(d): min(d, len(details)) for d in DEPTHS},
                closure_max=closure,
                standardized_amplitudes=[float(np.linalg.norm(d['coefficients'])) for d in details]))
            targets.append((value[origin:origin+32]-center)/scale)
        plans.append(source_plans)
        hindcasts.append(targets[:2])
        futures.append(targets[2])
    return bank, rows, plans, np.asarray(hindcasts), np.asarray(futures)


def prior_sources(root):
    """Audit recorded natural-source IDs in sibling study row/metadata artifacts."""
    found, files = set(), {}
    for pattern in ('*/rows.json', '*/metadata.json'):
        for path in sorted(root.glob(pattern)):
            data = read(path)
            if not isinstance(data, list):
                continue
            ids = set()
            for row in data:
                if not isinstance(row, dict):
                    continue
                source = row.get('source', '')
                if isinstance(source, str) and any(source.startswith(d+':') for d in DOMAINS):
                    ids.add(source)
                if row.get('dataset') in DOMAINS and 'id' in row:
                    ids.add(row['dataset']+':'+str(row['id']))
            if ids:
                found.update(ids)
                files[str(path)] = sha(path)
    return found, files


def prepare(out, cache):
    """Prepare a fresh run and distill either success or the exact failed attempt."""
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    try:
        _prepare(out, cache)
    except Exception as exc:
        record_knowledge(out, 'prepare', 'failed', dict(error_type=type(exc).__name__, error=str(exc),
            native_models_loaded=False, native_workers_launched=False),
            'Resolve the preparation failure explicitly; do not substitute sources or launch native workers.')
        raise
    p = read(out/'protocol.json')
    record_knowledge(out, 'prepare', 'complete', dict(sources=p['rows'], bank_nodes=p['nodes'],
        native_workers_launched=False, protocol_sha256=sha(out/'protocol.json')),
        'Await coordinator slot grant before serial native execution.')


def _prepare(out, cache):
    """Freeze ranks64..71, 4 FIT + 4 TEST/domain, without loading native models."""
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    sources = real_data(cache, 64, 8)
    ids = [m['source'] for m, _ in sources]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('Need 32 distinct source IDs')
    for domain in DOMAINS:
        if [m['rank'] for m, _ in sources if m['domain'] == domain] != list(range(64, 72)):
            raise ValueError('Rank contract changed')
    previous, audited = prior_sources(out.parent)
    if previous.intersection(ids):
        raise ValueError(f'Source reuse: {sorted(previous.intersection(ids))}')
    bank, rows, plans, hind, future = build(sources)
    checkpoints = read(out.parent/'fourier_poc4929000/protocol.json')['checkpoints']
    from experiments.feature_search import coherence_study, selective_tree, real_background_poc, homogeneous_normalizer
    from experiments.feature_search import natural_validation_analysis
    modules = [coherence_study, selective_tree, real_background_poc, homogeneous_normalizer,
               natural_validation_analysis]
    paths = [Path(__file__)] + [Path(m.__file__) for m in modules]
    out.mkdir(parents=True, exist_ok=False)
    bank.save(out/'bank.npz')
    write(out/'rows.json', rows)
    write(out/'plans.json', plans)
    np.savez_compressed(out/'hindcast_targets.npz', targets=hind)
    for split in ('FIT', 'TEST'):
        ix = np.asarray([i for i, row in enumerate(rows) if row['split'] == split])
        np.savez_compressed(out/f'{split}_targets.npz', indices=ix, targets=future[ix])
    write(out/'provenance.json', dict(sources=[m for m, _ in sources],
        freshness=dict(overlap=[], audited_artifacts=audited,
                       scope='Recorded sibling rows/metadata only; pretraining overlap unknown')))
    files = ['bank.npz', 'rows.json', 'plans.json', 'hindcast_targets.npz',
             'FIT_targets.npz', 'TEST_targets.npz', 'provenance.json']
    protocol = dict(version=1, run='natural_validation_pilot4951000', source_ids=ids,
        ranks=list(range(64, 72)), domains=list(DOMAINS), rows=32, nodes=len(bank.values),
        origins=list(ORIGINS), methods=list(METHODS), models=list(MODELS), checkpoints=checkpoints,
        worker_cap=3*2**30, seconds_per_model=600, cpu_threads=1, gpu=False,
        eligibility='coherence_study.real_data cached helper: past256 finite, main history nonconstant, hourly; no outcome replacement',
        normalization='Every origin independently centers/scales its own history, after helper normalization',
        tree='Fixed selective_tree defaults: stability .65, explained .15, prominence 8, max depth3',
        selector='For each fixed depth: hybrid iff mean of TWO own-history-standardized hindcast MSEs < native; exact ties/native and unusable past/native',
        fit_rule='Per domain/model choose native, fixed hybrid1/2/3 or past-selector1/2/3 by FIT future MSE; native first on ties; freeze before TEST opens',
        gates=dict(closure_fp64=1e-12, closure_fp32=2e-6, repeat_exact=True,
                   finite=True, fresh_source_ids=True, source_split_disjoint=True),
        interpretation='Exploratory behavioral pilot on unmodified real trajectories. Hindcasts overlap main past and are selector features, NOT independent prospective tests. Stability and abstention are descriptive cues only. No best TEST depth deployed; no significance/causal claim.',
        effect_gate='Descriptive per-domain FIT and TEST paired mean MSE gains both >0; no promotion or automatic expansion',
        worker_reads=['protocol.json', 'bank.npz', 'rows.json', 'plans.json'],
        sources={str(p.resolve()): sha(p) for p in paths},
        hashes={f: sha(out/f) for f in files})
    write(out/'protocol.json', protocol)
    print(dict(status='prepared; native slot pending', sources=32, nodes=len(bank.values)), flush=True)


def assemble(bank, plans, forecasts):
    """Compose all fixed policies in local origin units; preserve residual/DC."""
    result = np.zeros((len(plans), 3, len(METHODS), 32), float)
    closure = 0.
    for i, source in enumerate(plans):
        for j, plan in enumerate(source):
            root = plan['levels'][0][0]
            x = bank[root].astype(float)
            result[i, j, 0] = forecasts[root]
            result[i, j, METHODS.index('seasonal_naive24')] = np.resize(x[-24:], 32)
            for requested in DEPTHS:
                depth = min(requested, plan['extracted_depth'])
                ids = plan['levels'][depth]
                leaves = np.asarray([bank[k] for k in ids], float)
                closure = max(closure, float(np.max(abs(leaves.sum(0)-x))/max(1., abs(x).max())))
                tone = continuation(plan['details'], depth)
                result[i, j, METHODS.index(f'hybrid{requested}')] = tone+forecasts[ids[-1]]
                result[i, j, METHODS.index(f'output_free{requested}')] = forecasts[ids].sum(0)-depth*forecasts[plan['zero']]
                result[i, j, METHODS.index(f'tone_last{requested}')] = tone+leaves[-1, -1]
    if closure > 2e-6 or not np.isfinite(result).all():
        raise ValueError('Finite/reconstruction gate failed')
    return result, closure


def worker(out, model, native_factory=None):
    """Target-blind worker; native_factory enables model-free instrumentation tests."""
    if os.environ.get('NATURAL_VALIDATION_MONITORED') != '1':
        raise RuntimeError('Use run --slot-granted under the resource monitor')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in THREAD_VARS):
        raise RuntimeError('Requires CPU and one compute thread')
    started = time.monotonic()
    p = read(out/'protocol.json')
    for path, digest in p['sources'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed')
    for name in ('bank.npz', 'rows.json', 'plans.json'):
        if sha(out/name) != p['hashes'][name]:
            raise ValueError('Frozen worker input changed')
    if (out/f'{model}_forecasts.npz').exists():
        raise FileExistsError('Refuse forecast overwrite')
    checkpoint = p['checkpoints'][model]
    for name, digest in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/name) != digest:
            raise ValueError('Checkpoint changed')
    if native_factory is None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from experiments.feature_search.real_background_poc import Native
        native_factory = Native
    native = native_factory(model, checkpoint['path'])
    values = load_bank(out/'bank.npz')
    forecasts = np.zeros((len(values), 32), np.float32)
    for start in range(0, len(values), 4):
        forecasts[start:start+4] = native.predict(np.stack(values[start:start+4]))[0]
        if time.monotonic()-started > p['seconds_per_model'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['worker_cap']:
            raise RuntimeError('Resource cap exceeded')
    repeat = native.predict(np.stack(values[:4]))[0]
    if not np.array_equal(repeat, forecasts[:4]) or not np.isfinite(forecasts).all():
        raise ValueError('Finite/repeat gate failed')
    result, closure = assemble(values, read(out/'plans.json'), forecasts)
    np.savez_compressed(out/f'{model}_forecasts.npz', forecasts=result, bank_forecasts=forecasts,
        source_ids=np.asarray(p['source_ids']), methods=np.asarray(METHODS), origins=np.asarray(ORIGINS))
    write(out/f'{model}_worker.json', dict(protocol_sha256=sha(out/'protocol.json'),
        sha256=sha(out/f'{model}_forecasts.npz'), repeat_exact=True, finite=True,
        closure_fp32=closure, seconds=time.monotonic()-started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(out, slot_granted=False):
    """Distill every authorized execution attempt, including failed instrumentation."""
    if slot_granted is not True:
        raise RuntimeError('Native workers blocked: coordinator slot not granted')
    try:
        _run(out, slot_granted)
    except Exception as exc:
        record_knowledge(out, 'native_execution', 'failed', dict(error_type=type(exc).__name__,
            error=str(exc), execution=read(out/'execution.json') if (out/'execution.json').exists() else {}),
            'Inspect failure and gates; invalid forecasts cannot support inference.')
        raise
    record_knowledge(out, 'native_execution', 'complete', read(out/'execution.json'),
                     'Freeze past choices and FIT rules, then evaluate held-out future targets.')


def _run(out, slot_granted=False):
    """Execute models sequentially only after an explicit external slot grant."""
    import fcntl
    with (out/'native.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = read(out/'protocol.json')
        if (out/'execution.json').exists():
            raise FileExistsError('Refuse rerun in a frozen run')
        execution = {}
        for model in MODELS:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1',
                NATURAL_VALIDATION_MONITORED='1', PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                **{k: '1' for k in THREAD_VARS})
            started, peak, reason = time.monotonic(), 0, None
            with (out/f'{model}.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.natural_validation_study',
                    'worker', '--out', str(out), '--model', model], env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        try:
                            lines = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                            peak = max(peak, int(next(l.split()[1] for l in lines if l.startswith('VmRSS:')))*1024)
                        except (FileNotFoundError, StopIteration):
                            pass
                        if peak > p['worker_cap'] or time.monotonic()-started > p['seconds_per_model']:
                            reason = 'Resource cap exceeded'
                            break
                        time.sleep(.1)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            worker_path = out/f'{model}_worker.json'
            record = read(worker_path) if worker_path.exists() else {}
            valid = proc.returncode == 0 and reason is None and bool(record)
            valid = valid and record.get('peak_rss', float('inf')) <= p['worker_cap']
            valid = valid and record.get('seconds', float('inf')) <= p['seconds_per_model']
            execution[model] = dict(valid=valid, returncode=proc.returncode, reason=reason,
                seconds=time.monotonic()-started, monitored_peak=peak)
            write(out/'execution.json', execution)
            if not valid:
                raise RuntimeError(f'{model} invalid; see execution/log; stopping sequential run')


def main():
    """Model-free preparation and explicitly gated native execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--out', type=Path, default=Path('runs/natural_validation_pilot4951000'))
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--slot-granted', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare':
        prepare(a.out, a.cache)
    elif a.command == 'run':
        run(a.out, a.slot_granted)
    else:
        if a.model is None:
            parser.error('worker requires --model')
        worker(a.out, a.model)


if __name__ == '__main__':
    main()
