"""Bounded no-oracle decomposition study on synthetic and cached real histories.

Prepare freezes every model input before inference. Workers only forecast a
cached, deduplicated node bank. Model-free analysis consumes futures afterwards.
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
from experiments.feature_search import fourier_adapter as adapter

SEEDS = tuple(range(4929000, 4929016))
DATASETS = ('monash_electricity_hourly', 'm4_hourly', 'monash_pedestrian_counts')
MODELS = ('bolt', 'timesfm')
CAP = 3*2**30
TIMEOUT = 600


def write(path, value):
    """Finite JSON only."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    """Read UTF-8 records."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """Hash exact source/artifact bytes."""
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def synthetic():
    """Ground truth is stored for evaluation, never passed into decomposition."""
    t = np.arange(-128, 32)
    records, values = [], []
    for index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        bins = np.array([3., 5., 9., 17.])
        amplitudes = rng.uniform(.5, 1.5, 4)
        amplitudes /= np.sqrt(np.sum(amplitudes**2)/2)
        phases = rng.uniform(-np.pi, np.pi, 4)
        def waves(phase, freq=bins):
            return amplitudes[:, None]*np.sin(2*np.pi*freq[:, None]*t/128+phase[:, None])
        original = waves(phases).sum(0)
        crest = np.max(abs(original[:128]))/np.std(original[:128])
        candidates = []
        for _ in range(256):
            candidate_phase = rng.uniform(-np.pi, np.pi, 4)
            candidate = waves(candidate_phase).sum(0)
            mismatch = abs(np.max(abs(candidate[:128]))/np.std(candidate[:128])-crest)/crest
            candidates.append((float(mismatch), candidate_phase, candidate))
        mismatch, matched_phase, matched = min(candidates, key=lambda row: row[0])
        cases = dict(stationary=original, phase_coherent=waves(np.zeros(4)).sum(0),
                     phase_matched=matched, off_grid=waves(phases, bins+.35).sum(0),
                     chirp=(amplitudes[:, None]*np.sin(2*np.pi*(bins[:, None]*t/128+.6*(t/128)**2)+phases[:, None])).sum(0),
                     coupled=original*(1+.35*np.sin(2*np.pi*2*t/128+phases[0])))
        for case, value in cases.items():
            records.append(dict(source=f'synthetic:{seed}', domain='synthetic', family=case,
                                split='discovery' if index < 8 else 'check', seed=seed,
                                scale=float(np.std(value[:128])), tree='all' if case == 'stationary' else 'contiguous',
                                phase_crest_mismatch=mismatch, phase_control_valid=mismatch <= .03,
                                evaluator_only_bins=bins.tolist(), evaluator_only_amplitudes=amplitudes.tolist(),
                                evaluator_only_phases=(matched_phase if case == 'phase_matched' else phases).tolist()))
            values.append(value)
        noise = rng.normal(0, .05, len(t))
        target = .5*np.sin(2*np.pi*5*t/128+.3)
        for distractor_bin in [6, 17]:
            for amplitude in [.5, 2., 8.]:
                background = amplitude*np.sin(2*np.pi*distractor_bin*t/128+phases[0])+noise
                for sign in [0, -1, 1]:
                    records.append(dict(source=f'synthetic:{seed}', domain='synthetic', family='distractor',
                                        split='discovery' if index < 8 else 'check', seed=seed, scale=1., tree='four',
                                        distractor_bin=distractor_bin, distractor_amplitude=amplitude, sign=sign))
                    values.append(background+sign*target)
    return records, values


def real_sources(cache):
    """New source ranks8–15, disjoint from the earlier ranks0–7 pilot."""
    import pyarrow as pa
    records, values = [], []
    t = np.arange(-128, 32)
    target = .5*np.sin(2*np.pi*5*t/128+.3)
    for name in DATASETS:
        paths = sorted((cache/'datasets/autogluon___chronos_datasets'/name).rglob('*train.arrow'))
        if len(paths) != 1:
            raise ValueError('Expected exactly one cached Arrow file per dataset')
        with pa.memory_map(str(paths[0]), 'r') as f:
            table = pa.ipc.open_stream(f).read_all()
        selected = []
        for row in np.random.default_rng(492306).permutation(len(table)):
            record = table.slice(int(row), 1).to_pylist()[0]
            series = np.asarray(record['target'], dtype=float)
            if len(series) < 288:
                continue
            window = series[-288:]
            scale = float(np.std(window[128:256]))
            stamps = np.asarray(record['timestamp'][-288:], dtype='datetime64[ms]')
            if not np.isfinite(window[:256]).all() or scale <= 1e-8 or not np.all(np.diff(stamps) == np.timedelta64(1, 'h')):
                continue
            selected.append((record, window, scale))
            if len(selected) == 16:
                break
        if len(selected) < 16:
            raise ValueError('Not enough eligible independent sources')
        for index, (record, window, scale) in enumerate(selected[8:16]):
            if not np.isfinite(window).all():
                raise ValueError('Selected future missing; do not silently replace source')
            center = float(window[128:256].mean())
            series = (window[128:]-center)/scale
            for sign in [0, -1, 1]:
                records.append(dict(source=f"{name}:{record['id']}", domain=name, family='real',
                                    split='discovery' if index < 4 else 'check', seed=493000+len(records),
                                    scale=1., tree='all' if sign == 0 else 'contiguous', sign=sign,
                                    source_rank=8+index, original_center=center, original_scale=scale,
                                    observed_end=str(record['timestamp'][-1]),
                                    data_sha256=hashlib.sha256(window.tobytes()).hexdigest(),
                                    pretraining_status='unknown; NOT unseen-pretraining evidence'))
                values.append(series+sign*target)
    return records, values


class Bank:
    """Deduplicate exact FP32 model contexts across nodes, groupings and identity controls."""
    def __init__(self):
        self.inputs = []
        self.index = {}

    def add(self, context):
        """Return a stable ID; reject shape/nonfinite errors before model work."""
        value = np.asarray(context, dtype=np.float32)
        if value.shape != (128,) or not np.isfinite(value).all():
            raise ValueError('Bad model context')
        value = value.copy()
        value[value == 0] = 0
        key = value.tobytes()
        if key not in self.index:
            self.index[key] = len(self.inputs)
            self.inputs.append(value)
        return self.index[key]


def plan(history, policy, seed, bank, detrend='mean'):
    """Only histories enter tree construction, including random-control selection."""
    leaves = adapter.decompose(history, detrend)
    orders, matching = adapter.grouping_orders(leaves, seed)
    if policy != 'all':
        orders = {'contiguous': orders['contiguous']}
    levels = [1, 4] if policy == 'four' else list(adapter.LEVELS)
    schemes = {}
    for name, order in orders.items():
        schemes[name] = {}
        for k in levels:
            nodes = []
            for group in adapter.groups(order, k):
                value = leaves[list(group)].sum(0)
                if k == 1:
                    value = np.asarray(history)
                nodes.append(dict(leaves=list(group), bank_id=bank.add(value)))
            schemes[name][str(k)] = nodes
    closure = float(np.max(abs(leaves.sum(0)-history)))
    return dict(schemes=schemes, detrend=detrend, matching=matching,
                reconstruction_max_abs=closure, leaf_energy=np.mean(leaves**2, axis=1).tolist())


def prepare(out, cache):
    """Freeze corpus, input-only tree bank, full comparison family and resource bounds."""
    from experiments.feature_search import real_background_poc, homogeneous_normalizer
    out.mkdir(parents=True, exist_ok=False)
    records, values = synthetic()
    r, v = real_sources(cache)
    records.extend(r); values.extend(v)
    data = np.asarray(values, dtype=np.float64)
    bank = Bank()
    zero = bank.add(np.zeros(128))
    plans = []
    for record, series in zip(records, data):
        plans.append(plan(series[:128], record['tree'], record['seed'], bank))
        if record['tree'] == 'all':
            plans[-1]['linear_detrend'] = plan(series[:128], 'contiguous', record['seed'], bank, 'linear')
    inputs = np.asarray(bank.inputs)
    if len(inputs) > 6500:
        raise ValueError(f'Input bank exceeds fixed6500 cap: {len(inputs)}')
    checkpoints = {}
    for model, repo in [('bolt', 'amazon/chronos-bolt-small'), ('timesfm', 'google/timesfm-2.5-200m-pytorch')]:
        paths = sorted((cache/'hub'/('models--'+repo.replace('/', '--'))/'snapshots').iterdir())
        if len(paths) != 1:
            raise ValueError('Need one pinned cached revision')
        checkpoints[model] = dict(repo=repo, path=str(paths[0]), revision=paths[0].name,
                                  weights={p.name: digest(p) for p in paths[0].glob('*.safetensors')})
    np.savez_compressed(out/'corpus.npz', history=data[:, :128], future=data[:, 128:])
    np.savez_compressed(out/'bank.npz', inputs=inputs)
    write(out/'rows.json', records)
    write(out/'plans.json', plans)
    source_files = [Path(__file__), Path(adapter.__file__), Path(real_background_poc.__file__), Path(homogeneous_normalizer.__file__)]
    protocol = dict(version=1, history=128, horizon=32, leaf_count=8, levels=adapter.LEVELS,
                    source_seed_ids=SEEDS, real_source_rank=[8, 15], source_count=40, rows=len(data), node_count=len(inputs), zero_id=zero,
                    primary='Mean detrending, exact contiguous spectral leaves; root raw history, level once; retain all energy',
                    comparisons=['K1/2/4/8 curves', 'same leaves contiguous/interleaved/energy-matched random grouping',
                                 'same spectrum phase changes/coherent and crest-matched', 'distractor amplitude and distance',
                                 'off-grid/chirp/coupled stress', 'real paired +/- additions', 'linear-detrend sensitivity'],
                    primary_summary='Heldout check half reported separately from discovery; no tuning or test-selected K; K4 primary paired contrast',
                    nomination='Absolute paired mean effect>=0.10 fixed source-scale squared error, same sign discovery/check; candidates only, no automatic expansion or significance claim',
                    crossing='First tested K with normalized error<=0.25; censored if none; not semantic interpretability',
                    uncertainty='1999 source bootstrap, conditional on dataset/design, descriptive unadjusted95%; no multiple-testing claims',
                    validity='FP64 reconstruction<=1e-12 relative; native FP32 input reconstruction<=2e-6 relative; K1 identity exact; phase crest<=3%; random energy mismatch<=0.1',
                    amplitude='Known additions .5; period128/5. Real inputs use their own past128 SD; distractor target/noise fixed',
                    split='Synthetic first8 seeds discovery,last8 check; real4+4 per dataset; same source all conditions same split',
                    data_provenance='Cached real sources have unknown pretraining overlap; fresh trajectories do not imply unfamiliar generators',
                    scope='Behavioral no-oracle adapter diagnostic, no internal causal claim or newly fitted SAE',
                    budget=dict(cpu_threads=1, seconds_per_model=TIMEOUT, worker_bytes=CAP, max_inputs=6500, gpu=False),
                    checkpoints=checkpoints, source_bindings={str(p.resolve()): digest(p) for p in source_files},
                    artifact_hashes={name: digest(out/name) for name in ['corpus.npz', 'bank.npz', 'rows.json', 'plans.json']})
    write(out/'protocol.json', protocol)
    print(json.dumps(dict(rows=len(data), nodes=len(inputs), independent_sources=40)), flush=True)


def worker(out, model):
    """Inference reads only the node bank, never targets or synthetic parameters."""
    from experiments.feature_search.real_background_poc import Native
    started = time.monotonic()
    protocol = read(out/'protocol.json')
    if any(digest(path) != expected for path, expected in protocol['source_bindings'].items()):
        raise ValueError('Source binding changed after freeze')
    if digest(out/'bank.npz') != protocol['artifact_hashes']['bank.npz']:
        raise ValueError('Model input binding changed')
    checkpoint = protocol['checkpoints'][model]
    if any(digest(Path(checkpoint['path'])/name) != expected for name, expected in checkpoint['weights'].items()):
        raise ValueError('Checkpoint changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only worker must hide CUDA')
    model_obj = Native(model, checkpoint['path'])
    inputs = np.load(out/'bank.npz')['inputs']
    predictions = []
    for start in range(0, len(inputs), 64):
        prediction, _ = model_obj.predict(inputs[start:start+64])
        predictions.append(prediction)
        if start % 512 == 0:
            print(model, start, '/', len(inputs), 'elapsed', time.monotonic()-started, flush=True)
    predictions = np.concatenate(predictions)
    repeat, _ = model_obj.predict(inputs[:4])
    if not np.array_equal(repeat, predictions[:4]) or not np.isfinite(predictions).all():
        raise ValueError('Native reproducibility/finite gate failed')
    np.savez_compressed(out/f'{model}_forecasts.npz', predictions=predictions)
    record = dict(model=model, node_count=len(inputs), native_batch=4, native_calls=int(np.ceil(len(inputs)/4))+1,
                  native_context_tokens=int(len(inputs)*(128//model_obj.patch+(model == 'bolt'))),
                  patch=model_obj.patch, checkpoint=checkpoint, seconds=time.monotonic()-started,
                  peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  repeat_max_abs=float(np.max(abs(repeat-predictions[:4]))),
                  prediction_sha256=digest(out/f'{model}_forecasts.npz'),
                  installed_model_source_sha256=digest(sys.modules[type(model_obj.model).__module__].__file__))
    write(out/f'{model}_execution.json', record)
    print(model, 'complete', record['seconds'], flush=True)


def run(out):
    """Serial CPU workers; retain failures and invalidate over-budget results."""
    execution = {}
    protocol = read(out/'protocol.json')
    for model in MODELS:
        if (out/f'{model}_forecasts.npz').exists():
            raise ValueError('Refuse to overwrite existing inference')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        started = time.monotonic(); peak = 0; failure = None
        with (out/f'{model}.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.fourier_poc', 'worker',
                                      '--out', str(out), '--model', model], env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:
                    status = Path(f'/proc/{child.pid}/status').read_text(encoding='utf-8')
                    rss = int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))*1024
                    peak = max(peak, rss)
                except (FileNotFoundError, StopIteration):
                    pass
                if peak > protocol['budget']['worker_bytes'] or time.monotonic()-started > protocol['budget']['seconds_per_model']:
                    failure = 'Frozen time/memory cap exceeded'
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    break
                time.sleep(.2)
        completed = out/f'{model}_execution.json'
        if completed.exists() and read(completed)['peak_rss'] > protocol['budget']['worker_bytes']:
            failure = 'Worker high-water memory cap exceeded'
        execution[model] = dict(valid=child.returncode == 0 and failure is None, returncode=child.returncode,
                                failure=failure, observed_peak=peak, seconds=time.monotonic()-started)
        write(out/'execution.json', execution)
        print(model, execution[model], flush=True)


def main():
    """CLI phases keep data selection, forecasting and evaluation separated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out, args.cache)
    elif args.command == 'run': run(args.out)
    else: worker(args.out, args.model)


if __name__ == '__main__':
    main()
