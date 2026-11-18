"""Staged coherence studies: interference, fixed-normalizer controls and fresh real sources.

Native workers see only past-only input banks and normalization anchors. Futures
are separate. Small-model coverage training and selectors have separate drivers.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

DOMAINS = ('monash_electricity_hourly', 'm4_hourly', 'monash_pedestrian_counts', 'monash_traffic')


def read(path):
    """Read UTF-8 JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Write finite records."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    """Stream exact bytes."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def parts(x, mode='fft'):
    """Four lossless physical-frequency bands; alternate boundary or endpoint bridge."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n not in (128, 256): raise ValueError('Only declared context lengths supported')
    trend = np.linspace(x[0], x[-1], n) if mode == 'bridge' else np.full(n, x.mean())
    spectral = np.fft.rfft(x-trend)
    edges = [0, (6 if mode == 'shift' else 5)*n//128, 13*n//128, 33*n//128, n//2+1]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        selected = np.zeros_like(spectral); selected[a:b] = spectral[a:b]
        out.append(np.fft.irfft(selected, n=n))
    out = np.asarray(out); out[0] += trend
    tiny = np.linalg.norm(out, axis=1) <= 1e-12*max(np.linalg.norm(x), 1e-12)
    out[tiny] = 0; out[0] += x-out.sum(0)
    return out


class Bank:
    """Deduplicate equal FP32 contexts, retaining physical length."""
    def __init__(self): self.values = []; self.lookup = {}

    def add(self, x):
        """Return global bank ID."""
        x = np.asarray(x, dtype=np.float32).copy(); x[x == 0] = 0
        if x.shape not in [(128,), (256,)] or not np.isfinite(x).all(): raise ValueError('Bad context')
        key = x.tobytes()
        if key not in self.lookup:
            self.lookup[key] = len(self.values); self.values.append(x)
        return self.lookup[key]

    def save(self, path):
        """Length groups enable native batching without padding or truncation."""
        arrays = {}
        for n in (128, 256):
            ids = [i for i, x in enumerate(self.values) if len(x) == n]
            arrays[f'ids{n}'] = np.asarray(ids, dtype=int)
            arrays[f'inputs{n}'] = np.stack([self.values[i] for i in ids]) if ids else np.zeros((0, n), np.float32)
        np.savez_compressed(path, **arrays)


def load_bank(path):
    """Reconstruct the indexed list without touching future arrays."""
    with np.load(path) as f:
        values = [None]*(len(f['ids128'])+len(f['ids256']))
        for n in (128, 256):
            for i, x in zip(f[f'ids{n}'], f[f'inputs{n}']): values[int(i)] = x
    return values


def real_data(cache, offset, count):
    """Past eligibility and fixed source ordering, preserving earlier pilot identities."""
    import pyarrow as pa
    sources = []
    for domain in DOMAINS:
        paths = sorted((cache/'datasets/autogluon___chronos_datasets'/domain).rglob('*train.arrow'))
        if len(paths) != 1: raise ValueError('Need one cached hourly source file')
        with pa.memory_map(str(paths[0]), 'r') as stream: table = pa.ipc.open_stream(stream).read_all()
        eligible = []
        for i in np.random.default_rng(492306).permutation(len(table)):
            row = table.slice(int(i), 1).to_pylist()[0]
            value = np.asarray(row['target'], dtype=float)
            if len(value) < 288: continue
            value = value[-288:]; scale = float(np.std(value[128:256]))
            stamps = np.asarray(row['timestamp'][-288:], dtype='datetime64[ms]')
            if not np.isfinite(value[:256]).all() or scale <= 1e-8 or not np.all(np.diff(stamps) == np.timedelta64(1, 'h')): continue
            eligible.append((row, value, scale))
            if len(eligible) == offset+count: break
        if len(eligible) < offset+count: raise ValueError(f'Only {len(eligible)} eligible sources in {domain}')
        for j, (row, value, scale) in enumerate(eligible[offset:]):
            if not np.isfinite(value).all(): raise ValueError('Selected target missing; refusing outcome-based replacement')
            center = float(value[128:256].mean())
            sources.append((dict(source=domain+':'+row['id'], domain=domain, rank=offset+j,
                                 split='fit' if j < count//2 else 'test', source_scale=scale,
                                 observed_end=str(row['timestamp'][-1]), data_sha256=hashlib.sha256(value.tobytes()).hexdigest()),
                            (value-center)/scale))
    return sources


def prepare(out, cache, stage):
    """Freeze row membership, all counterfactual inputs and statistical family before inference."""
    from experiments.feature_search import real_background_poc, homogeneous_normalizer
    out.mkdir(parents=True, exist_ok=False)
    bank = Bank(); rows = []; targets = []; requests = []
    nseed = 8 if stage == 'pilot' else 64
    seedbase = 4931000 if stage == 'pilot' else 4932000
    def add_row(meta, history, future, modes):
        row = dict(meta, root=bank.add(history), zero=bank.add(np.zeros(len(history))), length=len(history), methods={})
        for mode in modes: row['methods'][mode] = [bank.add(p) for p in parts(history, mode)]
        row['index'] = len(rows); rows.append(row); targets.append(np.asarray(future)); return row
    for j, seed in enumerate(range(seedbase, seedbase+nseed)):
        rng = np.random.default_rng(seed); t = np.arange(-256, 32)
        noise = rng.normal(0, .05, len(t)); phi, psi = rng.uniform(-np.pi, np.pi, 2)
        frequencies = [(a, b) for a in [3., 5., 9.] for b in [a+1, 2*a, 3.4*a]] if stage == 'pilot' else [(5., 6.), (5., 17.), (5.35, 6.35), (5.35, 17.35)]
        for target_freq, distractor_freq in frequencies:
            for amplitude in [.5, 8.]:
                background = amplitude*np.sin(2*np.pi*distractor_freq*t/128+psi)+noise
                target = .5*np.sin(2*np.pi*target_freq*t/128+phi)
                lengths = [128, 256] if target_freq == 5. and distractor_freq in (6., 17.) and j < (8 if stage == 'pilot' else 16) else [128]
                for length in lengths:
                    group = []
                    for sign in [0, -1, 1]:
                        history = (background+sign*target)[256-length:256]
                        modes = ['fft'] if length == 128 else []
                        if target_freq == 5. and distractor_freq == 6. and length == 128: modes.append('shift')
                        if target_freq == 5.35 and length == 128: modes.append('bridge')
                        row = add_row(dict(source=f'syn:{seed}', domain='synthetic', kind='interference', split='test',
                                           target_frequency=target_freq, distractor_frequency=distractor_freq,
                                           amplitude=amplitude, sign=sign, seed=seed, target_phase=phi, scale=1.),
                                      history, (background+sign*target)[256:], modes)
                        group.append(row)
                    if target_freq == 5. and distractor_freq in (6., 17.) and amplitude == 8. and length == 128 and j < (8 if stage == 'pilot' else 16):
                        for row in group[1:]:
                            row['locked'] = {}
                            for policy in ['background', 'shared6']:
                                row['locked'][policy] = len(requests)
                                requests.append(dict(query=row['root'], anchor=group[0]['root'], policy=policy))
    offset, count = (16, 8) if stage == 'pilot' else (24, 32)
    for source, value in real_data(cache, offset, count):
        x = value[128:256]; y = value[256:]; t = np.arange(-128, 32)
        history = value[96:224]
        loc, scale = float(history.mean()), float(history.std())
        hindcast = bank.add((history-loc)/scale)
        base = add_row(dict(source, kind='real', sign=0, target_frequency=5., target_phase=.3, scale=1.), x, y, ['fft', 'bridge'])
        base['hindcast'] = dict(bank_id=hindcast, center=loc, scale=scale, target=value[224:256].tolist())
        for frequency in [5., 128/24]:
            target = .5*np.sin(2*np.pi*frequency*t/128+.3)
            for sign in [-1, 1]:
                add_row(dict(source, kind='real', sign=sign, target_frequency=frequency, target_phase=.3, scale=1.),
                        x+sign*target[:128], y+sign*target[128:], ['fft', 'bridge'])
    if len(bank.values) > (8000 if stage == 'pilot' else 22000): raise ValueError('Frozen bank size cap exceeded')
    bank.save(out/'bank.npz'); np.savez_compressed(out/'targets.npz', future=np.asarray(targets))
    write(out/'rows.json', rows); write(out/'requests.json', requests)
    old = Path(__file__).resolve().parents[2]/'runs/fourier_poc4929000/protocol.json'
    checkpoints = read(old)['checkpoints']
    sources = [Path(__file__), Path(real_background_poc.__file__), Path(homogeneous_normalizer.__file__)]
    protocol = dict(stage=stage, source_count=nseed+4*count, synthetic_seeds=list(range(seedbase, seedbase+nseed)),
                    real_count_per_domain=count, real_offset=offset, node_count=len(bank.values), rows=len(rows),
                    horizon=32, checkpoints=checkpoints, cpu_threads=1, worker_cap=3*2**30,
                    worker_seconds=900 if stage == 'pilot' else 1800,
                    primary_family='12 one-sided exact sign tests, Holm: per model, far on-grid high-vs-low response harm; high-amplitude far FFT response gain; electricity/pedestrian test FFT response gain and clean forecast harm',
                    primary_effect_gate=.1, inference='Directional majority tests, not population mean tests; empirical mean effect reported separately. Bootstrap sources within domain. No domain-pooled generalization.',
                    selectors='Fit only first half real source IDs; fixed ridge/RBF input-only vs input+diagnostic, train-best constant and past32 hindcast winner; untouched second half evaluation',
                    controls='FFT original vs shifted bin6 boundary; endpoint bridge; high-band forecast suppression; native vs anchored or shared6 normalization;128 vs256 physical history',
                    normalization_scope='Background stats removes paired-statistic changes; shared6 is off-native-manifold fixed mean0/SD6. Neither isolates training or all dynamic-range effects.',
                    provenance='Cached real pretraining overlap unknown; sources disjoint from previous rounds; source and temporal-domain familiarity not excluded',
                    source_hashes={str(p.resolve()): sha(p) for p in sources},
                    hashes={name: sha(out/name) for name in ['bank.npz', 'targets.npz', 'rows.json', 'requests.json']})
    write(out/'protocol.json', protocol)
    print(json.dumps(dict(stage=stage, nodes=len(bank.values), rows=len(rows), locked_queries=len(requests))), flush=True)


@contextlib.contextmanager
def normalization(native, anchor, policy):
    """Lock the actual native normalizer and its matching inverse, restoring on exit."""
    import torch
    if native.name == 'bolt':
        module = native.model.instance_norm
        original = module.forward
        with torch.no_grad(): _, reference = original(torch.as_tensor(anchor, dtype=torch.float32))
        if policy == 'shared6': reference = (torch.zeros_like(reference[0]), torch.full_like(reference[1], 6.))
        def fixed(x, loc_scale=None): return original(x, reference)
        module.forward = fixed
        try: yield
        finally: module.forward = original
    else:
        module = sys.modules[type(native.model).__module__]
        original = module.revin; captured = []
        def trace(x, mu, sigma, reverse=False):
            if not reverse: captured.append((mu.detach().clone(), sigma.detach().clone()))
            return original(x, mu, sigma, reverse=reverse)
        module.revin = trace
        try: native.predict(anchor)
        finally: module.revin = original
        if len(captured) != 1: raise ValueError('Expected one prefill normalization for <=4 contexts')
        mu, sigma = captured[0]
        if policy == 'shared6': mu, sigma = torch.zeros_like(mu), torch.full_like(sigma, 6.)
        def fixed(x, unused_mu, unused_sigma, reverse=False): return original(x, mu, sigma, reverse=reverse)
        module.revin = fixed
        try: yield
        finally: module.revin = original


def worker(out, model):
    """Forecast bank and declared normalization interventions, without reading futures."""
    from experiments.feature_search.real_background_poc import Native
    p = read(out/'protocol.json'); started = time.monotonic()
    for path, expected in p['source_hashes'].items():
        if sha(path) != expected: raise ValueError('Frozen source changed')
    for file in ['bank.npz', 'requests.json']:
        if sha(out/file) != p['hashes'][file]: raise ValueError('Bound input changed')
    checkpoint = p['checkpoints'][model]
    for file, expected in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/file) != expected: raise ValueError('Checkpoint changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only worker')
    native = Native(model, checkpoint['path']); values = load_bank(out/'bank.npz')
    forecasts = np.zeros((len(values), 32), dtype=np.float32)
    for length in [128, 256]:
        ids = [i for i, x in enumerate(values) if len(x) == length]
        for start in range(0, len(ids), 64):
            selected = ids[start:start+64]
            forecasts[selected] = native.predict(np.stack([values[i] for i in selected]))[0]
            if start % 1024 == 0: print(model, length, start, '/', len(ids), time.monotonic()-started, flush=True)
    requests = read(out/'requests.json'); locked = np.zeros((len(requests), 32)); locked_base = np.zeros_like(locked); identities = []
    for start in range(0, len(requests), 4):
        for policy in ['background', 'shared6']:
            ids = [i for i in range(start, min(start+4, len(requests))) if requests[i]['policy'] == policy]
            if not ids: continue
            anchor = np.stack([values[requests[i]['anchor']] for i in ids]); query = np.stack([values[requests[i]['query']] for i in ids])
            if policy == 'background':
                expected = native.predict(anchor)[0]
                with normalization(native, anchor, policy): actual = native.predict(anchor)[0]
                identity = float(np.max(abs(expected-actual))); identities.append(identity)
                if identity != 0: raise ValueError(f'Normalizer self identity failed {identity}')
            with normalization(native, anchor, policy):
                locked_base[ids] = native.predict(anchor)[0]
                locked[ids] = native.predict(query)[0]
    repeat = native.predict(np.stack(values[:4]))[0]
    if not np.array_equal(repeat, forecasts[:4]) or not np.isfinite(forecasts).all() or not np.isfinite(locked).all(): raise ValueError('Finite/repeat gate failed')
    np.savez_compressed(out/f'{model}.npz', forecasts=forecasts, locked=locked, locked_base=locked_base)
    write(out/f'{model}_worker.json', dict(seconds=time.monotonic()-started, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                          normalizer_identity_max=max(identities, default=0), repeat_exact=True, sha256=sha(out/f'{model}.npz')))


def run(out):
    """Serial monitored workers; over-budget outputs cannot qualify for inference."""
    p = read(out/'protocol.json'); execution = {}
    for model in ['bolt', 'timesfm']:
        if (out/f'{model}.npz').exists(): raise ValueError('Refuse overwrite')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        started = time.monotonic(); peak = 0; reason = None
        with (out/f'{model}.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.coherence_study', 'worker', '--out', str(out), '--model', model], env=env, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:
                    s = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8')
                    peak = max(peak, int(next(l.split()[1] for l in s.splitlines() if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError, StopIteration): pass
                if peak > p['worker_cap'] or time.monotonic()-started > p['worker_seconds']:
                    reason = 'Frozen resource cap exceeded'; proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    break
                time.sleep(.2)
        workerfile = out/f'{model}_worker.json'
        if workerfile.exists() and read(workerfile)['peak_rss'] > p['worker_cap']: reason = 'HWM exceeds cap'
        execution[model] = dict(valid=proc.returncode == 0 and reason is None, returncode=proc.returncode, reason=reason,
                                seconds=time.monotonic()-started, monitored_peak=peak)
        write(out/'execution.json', execution); print(model, execution[model], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--stage', choices=['pilot', 'replication'], default='pilot')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--model', choices=['bolt', 'timesfm'])
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.out, a.cache, a.stage)
    elif a.command == 'run': run(a.out)
    else: worker(a.out, a.model)
