"""Frozen slot-2 response-surface pilot following selective PM31--34.

Eight backgrounds, fixed source phases, continuous off-grid frequency paths and
amplitude paths. Saves every FP32 forecast and full-token first/quarter/half state.
Workers consume past inputs only; all arithmetic is rerunnable from saved arrays.
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
from experiments.feature_search.selective_study import blocks, capture, patched

CENTERS = (128/24, 128/19.2)
AMPLITUDES = (.1, .5, 2.)
OFFSETS = (-.12, -.06, -.03, 0., .03, .06, .12)


def tone(frequency, amplitude, phase, time_points):
    """Continuous cycles per128 samples, with phase anchored at forecast origin t=0."""
    return amplitude*np.sin(2*np.pi*frequency*np.asarray(time_points)/128+phase)


def make_design(backgrounds):
    """Deduplicate only within a source so anchors and source identities never mix."""
    inputs, rows, paths, interactions, signals = [], [], [], [], []
    for source_id, (meta, raw) in enumerate(backgrounds):
        x = np.asarray(raw, np.float32)
        scale = float(np.std(x.astype(float)))
        if x.shape != (128,) or not np.isfinite(x).all() or scale <= 1e-8:
            raise ValueError('Invalid background')
        phase = float(np.random.default_rng(4951000+source_id).uniform(-np.pi, np.pi))
        lookup = {}

        def add(value, kind, **details):
            """Bind exact context bytes and provenance to one global row."""
            value = np.asarray(value, np.float32)
            key = value.tobytes()
            if key not in lookup:
                index = len(inputs); lookup[key] = index; inputs.append(value)
                rows.append(dict(meta, index=index, source_id=source_id, kind=kind,
                                 phase=phase, scale=scale, **details))
            return lookup[key]

        base = add(x, 'background')

        def signal(f, a):
            """Amplitudes are in the fixed background's observed standard-deviation units."""
            return tone(f, a*scale, phase, np.arange(-128, 0))

        def node(f, a):
            """Register a physical tone perturbation with common phase and history."""
            return add(x.astype(float)+signal(f, a), 'tone', frequency=f, amplitude=a, base=base)

        for center in CENTERS:
            for amplitude in AMPLITUDES:
                sweep = {offset: node(center+offset, amplitude) for offset in OFFSETS}
                for offset, index in sweep.items():
                    signals.append(dict(index=index, base=base, source=meta['source'],
                                        domain=meta['domain'], center=center, offset=offset,
                                        frequency=center+offset, amplitude=amplitude, phase=phase, scale=scale))
                for radius in (.03, .06, .12):
                    minus, zero, plus = sweep[-radius], sweep[0.], sweep[radius]
                    mid = add((inputs[minus].astype(float)+inputs[plus])/2, 'chord', base=base)
                    paths.append(dict(source=meta['source'], domain=meta['domain'], base=base,
                                      axis='frequency', center=center, amplitude=amplitude,
                                      step=radius, relative_step=radius, minus=minus, zero=zero,
                                      plus=plus, midpoint=mid, scale=scale))
                for fraction in (.01, .02):
                    minus, zero, plus = node(center, amplitude*(1-fraction)), sweep[0.], node(center, amplitude*(1+fraction))
                    mid = add((inputs[minus].astype(float)+inputs[plus])/2, 'chord', base=base)
                    paths.append(dict(source=meta['source'], domain=meta['domain'], base=base,
                                      axis='amplitude', center=center, amplitude=amplitude,
                                      step=amplitude*fraction, relative_step=fraction, minus=minus,
                                      zero=zero, plus=plus, midpoint=mid, scale=scale))
        for a in AMPLITUDES:
            for b in AMPLITUDES:
                ia, ib = node(CENTERS[0], a), node(CENTERS[1], b)
                ab = add(x.astype(float)+signal(CENTERS[0], a)+signal(CENTERS[1], b),
                         'mixture', amplitude_a=a, amplitude_b=b, base=base)
                interactions.append(dict(source=meta['source'], domain=meta['domain'], base=base,
                                         a=ia, b=ib, ab=ab, amplitude_a=a, amplitude_b=b, scale=scale))
    return np.stack(inputs), rows, paths, interactions, signals


def prepare(out, prior):
    """Freeze all sources, source groups, phases, numerical paths and analysis before inference."""
    from experiments.feature_search import (response_surface_metrics, response_surface_analysis,
        selective_study, selective_tree, coherence_study, real_background_poc, homogeneous_normalizer)
    prior_rows = read(prior/'rows.json')
    backgrounds = []
    for j, (f, amplitude) in enumerate([(17.35, .5), (17.35, 8.), (6.35, .5), (6.35, 8.)]):
        seed = 4951000+j
        rng = np.random.default_rng(seed); phases = rng.uniform(-np.pi, np.pi, 2)
        t = np.arange(-128, 0)
        x = tone(5.35, .5, phases[0], t)+tone(f, amplitude, phases[1], t)+rng.normal(0, .05, 128)
        x = (x-x.mean())/x.std()
        backgrounds.append((dict(source=f'syn:{seed}', domain='synthetic', split='instrumentation',
                                 seed=seed, distractor_frequency=f, distractor_amplitude=amplitude), x))
    with np.load(prior/'inputs.npz') as previous:
        for domain in coherence_study.DOMAINS:
            source = next(r for r in prior_rows if r['domain'] == domain and r['split'] == 'fit' and r['case'] == 'real')
            backgrounds.append((dict(source=source['source'], domain=domain, split='instrumentation',
                                     original_split='fit', original_index=source['index'], original_rank=source['rank']),
                                previous['history'][source['index']].copy()))
    inputs, rows, paths, interactions, signals = make_design(backgrounds)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out/'inputs.npy', inputs)
    for name, value in [('rows', rows), ('paths', paths), ('interactions', interactions), ('signals', signals)]:
        write(out/f'{name}.json', value)
    modules = [response_surface_metrics, response_surface_analysis, selective_study, selective_tree,
               coherence_study, real_background_poc, homogeneous_normalizer]
    files = [Path(__file__)] + [Path(m.__file__) for m in modules]
    files += [Path(__file__).resolve().parents[2]/'tsfm_lens/extraction/hooks.py']
    files += sorted((Path(__file__).resolve().parents[2]/'tests').glob('test_response_surface_*.py'))
    prior_files = ['protocol.json', 'rows.json', 'inputs.npz']
    protocol = dict(stage='instrumentation_pilot', run=out.name, seed=4951000, background_count=8,
                    sources=[m for m, _ in backgrounds], nodes=len(inputs), horizon=32, context=128,
                    centers=list(CENTERS), offsets=list(OFFSETS), amplitudes=list(AMPLITUDES),
                    phase='One seeded phase per source, shared across every frequency/amplitude and both tones; t=0 is forecast origin',
                    normalization='native versus background-locked actual normalizer and matching inverse; anchor is same background for entire surface',
                    frequency='Continuous analytic sinusoids, no FFT bin snapping; centers 24h and19.2h; frequency steps .03/.06/.12 cycles per128',
                    amplitude='Peak amplitudes .1/.5/2 times background SD; local steps +/-1% and2%',
                    local='FP64 centered parameter derivatives; chord-midpoint model curvature removes frequency-path bend; sensitivities divide actual RMS input distances/background SD',
                    interaction='f(x+a+b)-f(x+a)-f(x+b)+f(x); RMS relative to each individual effect and sum; normalize mixed effect by input-distance product',
                    decodability='Fixed ridge alpha=.01, full-token hidden differences versus raw-input differences, joint fixed-frequency sine/cos coefficients; leave-one-source-out, train-only scaling and mean/permuted-label controls; exploratory only',
                    inference='Descriptive source-grouped summaries only; 4 synthetic seed groups and1 real FIT source per domain. No population, confirmation or general interpretability verdict.',
                    provenance='Real FIT sources explicitly reused from selective_replication4942000 for instrumentation; pretraining overlap unknown. No real futures loaded.',
                    limitations='One phase per background confounds phase with context; no phase-generalization claim. Locked normalization is a counterfactual control, potentially off native manifold at high amplitude. Hidden curvature is not causal semantics.',
                    checkpoints=read(prior/'protocol.json')['checkpoints'], worker_cap=3*2**30,
                    seconds_per_model=600, cpu_threads=1, slot=2, execution='serial native workers',
                    source_hashes={str(p.resolve()): sha(p) for p in files},
                    prior_hashes={str((prior/f).resolve()): sha(prior/f) for f in prior_files},
                    hashes={f: sha(out/f) for f in ['inputs.npy', 'rows.json', 'paths.json', 'interactions.json', 'signals.json']})
    write(out/'protocol.json', protocol)
    print(dict(run=str(out.resolve()), nodes=len(inputs), paths=len(paths), interactions=len(interactions)), flush=True)


def validate(out, include_weights=None):
    """Refuse altered protocol-bound code, inputs or checkpoints."""
    p = read(out/'protocol.json')
    for filename, expected in p['source_hashes'].items():
        if sha(filename) != expected: raise ValueError(f'Frozen source changed: {filename}')
    for filename, expected in p['hashes'].items():
        if sha(out/filename) != expected: raise ValueError(f'Frozen input changed: {filename}')
    if include_weights:
        checkpoint = p['checkpoints'][include_weights]
        for filename, expected in checkpoint['weights'].items():
            if sha(Path(checkpoint['path'])/filename) != expected: raise ValueError('Checkpoint changed')
    return p


def worker(out, model):
    """Native prediction and full-token capture with exact identity gates on every background."""
    from experiments.feature_search.real_background_poc import Native
    started = time.monotonic(); p = validate(out, model)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU only')
    native = Native(model, p['checkpoints'][model]['path']); sites = blocks(native)
    inputs = np.load(out/'inputs.npy', mmap_mode='r'); rows = read(out/'rows.json')
    gates = dict(repeat_forecast=0., repeat_hidden=0., normalizer_forecast=0., normalizer_hidden=0., self_patch=0.)
    artifacts = []
    forecasts = {}; hidden = {}
    for policy in ('native', 'locked'):
        name = f'{model}_{policy}_forecast.npy'; artifacts.append(name)
        forecasts[policy] = np.lib.format.open_memmap(out/name, mode='w+', dtype=np.float32, shape=(len(rows), 32))
    for source_id in range(8):
        ids = [r['index'] for r in rows if r['source_id'] == source_id]
        base = next(r['index'] for r in rows if r['source_id'] == source_id and r['kind'] == 'background')
        x = inputs[base:base+1].copy()
        expected, hx = capture(native, x, sites)
        repeated, hr = capture(native, x, sites)
        with normalization(native, x, 'background'): locked, hl = capture(native, x, sites)
        gates['repeat_forecast'] = max(gates['repeat_forecast'], float(abs(expected-repeated).max()))
        gates['normalizer_forecast'] = max(gates['normalizer_forecast'], float(abs(expected-locked).max()))
        for cut, module in sites.items():
            gates['repeat_hidden'] = max(gates['repeat_hidden'], float(abs(hx[cut]-hr[cut]).max()))
            gates['normalizer_hidden'] = max(gates['normalizer_hidden'], float(abs(hx[cut]-hl[cut]).max()))
            gates['self_patch'] = max(gates['self_patch'], float(abs(patched(native, x, module, hx[cut])-expected).max()))
        if any(gates.values()): raise ValueError(f'Exact identity gate failed {gates}')
        if not hidden:
            for policy in ('native', 'locked'):
                hidden[policy] = {}
                for cut in sites:
                    name = f'{model}_{policy}_hidden_{cut}.npy'; artifacts.append(name)
                    hidden[policy][cut] = np.lib.format.open_memmap(out/name, mode='w+', dtype=np.float32,
                                                                 shape=(len(rows),)+hx[cut].shape[1:])
        for start in range(0, len(ids), 4):
            batch = ids[start:start+4]; query = inputs[batch].copy(); anchor = np.repeat(x, len(batch), axis=0)
            for policy in ('native', 'locked'):
                ctx = contextlib.nullcontext() if policy == 'native' else normalization(native, anchor, 'background')
                with ctx: pred, hs = capture(native, query, sites)
                if not np.isfinite(pred).all() or any(not np.isfinite(v).all() for v in hs.values()):
                    raise ValueError('Nonfinite native output')
                forecasts[policy][batch] = pred
                for cut in sites: hidden[policy][cut][batch] = hs[cut]
            if time.monotonic()-started > p['seconds_per_model']: raise RuntimeError('Time cap')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['worker_cap']: raise RuntimeError('RSS cap')
        print(dict(model=model, source=source_id, completed=ids[-1]+1, total=len(rows), seconds=time.monotonic()-started), flush=True)
    for array in list(forecasts.values())+[a for group in hidden.values() for a in group.values()]: array.flush()
    for policy in ('native', 'locked'):
        for source_id in range(8):
            ids = [r['index'] for r in rows if r['source_id'] == source_id][:4]
            base = next(r['index'] for r in rows if r['source_id'] == source_id and r['kind'] == 'background')
            anchor = np.repeat(inputs[base:base+1], len(ids), axis=0)
            ctx = contextlib.nullcontext() if policy == 'native' else normalization(native, anchor, 'background')
            with ctx: repeat, h = capture(native, inputs[ids].copy(), sites)
            if not np.array_equal(repeat, forecasts[policy][ids]): raise ValueError('Batch/replay forecast identity')
            for cut in sites:
                if not np.array_equal(h[cut], hidden[policy][cut][ids]): raise ValueError('Batch/replay hidden identity')
    validate(out)
    write(out/f'{model}_worker.json', dict(model=model, cuts=list(sites), gates=gates, batch_replay_exact=True,
          seconds=time.monotonic()-started, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          hashes={name: sha(out/name) for name in artifacts}))


def run(out, models):
    """Serial monitored native processes occupying only slot2, with a nonblocking slot lock."""
    import fcntl
    p = validate(out)
    with (out/'slot2.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = read(out/'execution.json') if (out/'execution.json').exists() else {}
        for model in models:
            if any(out.glob(f'{model}_*.npy')) or model in execution: raise ValueError('Refuse overwrite native run')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            started = time.monotonic(); peak = 0; failure = None
            with (out/f'{model}.log').open('w', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.response_surface_study',
                                         'worker', '--out', str(out), '--model', model], env=env, stdout=log, stderr=subprocess.STDOUT)
                print(dict(event='native_launch', model=model, pid=proc.pid, slot=2, run=str(out.resolve())), flush=True)
                while proc.poll() is None:
                    try:
                        status = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8')
                        peak = max(peak, int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024)
                    except (FileNotFoundError, StopIteration): pass
                    if peak > p['worker_cap'] or time.monotonic()-started > p['seconds_per_model']:
                        failure = 'Frozen resource cap exceeded'; proc.kill(); proc.wait(); break
                    time.sleep(.2)
            workerfile = out/f'{model}_worker.json'
            if workerfile.exists() and read(workerfile)['peak_rss'] > p['worker_cap']: failure = 'HWM cap exceeded'
            execution[model] = dict(valid=proc.returncode == 0 and failure is None and workerfile.exists(),
                                    returncode=proc.returncode, failure=failure, seconds=time.monotonic()-started, peak_rss=peak)
            write(out/'execution.json', execution); print(dict(event='native_finish', model=model, **execution[model]), flush=True)
            if not execution[model]['valid']: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, default=Path('runs/selective_replication4942000'))
    parser.add_argument('--model', choices=['bolt', 'timesfm'])
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out, args.prior)
    elif args.command == 'run': run(args.out, [args.model] if args.model else ['bolt', 'timesfm'])
    else: worker(args.out, args.model)
