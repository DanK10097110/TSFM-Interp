"""One bounded offline native worker at a time, with future-file access denied.

Uses existing Native, blocks/capture/patched helpers without modifying them.
Raw full-token FP32 states retain identity and geometry for later analysis;
no token alignment or cross-model representation claims are made here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from experiments.feature_search.persistence_protocol import ENV, exact, finite, read, sha, write


def target_guard(root):
    """Return a Python audit hook denying all opens under the evaluation tree."""
    root = Path(root).resolve()
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path == root or root in path.parents:
                raise PermissionError('Native worker cannot open evaluation targets or labels')
    return guard


def bound(out):
    """Verify frozen inference sources and inputs without opening evaluation files."""
    p = read(out/'inference/protocol.json')
    for path, digest in p['sources'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: '+path)
    for name, digest in p['hashes'].items():
        if sha(out/'inference'/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    return p


def worker(out, model):
    """Infer only deduplicated histories/residuals and enforce exact replay gates."""
    start = time.monotonic()
    sys.addaudithook(target_guard(out/'evaluation'))
    p = bound(out)
    for name, expected in ENV.items():
        if os.environ.get(name) != expected:
            raise ValueError('Incorrect environment: '+name)
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import blocks, capture, patched
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    item = p['checkpoints'][model]
    for name, digest in item['weights'].items():
        if sha(Path(item['path'])/name) != digest:
            raise ValueError('Cached weights changed')
    data = np.load(out/'inference/inputs.npz')
    x, residual = data['history'], data['residual']
    native = Native(model, item['path']); sites = blocks(native)
    native_predictions, residual_predictions, saved = [], [], {}
    checks = dict(repeat=0, self_patch=0, duplicate=0, capture_vs_plain=0,
                  residual_repeat=0, target_access_denied=False)
    try:
        (out/'evaluation/targets.npz').open('rb')
    except PermissionError:
        checks['target_access_denied'] = True
    if not checks['target_access_denied']:
        raise ValueError('Target isolation guard failed')
    for i in range(len(x)):
        query = x[i:i+1]
        forecast, states = capture(native, query, sites)
        repeat = native.predict(query)[0]
        exact(forecast, repeat, 'capture/repeat native'); checks['repeat'] += 1
        checks['capture_vs_plain'] += 1
        for cut, module in sites.items():
            exact(patched(native, query, module, states[cut]), forecast, 'self patch')
            checks['self_patch'] += 1
        twin = native.predict(np.repeat(query, 2, axis=0))[0]
        exact(twin[0], twin[1], 'identical histories in duplicate batch')
        checks['duplicate'] += 1
        resforecast, resstates = capture(native, residual[i:i+1], sites)
        exact(resforecast, native.predict(residual[i:i+1])[0], 'residual repeat')
        checks['residual_repeat'] += 1
        for cut in sites:
            saved[f'q{i}_full_block{cut}'] = finite(states[cut][0])
            saved[f'q{i}_residual_block{cut}'] = finite(resstates[cut][0])
        native_predictions.append(forecast[0]); residual_predictions.append(resforecast[0])
        if time.monotonic()-start > p['max_seconds']:
            raise RuntimeError('Worker time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['max_rss_bytes']:
            raise RuntimeError('Worker RSS cap')
        print(model, i+1, '/', len(x), 'seconds', time.monotonic()-start, flush=True)
    forecast = dict(native=finite(native_predictions),
                    tones_native_residual=finite(data['tones']+residual_predictions),
                    tones_zero_residual=finite(data['tones']), seasonal=finite(data['seasonal']))
    plans = read(out/'inference/plans.json')
    for i, info in enumerate(plans):
        if not info['accepted']:
            exact(forecast['native'][i], forecast['tones_native_residual'][i], 'abstention')
    np.savez_compressed(out/f'{model}_forecasts.npz', **forecast)
    np.savez_compressed(out/f'{model}_states.npz', **saved)
    write(out/f'{model}_worker.json', dict(
        protocol_sha256=sha(out/'inference/protocol.json'), gates=checks,
        seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        layers=list(sites), state_shapes={k: list(v.shape) for k, v in saved.items()},
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
        files={name: sha(out/name) for name in [f'{model}_forecasts.npz', f'{model}_states.npz']}))


def run(out):
    """Monitor serial subprocesses; kill before600 seconds or upon RSS excess."""
    import fcntl
    p = bound(out)
    if any((out/f'{m}.log').exists() for m in ['bolt', 'timesfm']):
        raise ValueError('Refusing to reuse a native run directory')
    lock = (out/'native_worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    execution = {}
    for model in ['bolt', 'timesfm']:
        start = time.monotonic(); peak = 0; reason = None
        env = dict(os.environ, **ENV)
        with (out/f'{model}.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.persistence_study',
                                     'worker', '--out', str(out), '--model', model],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    try:
                        lines = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8').splitlines()
                        peak = max(peak, int(next(l.split()[1] for l in lines if l.startswith('VmRSS:')))*1024)
                    except (FileNotFoundError, StopIteration):
                        pass
                    if peak > p['max_rss_bytes'] or time.monotonic()-start >= p['max_seconds']-.5:
                        reason = 'RSS cap' if peak > p['max_rss_bytes'] else 'Time cap'
                        proc.kill(); proc.wait(); break
                    time.sleep(.1)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()
        execution[model] = dict(valid=proc.returncode == 0 and reason is None,
                                returncode=proc.returncode, reason=reason,
                                seconds=time.monotonic()-start, peak_rss_bytes=peak)
        write(out/'execution.json', execution)
        print(model, execution[model], flush=True)
        if not execution[model]['valid']:
            raise RuntimeError('Bounded native worker failed; see '+str(out/f'{model}.log'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=['bolt', 'timesfm'])
    args = parser.parse_args()
    if args.command == 'run':
        run(args.out)
    else:
        worker(args.out, args.model)
