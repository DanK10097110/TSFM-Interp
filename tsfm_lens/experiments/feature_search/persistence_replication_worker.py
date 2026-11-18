"""Slot-gated, singleton native replication with a hard15-minute model budget."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

from experiments.feature_search.persistence_protocol import ENV, exact, finite, read, sha, write
from experiments.feature_search.persistence_replication_protocol import binding
from experiments.feature_search.persistence_singleton import singleton_class
from experiments.feature_search.persistence_study import target_guard


def require_grant(out, model):
    """Direct worker calls also refuse to bypass the coordinator slot hold."""
    path = out/f'{model}_slot_grant.json'
    if not path.exists():
        raise PermissionError('HOLD: explicit coordinator slot grant required')
    grant = read(path)
    if not grant.get('coordinator_grant') or grant.get('protocol_sha256') != sha(out/'protocol.json'):
        raise PermissionError('Invalid or stale coordinator slot grant')
    return grant


def worker(out, model):
    """Forecast the exact deduplicated input bank; sentinels calibrate each run."""
    require_grant(out, model)
    started = time.monotonic()
    sys.addaudithook(target_guard(out/'evaluation'))
    p = binding(out)
    for name, value in ENV.items():
        if os.environ.get(name) != value:
            raise ValueError('Incorrect environment: '+name)
    import torch
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import blocks, capture, patched
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    checkpoint = p['checkpoints'][model]
    for name, digest in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/name) != digest:
            raise ValueError('Cached weight changed')
    native = singleton_class(Native)(model, checkpoint['path'])
    data = np.load(out/'inference/inputs.npz')
    bank = data['bank']; plans = read(out/'inference/plans.json')
    predictions = []
    for i, value in enumerate(bank):
        prediction = finite(native.predict(value[None])[0])
        if prediction.shape != (1, 32):
            raise ValueError('Native output shape')
        predictions.append(prediction[0])
        if i % 100 == 0:
            print(model, i, '/', len(bank), 'seconds', time.monotonic()-started, flush=True)
        if time.monotonic()-started > 895 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > p['max_rss_bytes']:
            raise RuntimeError('Model resource cap')
    predictions = finite(predictions)
    sites = blocks(native); states = {}; gates = dict(repeat=0, self_patch=0, duplicate=0, residual_repeat=0, target_denied=False)
    for q in p['sentinels']:
        plan = plans[q]; query = bank[plan['native']][None]
        pred, hidden = capture(native, query, sites)
        exact(pred[0], predictions[plan['native']], 'sentinel capture/plain repeat'); gates['repeat'] += 1
        for cut, module in sites.items():
            exact(patched(native, query, module, hidden[cut]), pred, 'sentinel self patch')
            states[f'q{q}_full_block{cut}'] = finite(hidden[cut][0]); gates['self_patch'] += 1
        twin = native.predict(np.repeat(query, 2, axis=0))[0]
        exact(twin[0], twin[1], 'independent singleton duplicate'); exact(twin[0], pred[0], 'duplicate/reference')
        gates['duplicate'] += 1
        for role in ['residual', 'recent_residual']:
            exact(native.predict(bank[plan[role]][None])[0][0], predictions[plan[role]], role+' repeat')
            gates['residual_repeat'] += 1
    try:
        (out/'evaluation/test/targets.npz').open('rb')
    except PermissionError:
        gates['target_denied'] = True
    if not gates['target_denied']:
        raise ValueError('Target isolation failed')
    forecast = dict(native=predictions[[p['native'] for p in plans]],
                    hybrid=data['tones']+predictions[[p['residual'] for p in plans]],
                    tones_zero=data['tones'], seasonal=data['seasonal'],
                    recent_hybrid=data['recent_tones']+predictions[[p['recent_residual'] for p in plans]],
                    recent_zero=data['recent_tones'])
    for q, plan in enumerate(plans):
        if not plan['accepted']:
            exact(forecast['hybrid'][q], forecast['native'][q], 'full abstention')
            exact(forecast['recent_hybrid'][q], forecast['native'][q], 'recent abstention')
    for value in forecast.values():
        finite(value)
    np.savez_compressed(out/f'{model}_forecasts.npz', bank_predictions=predictions, **forecast)
    np.savez_compressed(out/f'{model}_states.npz', **states)
    write(out/f'{model}_worker.json', dict(valid=True, gates=gates, layers=list(sites),
          seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
          protocol_sha256=sha(out/'inference/protocol.json'), files={f: sha(out/f) for f in [f'{model}_forecasts.npz', f'{model}_states.npz']}))


def run(out, model, slot_grant):
    """An explicit later slot grant is required before spawning any native worker."""
    if not slot_grant:
        raise PermissionError('HOLD: native launch not authorized until coordinator grants a slot')
    import fcntl
    p = binding(out)
    lock = (out/'native_worker.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/f'{model}.log').exists():
        raise ValueError('Refuse native attempt overwrite')
    write(out/f'{model}_slot_grant.json', dict(coordinator_grant=slot_grant, protocol_sha256=sha(out/'protocol.json')))
    started = time.monotonic(); peak = 0; failure = None
    with (out/f'{model}.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.persistence_replication_worker',
                                 'worker', '--out', str(out), '--model', model],
                                env=dict(os.environ, **ENV), stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                try:
                    lines = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8').splitlines()
                    peak = max(peak, int(next(l.split()[1] for l in lines if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError, StopIteration):
                    pass
                if peak > p['max_rss_bytes'] or time.monotonic()-started >= 899.5:
                    failure = 'RSS cap' if peak > p['max_rss_bytes'] else 'Time cap'
                    proc.kill(); proc.wait(); break
                time.sleep(.1)
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    result = dict(valid=proc.returncode == 0 and failure is None, returncode=proc.returncode,
                  failure=failure, seconds=time.monotonic()-started, peak_rss_bytes=peak)
    write(out/f'{model}_execution.json', result)
    knowledge = read(out/'knowledge.json')
    knowledge.setdefault('native_execution', {})[model] = result
    knowledge['next_decision'] = 'Worker valid; select using FIT only after both workers finish' if result['valid'] else 'Failed instrumentation; preserve attempt and diagnose before further inference'
    write(out/'knowledge.json', knowledge)
    print(result, flush=True)
    if not result['valid']:
        raise RuntimeError('Native worker failed; partial results invalid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=['bolt', 'timesfm'], required=True)
    parser.add_argument('--slot-grant')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.out, args.model, args.slot_grant)
    else:
        worker(args.out, args.model)
