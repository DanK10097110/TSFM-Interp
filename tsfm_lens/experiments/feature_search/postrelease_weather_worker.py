"""Grant-gated weather native worker, hard bounded to300seconds and3GiB."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.persistence_protocol import ENV, exact, finite, read, sha, write
from experiments.feature_search.postrelease_weather_pilot import binding


def require_grant(out, model):
    """Refuse even direct worker invocation without an explicit bound slot grant."""
    path = out/f'{model}_slot_grant.json'
    if not path.exists():
        raise PermissionError('HOLD: coordinator slot grant required')
    grant = read(path)
    if not grant.get('coordinator_grant') or grant.get('protocol_sha256') != sha(out/'protocol.json'):
        raise PermissionError('Stale or invalid slot grant')


def predict_bank(native, data, plans, sentinel):
    """Execute all singleton inputs and reversed-order exact repeats; form forecasts."""
    bank = finite(data['bank'])
    predictions = []
    for i, value in enumerate(bank):
        pred = finite(native.predict(value[None])[0])
        if pred.shape != (1,32):
            raise ValueError('Native shape')
        predictions.append(pred[0])
        print('native',i+1,'/',len(bank),flush=True)
    predictions = finite(predictions)
    for i in reversed(range(len(bank))):
        repeated = finite(native.predict(bank[i:i+1])[0])
        exact(repeated, predictions[i:i+1], 'batch1 reversed-order repeat')
    native_pred = predictions[[p['native'] for p in plans]]
    residual_pred = predictions[[p['residual'] for p in plans]]
    forecast = dict(native=native_pred, hybrid=finite(data['tones'])+residual_pred,
                    seasonal24=finite(data['seasonal24']), tones_zero=finite(data['tones']))
    abstentions = 0
    for i, plan in enumerate(plans):
        if not plan['accepted']:
            exact(forecast['hybrid'][i], forecast['native'][i], 'abstention forecast')
            abstentions += 1
    exact(predictions[sentinel]+data['sentinel_tone'], predictions[sentinel], 'constant abstention forecast')
    for value in forecast.values():
        finite(value)
    return predictions, forecast, dict(singleton_repeat=len(bank), abstentions=abstentions,
                                     constant_abstention=1, all_finite=True)


def worker(out, model):
    """Load only cached weights after grants, bindings, environment and target denial."""
    require_grant(out, model)
    p = binding(out)
    from experiments.feature_search.persistence_study import target_guard
    for forbidden in p['forbidden_data_roots']:
        sys.addaudithook(target_guard(forbidden))
    for key, expected in ENV.items():
        if os.environ.get(key) != expected:
            raise ValueError('Incorrect environment: '+key)
    started = time.monotonic()
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.persistence_singleton import singleton_class
    cp = p['checkpoints'][model]
    for name, digest in cp['weights'].items():
        if sha(Path(cp['path'])/name) != digest:
            raise ValueError('Cached weights changed')
    native = singleton_class(Native)(model, cp['path'])
    with np.load(out/'inference/inputs.npz') as data:
        predictions, forecast, gates = predict_bank(native, data, read(out/'inference/plans.json'), p['sentinel'])
    denied = 0
    for forbidden in p['forbidden_data_roots']:
        try:
            (Path(forbidden)/'targets.npz').open('rb')
        except PermissionError:
            denied += 1
    if denied != len(p['forbidden_data_roots']):
        raise ValueError('Target denial gate')
    gates['target_roots_denied'] = denied
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    elapsed = time.monotonic()-started
    if peak > p['max_rss_bytes'] or elapsed >= p['max_seconds']:
        raise RuntimeError('Native cap exceeded')
    np.savez(out/f'{model}_forecasts.npz', bank_predictions=predictions, **forecast)
    write(out/f'{model}_worker.json', dict(valid=True, gates=gates, seconds=elapsed,
          peak_rss_bytes=peak, threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
          forecast_sha256=sha(out/f'{model}_forecasts.npz'), protocol_sha256=sha(out/'protocol.json')))


def run(out, model, slot_grant):
    """Supervise one worker, retain failures and save knowledge before returning."""
    if not slot_grant:
        raise PermissionError('HOLD: explicit coordinator slot grant required before launch')
    import fcntl
    p = binding(out)
    lock = (out/'native_worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((out/f'{model}{suffix}').exists() for suffix in ['.log','_execution.json','_forecasts.npz']):
        raise ValueError('Refuse native attempt overwrite')
    write(out/f'{model}_slot_grant.json', dict(coordinator_grant=slot_grant, protocol_sha256=sha(out/'protocol.json')))
    peak, failure = 0, None
    started = time.monotonic()
    with (out/f'{model}.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable,'-m','experiments.feature_search.postrelease_weather_worker',
               'worker','--out',str(out),'--model',model], env=dict(os.environ,**ENV), stdout=log,stderr=subprocess.STDOUT)
        print(dict(model=model,pid=proc.pid,run=str(out.resolve()),status='native launched'),flush=True)
        try:
            while proc.poll() is None:
                try:
                    lines = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8').splitlines()
                    peak = max(peak,int(next(s.split()[1] for s in lines if s.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):
                    pass
                if peak > p['max_rss_bytes'] or time.monotonic()-started >= p['max_seconds']-.5:
                    failure = 'RSS cap' if peak > p['max_rss_bytes'] else 'Time cap'
                    proc.kill(); proc.wait(); break
                time.sleep(.05)
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    result = dict(valid=proc.returncode==0 and failure is None, returncode=proc.returncode,
                  failure=failure,seconds=time.monotonic()-started,peak_rss_bytes=peak,pid=proc.pid)
    write(out/f'{model}_execution.json',result)
    knowledge = read(out/'knowledge.json')
    knowledge['exact_findings'].setdefault('native_execution',{})[model] = result
    knowledge['exact_findings']['native_launched'] = True
    knowledge['next_decision'] = 'Release native slot immediately when final capture completes; analyze saved arrays only after valid workers.' if result['valid'] else 'Preserve failed instrumentation; no partial scoring; diagnose without native relaunch.'
    write(out/'knowledge.json',knowledge);write(out/f'{model}_knowledge.json',knowledge)
    print(result,flush=True)
    if not result['valid']:
        raise RuntimeError('Native worker failed; preserve run, no partial scoring')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['run','worker'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',choices=['bolt','timesfm'],required=True)
    parser.add_argument('--slot-grant')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.out,args.model,args.slot_grant)
    else:
        worker(args.out,args.model)
