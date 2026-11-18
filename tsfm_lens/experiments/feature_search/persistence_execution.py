"""Reusable single native slot supervisor for separately frozen recovery runs."""
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.feature_search.persistence_protocol import ENV, write


def execute(out, module, model):
    """Monitor one native worker for at most600s and3GiB RSS; preserve failures."""
    import fcntl
    lock = (out/'native_worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logpath = out/f'{model}.log'
    if logpath.exists():
        raise ValueError('Refuse to overwrite native attempt')
    start = time.monotonic(); peak = 0; reason = None
    with logpath.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-m', module, 'worker', '--out', str(out), '--model', model],
                                env=dict(os.environ, **ENV), stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                try:
                    lines = Path(f'/proc/{proc.pid}/status').read_text(encoding='utf-8').splitlines()
                    peak = max(peak, int(next(l.split()[1] for l in lines if l.startswith('VmRSS:')))*1024)
                except (FileNotFoundError, StopIteration):
                    pass
                if peak > 3*2**30 or time.monotonic()-start >= 599.5:
                    reason = 'RSS cap' if peak > 3*2**30 else 'Time cap'
                    proc.kill(); proc.wait(); break
                time.sleep(.1)
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
    result = dict(valid=proc.returncode == 0 and reason is None, returncode=proc.returncode,
                  reason=reason, seconds=time.monotonic()-start, peak_rss_bytes=peak)
    write(out/f'{model}_execution.json', result)
    print(model, result, flush=True)
    if not result['valid']:
        raise RuntimeError('Native worker failed: '+str(logpath))
