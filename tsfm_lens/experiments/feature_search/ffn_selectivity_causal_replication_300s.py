"""Explicit300-second execution amendment; original replication protocol retained.

Both worker's internal guard and supervising process enforce the user's tighter
deadline. Scientific schedule/functions are imported from the frozen module.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from experiments.feature_search import ffn_selectivity_causal_replication as study

SECONDS=300


def validate(out):
    """Require a bound execution-only amendment before any native loading."""
    p=study.read(out/'protocol.json');a=study.read(out/'execution_amendment.json')
    if (a['base_protocol_sha256']!=study.sha(out/'protocol.json') or a['launcher_sha256']!=study.sha(__file__)
            or a['seconds_per_worker']!=300 or a['cells']!=1088 or p['cells']!=1088 or p['model']!='bolt'
            or not a['science_and_controls_unchanged']):
        raise ValueError('Bound300-second execution amendment required')
    return p


def run(out,slot_granted=False):
    """Only one granted Bolt worker, supervised at300seconds and3GiB."""
    if not slot_granted:
        raise RuntimeError('HOLD replication until explicit slot grant')
    validate(out)
    if (out/'execution.json').exists():
        raise FileExistsError('Refuse replication rerun')
    import fcntl
    result={}
    try:
        with (out/'native.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',FFN_CROSSED_MONITORED='1',
                TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',**{k:'1' for k in study.causal.stage1.THREAD_VARS})
            started=time.monotonic();peak=0;reason=None
            with (out/'bolt.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.ffn_selectivity_causal_replication_300s',
                    'worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        try:
                            lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                            peak=max(peak,int(next(v.split()[1] for v in lines if v.startswith('VmRSS:')))*1024)
                        except (FileNotFoundError,StopIteration):
                            pass
                        if peak>study.MEMORY or time.monotonic()-started>SECONDS:
                            reason='300-second/3GiB cap exceeded';break
                        time.sleep(.1)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill();proc.wait()
            worker=study.read(out/'bolt_worker.json') if (out/'bolt_worker.json').exists() else {}
            elapsed=time.monotonic()-started
            valid=(proc.returncode==0 and reason is None and worker.get('valid',False) and worker.get('cells')==1088
                and worker.get('peak_rss',float('inf'))<=study.MEMORY and worker.get('seconds',float('inf'))<=300 and elapsed<=300)
            result=dict(bolt=dict(valid=valid,returncode=proc.returncode,seconds=elapsed,monitored_peak=peak,reason=reason,cells=worker.get('cells'),seconds_cap=300,
                execution_amendment_sha256=study.sha(out/'execution_amendment.json')))
            study.write(out/'execution.json',result)
            if not valid:
                raise RuntimeError('Replication failed amended execution gates')
    except Exception as exc:
        study.knowledge(out,'native_execution','failed',dict(error_type=type(exc).__name__,error=str(exc),execution=result),
            'Release slot; inspect failure before any new run.')
        raise
    study.knowledge(out,'native_execution','complete',result,'Release native slot immediately; analyze/audit saved outputs only.')


def worker(out):
    """Apply the tighter internal deadline before invoking the original worker."""
    validate(out)
    study.SECONDS=SECONDS
    study.worker(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','worker'])
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal_replication4964000'))
    parser.add_argument('--slot-granted',action='store_true')
    args=parser.parse_args()
    run(args.out,args.slot_granted) if args.command=='run' else worker(args.out)
