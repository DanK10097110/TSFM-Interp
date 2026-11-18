"""Explicitly granted, monitored single-Bolt dispatcher for the causal pilot."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from experiments.feature_search.ffn_selectivity_causal import read,write,knowledge,MEMORY,SECONDS,CELL_CAP,stage1


def run(out,slot_granted=False):
    """HOLD by default; one CPU worker and no automatic subsequent experiment."""
    if not slot_granted:
        raise RuntimeError('HOLD causal native pending new slot grant')
    if (out/'execution.json').exists():
        raise FileExistsError('Refuse causal rerun')
    p=read(out/'protocol.json')
    if p['model']!='bolt' or p['cells']!=CELL_CAP:
        raise ValueError('Only frozen384-cell Bolt schedule permitted')
    import fcntl
    execution={}
    try:
        with (out/'native.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',FFN_CAUSAL_MONITORED='1',
                PYTHONPATH=str(Path(__file__).resolve().parents[2]),**{k:'1' for k in stage1.THREAD_VARS})
            started=time.monotonic();peak=0;reason=None
            with (out/'bolt.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.ffn_selectivity_causal',
                    'worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        try:
                            lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                            peak=max(peak,int(next(line.split()[1] for line in lines if line.startswith('VmRSS:')))*1024)
                        except (FileNotFoundError,StopIteration):
                            pass
                        if peak>MEMORY or time.monotonic()-started>SECONDS:
                            reason='Resource cap exceeded';break
                        time.sleep(.1)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill();proc.wait()
            worker=read(out/'bolt_worker.json') if (out/'bolt_worker.json').exists() else {}
            valid=(proc.returncode==0 and reason is None and worker.get('valid',False)
                and worker.get('cells')==CELL_CAP and worker.get('peak_rss',float('inf'))<=MEMORY
                and worker.get('seconds',float('inf'))<=SECONDS)
            execution=dict(bolt=dict(valid=valid,returncode=proc.returncode,reason=reason,
                seconds=time.monotonic()-started,monitored_peak=peak,cells=worker.get('cells')))
            write(out/'execution.json',execution)
            if not valid:
                raise RuntimeError('Causal worker failed gates/caps')
    except Exception as exc:
        knowledge(out,'native_execution','failed',dict(error_type=type(exc).__name__,error=str(exc),execution=execution),
                  'Release native slot; inspect failed instrumentation before any new run.')
        raise
    knowledge(out,'native_execution','complete',execution,'Release native slot immediately; analyze saved causal outputs only.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal4963000'))
    parser.add_argument('--slot-granted',action='store_true')
    args=parser.parse_args()
    run(args.out,args.slot_granted)
