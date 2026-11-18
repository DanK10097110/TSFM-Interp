"""Serial monitored stage-one dispatcher; no causal-stage launch path."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

from experiments.feature_search.ffn_selectivity_study import MODELS,MEMORY,SECONDS,THREAD_VARS,read,write,knowledge


def run(out,slot_granted=False):
    """Require a new explicit grant; cap resident bytes, elapsed time and cell count."""
    if not slot_granted:
        raise RuntimeError('HOLD FFN native: a new slot grant is required')
    if (out/'execution.json').exists():
        raise FileExistsError('Refuse native rerun')
    import fcntl
    execution={}
    p=read(out/'protocol.json')
    if p['cells_per_model']>500:
        raise ValueError('Frozen native-cell cap exceeded')
    try:
        with (out/'native.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for model in MODELS:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',FFN_SELECTIVITY_MONITORED='1',
                    PYTHONPATH=str(Path(__file__).resolve().parents[2]),**{key:'1' for key in THREAD_VARS})
                started,peak,reason=time.monotonic(),0,None
                with (out/f'{model}.log').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.ffn_selectivity_study',
                        'worker','--out',str(out),'--model',model],env=env,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        while proc.poll() is None:
                            try:
                                lines=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                                peak=max(peak,int(next(s.split()[1] for s in lines if s.startswith('VmRSS:')))*1024)
                            except (FileNotFoundError,StopIteration):
                                pass
                            if peak>MEMORY or time.monotonic()-started>SECONDS:
                                reason='Resource cap exceeded'
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
                path=out/f'{model}_worker.json'
                record=read(path) if path.exists() else {}
                valid=proc.returncode==0 and reason is None and record.get('valid',False)
                valid=valid and record.get('peak_rss',float('inf'))<=MEMORY and record.get('seconds',float('inf'))<=SECONDS
                valid=valid and record.get('cells')==p['cells_per_model']
                execution[model]=dict(valid=valid,returncode=proc.returncode,reason=reason,
                    seconds=time.monotonic()-started,monitored_peak=peak,cells=record.get('cells'))
                write(out/'execution.json',execution)
                if not valid:
                    raise RuntimeError(f'{model} failed native gates/budget; stopping')
    except Exception as exc:
        knowledge(out,'native_execution','failed',dict(error_type=type(exc).__name__,error=str(exc),execution=execution),
                  'Inspect failed instrumentation; no causal-stage execution.')
        raise
    knowledge(out,'native_execution','complete',execution,
              'Native slot can be released. Analyze source-held-out selectivity only; no automatic causal second stage.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_pilot4961000'))
    parser.add_argument('--slot-granted',action='store_true')
    args=parser.parse_args()
    run(args.out,args.slot_granted)
