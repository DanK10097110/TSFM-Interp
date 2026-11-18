"""Two monitored CPU workers for the frozen selective replication.

Each model still uses one thread and its own3GiB/time cap. Parallel execution
reduces elapsed time without increasing the number of model evaluations.
"""
from pathlib import Path
import argparse,concurrent.futures,os,subprocess,sys,time
from experiments.feature_search.coherence_study import read,write,sha


def monitor(out,model):
    """Monitor one model in an isolated process, with no target access in that worker."""
    p=read(out/'protocol.json');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if (out/f'{model}_worker.json').exists():raise ValueError('Refuse overwrite')
    start=time.monotonic();peak=0;reason=None
    with (out/f'{model}.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.selective_study','worker','--out',str(out),'--model',model],env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:peak=max(peak,int(next(l.split()[1] for l in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024)
            except (FileNotFoundError,StopIteration):pass
            if peak>p['worker_cap'] or time.monotonic()-start>p['seconds_per_model']+30:reason='Resource cap';proc.kill();proc.wait();break
            time.sleep(.25)
    result=dict(valid=proc.returncode==0 and reason is None,returncode=proc.returncode,reason=reason,seconds=time.monotonic()-start,peak_rss=peak)
    write(out/f'{model}_execution.json',result);print(model,result,flush=True);return model,result


def run(out):
    """Declare the execution-only amendment before any worker is launched."""
    p=read(out/'protocol.json');p['execution_mode']='Two isolated concurrent processes, one CPU thread and3GiB per model; maximum2 CPU threads,6GiB total; unchanged inference count'
    p['sources'][str(Path(__file__).resolve())]=sha(__file__);write(out/'protocol.json',p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda m:monitor(out,m),['bolt','timesfm']))
    write(out/'execution.json',dict(results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
