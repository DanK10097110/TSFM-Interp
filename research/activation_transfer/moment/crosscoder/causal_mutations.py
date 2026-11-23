"""Prove each forecast-scoring invariant fails against a plausible isolated regression."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent;root=here.parents[3]
cases=[
    ('causal_scores.py','1-ratio(accumulator(error),scale)','1-ratio(accumulator(error*open_gate),scale)','closed_gate_wrong'),
    ('causal_scores.py','station_weights,grid,date_weights,optimize=True','station_weights,grid,np.ones_like(date_weights),optimize=True','crossed_resampling_repeats'),
    ('causal_scores.py',"return summary(seed_scores['masked']-seed_scores['shuffled'])","return summary(seed_scores['joint']-seed_scores['shuffled'])",'pairing_comparison_holds'),
    ('interventions.py','for t in (False, True):','for t in (True,):','native_replay_samples')]
for file,old,new,case in cases:
    with tempfile.TemporaryDirectory(prefix='causal_cc_bug_') as td:
        dest=Path(td)/here.relative_to(root);dest.mkdir(parents=True)
        for name in ('interventions.py','causal_scores.py','test_causal_interventions.py'):
            shutil.copy(here/name,dest/name)
        mod=dest/file;s=mod.read_text();assert s.count(old)==1;mod.write_text(s.replace(old,new))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([td,str(root),str(root/'tsfm_lens'),env.get('PYTHONPATH','')])
        proc=subprocess.run([sys.executable,'-m','pytest','-q',str(dest/'test_causal_interventions.py'),'-k',case],cwd=td,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(case,proc.stdout.splitlines()[-1],flush=True)
        assert proc.returncode==1 and '1 failed' in proc.stdout and case in proc.stdout,proc.stdout
