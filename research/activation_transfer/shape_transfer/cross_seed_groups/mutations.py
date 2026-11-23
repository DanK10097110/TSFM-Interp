"""Exercise the rotation, matched dose and distinct aggregation assertions in temporary source copies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
cases=[('study.py','    return q\n','    return np.eye(d)\n','global_control'),
       ('study.py','return v*np.linalg.norm(ref,axis=-1,keepdims=True)/have','return v/have','specific_group_norm'),
       ('score.py',"float(metrics(sums.sum(0))['skill'])","float(np.nanmean(per))",'pooled_responses')]
for filename,old,new,case in cases:
    with tempfile.TemporaryDirectory(prefix='tsfm_groups_plant_') as tmp:
        p=Path(tmp)/HERE.relative_to(ROOT);p.mkdir(parents=True)
        for f in HERE.glob('*.py'):shutil.copy(f,p/f.name)
        f=p/filename;x=f.read_text();assert x.count(old)==1;f.write_text(x.replace(old,new))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([tmp,str(ROOT),str(ROOT/'tsfm_lens'),env.get('PYTHONPATH','')])
        res=subprocess.run([sys.executable,'-m','pytest','-q',str(p/'test_cross_seed_groups.py'),'-k',case],cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(case,res.stdout.splitlines()[-1],flush=True)
        assert res.returncode==1 and '1 failed' in res.stdout and case in res.stdout,res.stdout
