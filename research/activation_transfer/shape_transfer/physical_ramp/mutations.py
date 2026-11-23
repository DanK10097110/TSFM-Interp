"""Plant one outcome-relevant implementation error per new test in independent /tmp copies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
cases=[('study.py','np.linspace(-2.,2.,8,dtype=np.float32)','np.linspace(0.,2.,8,dtype=np.float32)','ramp_has_zero_mean'),
       ('study.py','lookup[(wrong[s],o)]','lookup[(wrong[s],(o+1)%4)]','other_station_preserves'),
       ('score.py','summed=counts@mat.reshape(len(mat),-1)','summed=counts.sum(1)[:,None]*mat.mean(0).reshape(1,-1)','template_is_reestimated')]
for fname,old,new,case in cases:
    with tempfile.TemporaryDirectory(prefix='raw_ramp_plant_') as td:
        p=Path(td)/HERE.relative_to(ROOT);p.mkdir(parents=True)
        for f in HERE.glob('*.py'):shutil.copy(f,p/f.name)
        f=p/fname;v=f.read_text();assert v.count(old)==1;f.write_text(v.replace(old,new))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([td,str(ROOT),str(ROOT/'tsfm_lens'),env.get('PYTHONPATH','')])
        proc=subprocess.run([sys.executable,'-m','pytest','-q',str(p/'test_physical_ramp.py'),'-k',case],cwd=td,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(case,proc.stdout.splitlines()[-1],flush=True)
        assert proc.returncode==1 and '1 failed' in proc.stdout and case in proc.stdout,proc.stdout
