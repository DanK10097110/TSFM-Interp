"""Plant the same-date leakage bug in an isolated file tree."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent;root=here.parents[5]
with tempfile.TemporaryDirectory(prefix='past_only_date_plant_') as temp:
    dest=Path(temp)/here.relative_to(root);dest.mkdir(parents=True)
    for f in here.glob('*.py'):shutil.copy(f,dest/f.name)
    f=dest/'study.py';old=f.read_text();assert old.count(' < requested[:,None]')==1
    f.write_text(old.replace(' < requested[:,None]',' <= requested[:,None]'))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([temp,str(root),str(root/'tsfm_lens'),env.get('PYTHONPATH','')])
    p=subprocess.run([sys.executable,'-m','pytest','-q',str(dest/'test_ramp_past_only.py')],cwd=temp,env=env,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(p.stdout.splitlines()[-1]);assert p.returncode==1 and '1 failed' in p.stdout and 'requires_strictly_prior' in p.stdout,p.stdout
