"""Verify the two order invariants detect actual planted bugs in isolated copies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent;root=here.parents[5]
for file,original,replacement,case in (
    ('study.py','ORDER=[0,6,5,4,3,2,1,7]','ORDER=[0,1,2,3,4,5,6,7]','reverses_only_interior'),
    ('score.py',"(scores['BB']-scores['BA'])","(scores['BA']-scores['BB'])",'balanced_advantage_compares')
):
    with tempfile.TemporaryDirectory(prefix='ramp_order_plant_') as d:
        td=Path(d)/here.relative_to(root);td.mkdir(parents=True)
        for f in here.glob('*.py'):shutil.copy(f,td/f.name)
        f=td/file;text=f.read_text();assert text.count(original)==1;f.write_text(text.replace(original,replacement))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([d,str(root),str(root/'tsfm_lens'),env.get('PYTHONPATH','')])
        res=subprocess.run([sys.executable,'-m','pytest','-q',str(td/'test_ramp_interior_order.py'),'-k',case],
                           cwd=d,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(case,res.stdout.splitlines()[-1],flush=True)
        assert res.returncode==1 and '1 failed' in res.stdout and case in res.stdout,res.stdout
