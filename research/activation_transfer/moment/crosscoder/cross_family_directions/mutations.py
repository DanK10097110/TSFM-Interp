"""Each protected scientific invariant must detect an actual isolated regression."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent;root=here.parents[4]
cases=(
    ('data.py','np.testing.assert_array_equal(spans[ids[:-1],1],spans[ids[1:],0])','pass','native_intervals_cover'),
    ('directions.py','-ta*b_ok[...,None]','+0.0*ta*b_ok[...,None]','other_coordinate_excludes'),
    ('directions.py',"[archive[k].astype(float)*scale for k,scale","[archive[k].astype(float) for k,scale",'relative_view_scales'),
    ('precision.py','rng.choice(active,4,replace=False)','rng.choice(active,4,replace=True)','precision_uses_four'),
)
for file,orig,bug,case in cases:
    with tempfile.TemporaryDirectory(prefix='tsfm_families_plant_') as temp:
        dest=Path(temp)/here.relative_to(root);dest.mkdir(parents=True)
        for path in here.glob('*.py'):shutil.copy(path,dest/path.name)
        f=dest/file;s=f.read_text();assert s.count(orig)==1;f.write_text(s.replace(orig,bug))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([temp,str(root),str(root/'tsfm_lens'),env.get('PYTHONPATH','')])
        run=subprocess.run([sys.executable,'-m','pytest','-q',str(dest/'test_cross_family_directions.py'),'-k',case],
            cwd=temp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(case,run.stdout.splitlines()[-1],flush=True)
        assert run.returncode==1 and '1 failed' in run.stdout and case in run.stdout,run.stdout
