"""Plant an actual omitted date-weight bug in an isolated copy."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
here=Path(__file__).resolve().parent;root=here.parents[4]
with tempfile.TemporaryDirectory(prefix='ramp_dates_mutation_') as td:
    tmp=Path(td)/here.relative_to(root);tmp.mkdir(parents=True)
    for path in here.glob('*.py'):shutil.copy(path,tmp/path.name)
    file=tmp/'score.py';orig=file.read_text();assert orig.count('((station_weights@square)*date_weights)')==1
    file.write_text(orig.replace('((station_weights@square)*date_weights)','((station_weights@square)*np.ones_like(date_weights))'))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([td,str(root),str(root/'tsfm_lens'),env.get('PYTHONPATH','')])
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(tmp/'test_ramp_station_date.py')],env=env,cwd=td,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(result.stdout.splitlines()[-1]);assert result.returncode==1 and '1 failed' in result.stdout and 'test_independent_station_date_resampling' in result.stdout,result.stdout
