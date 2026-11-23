"""Plant load-bearing mistakes only in a temporary research subtree and run the designated tests."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PLANTS=[
    ('score.py','ok=(events>0)&(ref>0)','ok=(events>0)&(ref>1e-10*events)','small_measurable'),
    ('score.py',"return data[key+name]-data['noop'+name]","return data[key+name]",'identical_batch'),
    ('study.py','return observed.stop-1,future','return 61,future','reported_observed'),
    ('example.py','np.argmin(np.abs(rms-np.median(rms)))','np.argmax(rms)','example_median'),
]
for filename,original,planted,test in PLANTS:
    with tempfile.TemporaryDirectory(prefix='moment_weather_plant_') as dest:
        copy=Path(dest);path=copy/HERE.relative_to(ROOT);path.mkdir(parents=True)
        for f in HERE.glob('*.py'):shutil.copy(f,path/f.name)
        src=path/filename;value=src.read_text();assert value.count(original)==1
        src.write_text(value.replace(original,planted))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(copy),str(ROOT),str(ROOT/'tsfm_lens'),env.get('PYTHONPATH','')))
        proc=subprocess.run([sys.executable,'-m','pytest','-q',str(path/'test_moment_weather.py'),'-k',test],cwd=dest,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(test,proc.stdout.splitlines()[-1],flush=True)
        assert proc.returncode==1 and '1 failed' in proc.stdout and test in proc.stdout,proc.stdout
