"""Check that a threshold, split or false-positive error is caught in isolated copies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
cases=[
 ('fit.py','np.where(raw_coef>=threshold,np.maximum(raw_coef,0.),0.)','np.maximum(raw_coef,0.)','closed_receiving_threshold'),
 ('fit.py','center=inputs[training].mean(0)','center=inputs.mean(0)','dose_scaling'),
 ("score.py","fp=int(np.sum(positive&~active))","fp=0","all_history_detection"),
]
for fname,old,new,test in cases:
    with tempfile.TemporaryDirectory(prefix='moment_receiver_dose_') as dest:
        path=Path(dest)/HERE.relative_to(ROOT);path.mkdir(parents=True)
        for f in HERE.glob('*.py'):shutil.copy(f,path/f.name)
        f=path/fname;s=f.read_text();assert s.count(old)==1;f.write_text(s.replace(old,new))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((dest,str(ROOT),str(ROOT/'tsfm_lens'),env.get('PYTHONPATH','')))
        check=subprocess.run([sys.executable,'-m','pytest','-q','-k',test,str(path/'test_receiver_detector.py')],cwd=dest,env=env,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(test,check.stdout.splitlines()[-1],flush=True)
        assert check.returncode==1 and '1 failed' in check.stdout and test in check.stdout,check.stdout
