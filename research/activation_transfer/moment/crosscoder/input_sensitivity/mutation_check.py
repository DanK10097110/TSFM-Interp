"""Verify three physical-sensitivity assertions against isolated plausible regressions."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PLANTS=(('capture.py','patterns[-1]=1.','patterns[-1]=-1.','physical_plus_minus'),
        ('score.py','su,tu=unit(s),unit(t)','su,tu=unit(s-s.mean(2,keepdims=True)),unit(t-t.mean(2,keepdims=True))','inner_product'),
        ('score.py','equal_atom=float(per.mean())','equal_atom=float(events.mean())','equal_feature'))
for filename,orig,repl,match in PLANTS:
    with tempfile.TemporaryDirectory(prefix='crosscoder_input_sens_') as dest:
        local=Path(dest)/HERE.relative_to(ROOT);local.mkdir(parents=True)
        for f in HERE.glob('*.py'):
            shutil.copy(f,local/f.name)
        p=local/filename;s=p.read_text();assert s.count(orig)==1;p.write_text(s.replace(orig,repl))
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((dest,str(ROOT),str(ROOT/'tsfm_lens'),env.get('PYTHONPATH','')))
        proc=subprocess.run([sys.executable,'-m','pytest','-q',str(local/'test_sensitivity.py'),'-k',match],
                            cwd=dest,env=env,capture_output=True,text=True)
        print(proc.stdout,flush=True)
        assert proc.returncode==1 and 'AssertionError' in proc.stdout and '1 failed' in proc.stdout,proc.stderr
