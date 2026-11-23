"""Check that using the changed-history scale trips the frozen-normalization test."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
repo = here.parents[6]
with tempfile.TemporaryDirectory(prefix='moment_fixed_revin_plant_') as tmp:
    local = Path(tmp) / here.relative_to(repo)
    local.mkdir(parents=True)
    path = local / 'run.py'
    source = (here/'run.py').read_text()
    original = 'normalizer.mean, normalizer.stdev = mu, sd'
    assert source.count(original) == 1
    path.write_text(source.replace(original, 'normalizer.mean, normalizer.stdev = mu, actual_sd'))
    test = local/'test_frozen_norm.py'
    shutil.copy(here/test.name, test)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([tmp,str(repo),str(repo/'tsfm_lens'),env.get('PYTHONPATH','')])
    proc = subprocess.run([sys.executable,'-m','pytest','-q',str(test)],env=env,cwd=tmp,capture_output=True,text=True)
    print(proc.stdout)
    assert proc.returncode == 1 and 'AssertionError' in proc.stdout and '1 failed' in proc.stdout,proc.stderr
