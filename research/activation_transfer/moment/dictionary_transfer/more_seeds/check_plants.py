"""Run each new test against its designated regression in an isolated namespace package."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PLANTS = (
    ('run.py', "-match['target_z'][r, c, None] * single", "-original['z'][r, c, None] * single", 'native_gate'),
    ('score.py', 'macro_skill=cell(point, sb, fb)', 'macro_skill=cell(1-residual/truth, sb, fb)', 'equal_feature'),
)
for filename, old, new, test in PLANTS:
    with tempfile.TemporaryDirectory(prefix='moment_refit_plant_') as dest:
        local = Path(dest) / HERE.relative_to(ROOT)
        local.mkdir(parents=True)
        for src in HERE.glob('*.py'):
            shutil.copy(src, local / src.name)
        path = local / filename
        content = path.read_text()
        assert content.count(old) == 1
        path.write_text(content.replace(old, new))
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([dest, str(ROOT), str(ROOT / 'tsfm_lens'), env.get('PYTHONPATH', '')])
        res = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(local / 'test_more_seeds.py'), '-k', test],
                             cwd=dest, env=env, capture_output=True, text=True)
        print(res.stdout, flush=True)
        assert res.returncode == 1 and 'AssertionError' in res.stdout and '1 failed' in res.stdout, res.stderr
