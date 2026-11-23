"""Exercise control tests against single planted regressions in independent scratch copies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REL = HERE.relative_to(ROOT)
CHANGES = (
    ('mean_donor_uses', 'local[cal].mean(axis=0)', 'local[~cal].mean(axis=0)', 'Average the donor derivatives on evaluation histories.'),
    ('mean_donor_uses', "arm == 'mean_donor'", "arm == 'never'", 'Supply the true item derivative to the mean-donor control.'),
    ('wrong_lead_control', 'target_jac[:, ORDER]', 'target_jac', 'Leave recipient leads in their correct order for the wrong-lead control.'),
)


def run():
    lines = []
    for key, old, new, description in CHANGES:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in (ROOT/'research'/'activation_transfer').rglob('*.py'):
                dest = root / path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            source_path = root/REL/'run.py'
            source = source_path.read_text()
            assert source.count(old) == 1
            source_path.write_text(source.replace(old, new))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,[root, ROOT/'tsfm_lens'])))
            result = subprocess.run([sys.executable,'-m','pytest','-q',str(root/REL/'test_timer_receiver.py'),'-k',key],
                                    cwd=root,env=env,text=True,capture_output=True)
            assert result.returncode == 1 and 'AssertionError' in result.stdout and '1 failed' in result.stdout, result.stdout
            lines.append((key,description,result.stdout.strip().splitlines()[-1]))
    content = ['# Independent planted control errors', '', 'The clean suite passed both control tests. Each row copied the actual source and planted one change before running the named test. [Runner](mutation_check.py)', '', '| Test | Plant | Observed |', '|---|---|---|']
    content += [f'| `{k}` | {d} | `{r}` |' for k,d,r in lines]
    (HERE/'MUTATION_CHECK.md').write_text('\n'.join(content)+'\n')


if __name__ == '__main__':
    run()
