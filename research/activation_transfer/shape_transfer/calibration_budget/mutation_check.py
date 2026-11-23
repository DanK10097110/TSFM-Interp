"""Verify material regressions against isolated copies of the exact study sources."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CHANGES=(
    ('pilot_retains','study.py','boost_shape(jac,weight).mean(axis=0)','boost_shape(jac,weight).reshape(-1,jac.shape[-1])','Choose directions responsive to inconsistent or sign-reversing item shifts.'),
    ('draws_are','study.py','default_rng(seed).permutation(n)','default_rng(SEEDS[0]).permutation(n)','Use one identical calibration split for all replicates.'),
    ('draws_are','score.py',"'pilot':768","'pilot':576",'Omit every pilot probe from reporting the cost.'),
    ('seed_counts','score.py','draws.T*seed_counts','draws.T*seed_counts[:,::-1]','Mispair calibration weights and fitted-library scores.'),
)


def run():
    rows=[]
    for key,file,old,new,reason in CHANGES:
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for s in (ROOT/'research'/'activation_transfer').rglob('*.py'):
                to=root/s.relative_to(ROOT);to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,to)
            folder=root/HERE.relative_to(ROOT);src=folder/file;body=src.read_text();assert body.count(old)==1;src.write_text(body.replace(old,new))
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[root,ROOT/'tsfm_lens'])))
            r=subprocess.run([sys.executable,'-m','pytest','-q',str(folder/'test_calibration_budget.py'),'-k',key],
                             cwd=root,env=env,capture_output=True,text=True)
            assert r.returncode==1 and 'AssertionError' in r.stdout and '1 failed' in r.stdout,r.stdout
            rows.append(f'| `{key}` | {reason} | `{r.stdout.strip().splitlines()[-1]}` |')
    doc=['# Verification with planted errors','','The three new tests passed on the original sources. Each independent replacement below had to produce an assertion failure. [Mutation runner](mutation_check.py)','','| Test | Planted error | Result |','|---|---|---|',*rows]
    (HERE/'MUTATION_CHECK.md').write_text('\n'.join(doc)+'\n')


if __name__=='__main__':run()
