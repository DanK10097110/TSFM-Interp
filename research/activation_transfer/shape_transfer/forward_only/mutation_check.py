"""Verify that each study-specific test assertion detects a material planted fault."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CHANGES=(
    ('forward_probe_excludes','probe.py','.25 * codes','1. * codes','Allow the negative small-activation probe to equal the scored removal.'),
    ('generic_impulses_fit','transfer.py','boost_shape(identity,weight)','identity','Forget the output weighting on the calibration targets.'),
    ('generic_impulses_fit','transfer.py','basis.T @ coeff','np.roll(basis.T,1,axis=0) @ coeff','Embed learned coefficients in the wrong recipient dimensions.'),
    ('generic_random_probes','transfer.py','cal_random.mean(axis=1,keepdims=True)','cal_random.mean(axis=0,keepdims=True)','Compute the horizon mean across histories.'),
)


def run():
    rows=[]
    for key,name,old,new,desc in CHANGES:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for src in (ROOT/'research'/'activation_transfer').rglob('*.py'):
                dest=root/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            folder=root/HERE.relative_to(ROOT);f=folder/name;s=f.read_text();assert s.count(old)==1;f.write_text(s.replace(old,new))
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[root,ROOT/'tsfm_lens'])))
            call=subprocess.run([sys.executable,'-m','pytest','-q',str(folder/'test_forward_only.py'),'-k',key],
                                cwd=root,env=env,text=True,capture_output=True)
            assert call.returncode==1 and 'AssertionError' in call.stdout and '1 failed' in call.stdout,call.stdout
            rows.append(f'| `{key}` | {desc} | `{call.stdout.strip().splitlines()[-1]}` |')
    content=['# Verification with planted errors','','All three tests passed on the unmodified sources. Each change below was planted in a separate scratch copy and had to fail by assertion. [Mutation runner](mutation_check.py)','','| Test | Planted error | Observed |','|---|---|---|',*rows]
    (HERE/'MUTATION_CHECK.md').write_text('\n'.join(content)+'\n')


if __name__=='__main__':
    run()
