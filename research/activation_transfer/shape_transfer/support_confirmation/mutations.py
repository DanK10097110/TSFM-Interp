"""Validate that a plausible source-level regression really fails each newly added test."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from research.activation_transfer.shape_transfer.support_confirmation.data import HERE,ROOT

CHANGES=[
 ('derivative_swap','derivative.py','response=held_goal[...,None]','response=-held_goal[...,None]','Invert the sign of the fixed current waveform supplied to a foreign derivative.'),
 ('derivative_swap','derivative.py',"if mode=='dc':","if mode=='unused':",'Let the mean-only controller see horizon shape contrasts.'),
 ('orthogonalizing','residual.py','rank=(strength>TOL).sum','rank=(strength>=0).sum','Count an exactly annihilated candidate row as retained rank.'),
 ('frequency_uses','supports.py','np.lexsort((np.arange(width),-counts))','np.lexsort((-np.arange(width),-counts))','Reverse the feature-ID tie break.'),
 ('low_overlap','supports.py','linear_sum_assignment(cost)','linear_sum_assignment(np.where(np.isfinite(cost),-cost,cost))','Choose the most overlapping permitted perfect assignment.'),
 ('low_overlap','supports.py','replacement[:,:k-value]','replacement[:,::-1][:,:k-value]','Reverse train-only replacements when own-feature count is zero.'),
 ('norm_cap','run.py','np.minimum(1.,np.asarray(budget)','np.maximum(1.,np.asarray(budget)','Expand instead of clip the proposed displacement.'),
 ('seed_contrasts','score.py','np.mean(np.stack(values),axis=0)','np.max(np.stack(values),axis=0)','Select the best seed separately in each bootstrap draw.'),
]


def run():
    result=[]
    for key,file,before,after,reason in CHANGES:
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);pack=folder/HERE.relative_to(ROOT);pack.mkdir(parents=True)
            for p in HERE.glob('*.py'):shutil.copy2(p,pack/p.name)
            src=pack/file;s=src.read_text(encoding='utf-8');assert s.count(before)==1;src.write_text(s.replace(before,after),encoding='utf-8')
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[folder,ROOT,ROOT/'tsfm_lens'])))
            q=subprocess.run([sys.executable,'-m','pytest','-q',str(pack/'test_support_confirmation.py'),'-k',key],cwd=folder,env=env,capture_output=True,text=True)
            assert q.returncode==1 and '1 failed' in q.stdout and 'AssertionError' in q.stdout,(key,q.stdout)
            result.append(f'| {key} | {reason} | {q.stdout.strip().splitlines()[-1]} |')
    (HERE/'MUTATION_CHECK.md').write_text('# Mutation checks\n\nSix tests passed on the original code. Each row records the assertion failure from an isolated modified package.\n\n| Test | Change | Pytest result |\n|---|---|---|\n'+'\n'.join(result)+'\n',encoding='utf-8')


if __name__=='__main__':run()
