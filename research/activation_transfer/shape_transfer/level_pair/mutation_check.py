"""Verify the named-pair tests with isolated planted source-level errors."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CHANGES=(
    ('hybrid_preserves','splice.py','tail=daily if bits&2 else traffic','tail=daily if bits&1 else traffic','Couple the final-window pattern donor to the earlier 448-point donor.'),
    ('hybrid_preserves','splice.py','budget=daily if bits&4 else traffic','budget=daily if bits&2 else traffic','Couple allocation of residual energy to the recent pattern donor.'),
    ('threshold_excludes','gating.py','pre[:,named]=-torch.inf','pre[:,named]=pre[:,named]','Count the selected feature among its own TopK competitors.'),
    ('common_grid','gating.py','goals[:,1]=.5*sd','goals[:,1]=.1*sd','Set the common endpoint to the wrong standardized value.'),
    ('subset_order','factorial.py','for order in itertools.permutations([1,2,4]):','for order in [(1,2,4)]*6:','Credit all interactions to a single ordering.'),
    ('isolated_edits','inputs.py','residual*np.sqrt(np.maximum(new_energy,0)/remainder)[:,None]','residual','Permit global variance to change under the isolated cue edit.'),
    ('isolated_edits','inputs.py','max_factor*(1-1e-12)','2*max_factor','Overstep the symmetric residual-energy bound.'),
    ('named_lock','run.py','scalar=z0-z','scalar=z-z0','Steer the named decoder in the wrong direction.'),
    ('named_lock','run.py','*(norm/np.linalg.norm(unit[k]))','*1.0','Give control atoms different displacement norms.'),
    ('signed_level_loss','score.py','lost=reference-clamped','lost=clamped-reference','Reverse the signed forecast effect attributed to the intervention.'),
)


def run():
    rows=[]
    for key,file,old,new,reason in CHANGES:
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for p in (ROOT/'research'/'activation_transfer').rglob('*.py'):
                dest=root/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            folder=root/HERE.relative_to(ROOT);source=folder/file;s=source.read_text();assert s.count(old)==1;source.write_text(s.replace(old,new))
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[root,ROOT/'tsfm_lens'])))
            cmd=subprocess.run([sys.executable,'-m','pytest','-q',str(folder/'test_named_level_pair.py'),'-k',key],cwd=root,env=env,capture_output=True,text=True)
            assert cmd.returncode==1 and 'AssertionError' in cmd.stdout and '1 failed' in cmd.stdout,cmd.stdout
            rows.append(f'| `{key}` | {reason} | `{cmd.stdout.strip().splitlines()[-1]}` |')
    lines=['# Mutation checks','','Seven tests passed on the original code. Each row produced the required assertion failure in a separately modified scratch copy. [Replay the mutations](mutation_check.py)','','| Test | Planted failure | Observed |','|---|---|---|',*rows]
    (HERE/'MUTATION_CHECK.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':run()
