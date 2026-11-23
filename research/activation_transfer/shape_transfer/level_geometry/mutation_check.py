"""Require a real assertion failure for each planted geometry or scoring regression."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PLANTS=[
    ('zero_scalar_replays_are_exactly_zero_for_directional_metrics','score','return np.where(blank[None,:,None],0.,effects),floor','return effects,floor','Score FP32 replay noise as an actual zero-dose direction.'),
    ('correction_cancels_the_measured_offset_at_fixed_norm','run','coefficient=-mean_output/g2','coefficient=-((mapped@basis.T)@g)/g2','Ignore measured output movement outside the PCA span.'),
    ('correction_cancels_the_measured_offset_at_fixed_norm','run','corrected=unit(before_rescale)*length[:,None]','corrected=before_rescale','Drop the matched hidden norm after correction.'),
    ('spectrum_separates_isotropic_energy_from_shared_response','run','float(np.square(level).sum()/total)','float(np.square(e@avg).sum()/np.square(avg).sum())','Use the mean Jacobian when estimating total per-history output energy.'),
    ('random_control_matches_budget_and_uses_fit_level_sign','run','return (coeff*signs[:,None])@basis*norm,signs','return coeff@basis*norm,signs','Discard the FIT-based lift orientation of the random controls.'),
    ('decomposition_distinguishes_wrong_waveform_from_level_offset','score','np.square(y-w).mean(-1)/need','np.square(effect-w).mean(-1)/need','Count an output level error also as a centered shape error.'),
]


def run():
    logs=[]
    with tempfile.TemporaryDirectory(prefix='level-geometry-mutation-') as tmp:
        for i, row in enumerate([None,*PLANTS]):
            box=Path(tmp)/str(i);folder=box/'research/activation_transfer/shape_transfer/level_geometry'
            folder.mkdir(parents=True)
            for f in HERE.glob('*.py'):shutil.copyfile(f,folder/f.name)
            selected=str(folder/'test_geometry.py');env={**os.environ,'PYTHONPATH':os.pathsep.join(map(str,[box,ROOT,ROOT/'tsfm_lens'])),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            if row:
                name,module,before,after,reason=row;selected+='::test_'+name
                p=folder/(module+'.py');original=p.read_text();assert original.count(before)==1
                p.write_text(original.replace(before,after));assert p.read_bytes()!=(HERE/p.name).read_bytes()
            cmd=[sys.executable,'-m','pytest','-q','--tb=short',selected]
            r=subprocess.run(cmd,cwd=box,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            summary=r.stdout.strip().splitlines()[-1]
            if row:
                assert r.returncode==1 and '1 failed' in summary and ('AssertionError' in r.stdout or re.search(r'^E\s+assert\b',r.stdout,re.M)),r.stdout
                logs.append('| `test_'+name+'` | '+reason+' | `'+summary+'` |')
            else:
                assert r.returncode==0 and '5 passed' in summary,r.stdout;baseline=summary
    return '\n'.join(['# Geometry test verification','',f'The unmodified isolated suite reported `{baseline}`. Each targeted test had to fail by assertion on an independently altered source copy with the exact named alteration. [Reproduce](mutation_check.py)','','| Test | Planted change | Actual result |','|---|---|---|',*logs,''])


if __name__=='__main__':
    print(run())
