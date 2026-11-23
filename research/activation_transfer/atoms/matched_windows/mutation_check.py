"""Exercise each new test with a plausible single-file regression in an isolated copy."""
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
    ('adjacent_uses_the_original_center_and_an_actual_predecessor','prepare','raw[start+step:end+step]-mu','raw[start+step:end+step]-raw[start+step:end+step].mean()','Independently recenter the adjacent window.'),
    ('natural_write_matches_each_feature_dose_and_keeps_the_series_direction','patch','per_row[i]).astype','per_row[j]).astype','Select a window direction by atom index instead of series index.'),
    ('natural_write_matches_each_feature_dose_and_keeps_the_series_direction','patch','codes.astype(np.float64)*np.linalg.norm(decoder.astype(np.float64),axis=1)[None]','codes.astype(np.float64)','Drop the decoder norm from the natural write dose.'),
    ('comparison_keeps_both_references_measurable_and_uses_paired_draws','score',"common=np.isfinite(point['atom']['cosine'])&np.isfinite(point['window']['cosine'])","common=np.isfinite(point['atom']['cosine'])",'Include features with a zero window-source reference in the common average.'),
    ('comparison_keeps_both_references_measurable_and_uses_paired_draws','score',"ci(values['window']-values['atom'],draws['window']-draws['atom'])","ci(values['atom']-values['window'],draws['window']-draws['atom'])",'Reverse the sign of the reported window-minus-atom estimate.'),
]


def run():
    logs=[]
    with tempfile.TemporaryDirectory(prefix='matched-window-tests-') as tmp:
        for i, row in enumerate([None,*PLANTS]):
            box=Path(tmp)/str(i);folder=box/'research/activation_transfer/atoms/matched_windows'
            folder.mkdir(parents=True)
            for f in HERE.glob('*.py'):shutil.copyfile(f,folder/f.name)
            selected=str(folder/'test_matched_windows.py'); env={**os.environ,'PYTHONPATH':os.pathsep.join(map(str,[box,ROOT,ROOT/'tsfm_lens'])),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            if row:
                name,module,before,after,reason=row; selected+='::test_'+name
                p=folder/(module+'.py');old=p.read_text();assert old.count(before)==1
                p.write_text(old.replace(before,after));assert p.read_bytes()!=(HERE/p.name).read_bytes()
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short',selected],env=env,cwd=box,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            summary=r.stdout.strip().splitlines()[-1]
            if row:
                assert r.returncode==1 and '1 failed' in summary and ('AssertionError' in r.stdout or re.search(r'^E\s+assert\b',r.stdout,re.M)),r.stdout
                logs.append('| `test_'+name+'` | '+reason+' | `'+summary+'` |')
            else:
                assert r.returncode==0 and '3 passed' in summary,r.stdout
                baseline=summary
    return '\n'.join(['# Verification of the matched-window tests','',f'The unchanged isolated suite reported `{baseline}`. Each regression was applied to a fresh temporary copy with a unique source match and a verified byte change. The runner required the targeted assertion to fail and checked the pytest exit and summary. [Replay script](mutation_check.py)','','| Test | Planted regression | Pytest summary |','|---|---|---|',*logs,''])


if __name__=='__main__':
    print(run())
