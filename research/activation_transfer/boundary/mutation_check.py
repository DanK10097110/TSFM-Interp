"""Run each research-test mutation against an isolated, fresh copy of the Python module set."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
# Test ordinal, file, literal before, literal after, and the observable regression.
PLANTS=[
(1,'analyze','decoded = dz @ sae.W_dec','decoded = dzlocked @ sae.W_dec','Decode the ordinary edit with the anchored support.'),
(1,'analyze','torch.relu(pre1) * (z0 > 0)','torch.relu(pre1) * (z1 > 0)','Use the edited support in the anchored calculation.'),
(2,'smooth','r*torch.clamp(distance/width, max=1)','r*(distance>0)','Restore the hard jump at the shoulder boundary.'),
(3,'smooth','(shoulder_end-floor).clamp_min(1e-8)','(shoulder_end-floor)','Divide by a zero-width shoulder for a tied/empty row.'),
(4,'analyze',') + self.mean_y',')','Omit the learned target intercept from a ridge forecast.'),
(4,'analyze','"cosine": dot/np.maximum','"cosine": np.abs(dot)/np.maximum','Lose the sign of cosine.'),
(4,'analyze','"n_items": len(values)','"n_items": np.asarray(item_scores).size','Count both cue measurements as separate bootstrap items.'),
(5,'patch_forecast','selected = valid[positions]','selected = positions','Ignore the adapter’s live-token offset.'),
(6,'capture','np.sqrt(np.mean(curve**2))','np.sqrt(np.sum(curve**2))','Normalize to unit L2 instead of unit RMS.'),
(6,'capture','f"cue|{dataset}|{item}"','f"cue|{dataset}|{i}"','Tie edit shape to row order.'),
(6,'capture','curve = curve * np.hanning(t.size)','curve = curve','Remove the endpoint taper.'),
(7,'audit_forecasts','"energy_ratio": pred_e/np.maximum(true_e,1e-24)','"energy_ratio": np.sqrt(pred_e/np.maximum(true_e,1e-24))','Return an amplitude ratio in the energy field.'),
(8,'gate_controls','return hard * (torch.linalg.vector_norm(soft,dim=-1,keepdim=True)/denom)','return soft','Use the smooth coefficients as the norm-matched hard control.'),
(9,'analyze','ntrain = SPLITS["train"][1]\n    endval','ntrain = len(source)\n    endval','Fit on evaluation labels as well as calibration labels.'),
(10,'fixed','-clean_codes.mean(axis=0), -active','clean_codes.mean(axis=0), -active','Break prevalence ties toward the smallest mean activation.'),
(10,'fixed','np.linalg.norm(matrix, axis=0)','np.linalg.norm(matrix, axis=1)','Normalize rows instead of the chosen encoder directions.'),
(11,'seed_identity','linear_sum_assignment(similarity,maximize=True)','linear_sum_assignment(similarity,maximize=False)','Assign to the least similar decoder directions.'),
(11,'seed_identity','z[...,b]=source[...,a]*scale','z[...,a]=source[...,a]*scale','Apply a fitted atom permutation to the wrong output coordinate.'),
(12,'fixed_geometry','return q.astype(np.float32),r.astype(np.float32)','return q.astype(np.float32),np.eye(r.shape[0],dtype=np.float32)','Lose the learned Gram geometry by replacing the QR scale factor.'),
(13,'crossing','np.argmax(switched[keep],axis=1)','1-np.argmax(switched[keep,::-1],axis=1)','Prefer the sine when both pulses and sines cross.'),
(13,'crossing','(.03*t[:,None]).astype','(.10*t[:,None]).astype','Apply the maximum dose while bracketing the primary dose.'),
(14,'seed_translation','pred=reg.predict(projected,penalty)-reg.mean_y','pred=reg.predict(projected,penalty)','Add the clean-state offset to a displacement.'),
(15,'decoder_refit','(bias+model.mean_y-model.mean_x@change)','(bias-model.mean_x@change)','Forget the calibration mean residual in the new decoder bias.'),
]


def main(out):
    node=ast.parse((HERE/'test_boundary.py').read_text())
    tests=[f.name for f in node.body if isinstance(f,ast.FunctionDef) and f.name.startswith('test_')]
    if {p[0] for p in PLANTS}!=set(range(1,len(tests)+1)):
        raise ValueError('a test lacks a targeted plant')
    with tempfile.TemporaryDirectory(prefix='tsfm-boundary-mutations-') as tmp:
        def execute(i,plant=None):
            box=Path(tmp)/str(i)
            folder=box/'research/activation_transfer/boundary'
            folder.mkdir(parents=True)
            for src in HERE.glob('*.py'):
                shutil.copyfile(src,folder/src.name)
            env={**os.environ,'PYTHONPATH':str(box)+os.pathsep+str(ROOT/'tsfm_lens'),
                 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'PYTHONDONTWRITEBYTECODE':'1'}
            selected=str(folder/'test_boundary.py')
            if plant:
                ix,filename,before,after,what=plant
                f=folder/(filename+'.py')
                s=f.read_text()
                if s.count(before)!=1:
                    raise ValueError(f'{what} matches {s.count(before)} source locations')
                f.write_text(s.replace(before,after))
                assert hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256((HERE/f.name).read_bytes()).digest()
                selected+='::'+tests[ix-1]
            proc=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short',selected],cwd=box,env=env,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            lines=proc.stdout.strip().splitlines()
            if not plant:
                if proc.returncode or not re.search(rf'\b{len(tests)} passed\b',lines[-1]):
                    raise RuntimeError(proc.stdout)
                return lines[-1]
            assertion=('AssertionError' in proc.stdout or re.search(r'^E\s+assert\b',proc.stdout,re.MULTILINE))
            if proc.returncode!=1 or not re.search(r'\b1 failed\b',lines[-1]) or not assertion:
                raise RuntimeError(what+'\n'+proc.stdout)
            return (tests[ix-1],filename+'.py',what,lines[-1])
        baseline=execute(0)
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows=list(pool.map(lambda i:execute(i+1,PLANTS[i]),range(len(PLANTS))))
    lines=['# Boundary experiment test verification','',
           'Each row was executed in a fresh temporary-directory copy. The original source and test files were preserved. The runner required one unique source replacement, verified a changed file SHA-256, targeted the named test, and accepted only a pytest assertion failure with exit code 1. Collection errors, imports, missing plants and successful tests fail this check. The exact old and new source expressions are in [the replay script](mutation_check.py).','',
           'The unmodified isolated suite reported **'+baseline+'**. The '+str(len(rows))+' planted runs covered all '+str(len(tests))+' new tests.','',
           '| Target test | Changed file | Planted regression | Pytest summary |','|---|---|---|---|']
    lines.extend('| '+ ' | '.join(['`'+t+'`','`'+f+'`',what,'`'+summary+'`'])+' |' for t,f,what,summary in rows)
    lines+=['','Run again with the study Python environment.','',
            '```bash','python research/activation_transfer/boundary/mutation_check.py --out /tmp/boundary-mutation-check.md','```','']
    out.write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
