"""Regression plants distinguishing actual selected edits from full-FFN reach."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS=[
    ('wrong_edit_norm','norm=float(np.linalg.norm(actual))','norm=float(np.linalg.norm(clean))',
     'test_zero_or_dead_selected_edit_does_not_inherit_full_FFN_reach'),
    ('full_reach_substituted',"output_reach_gt_1e6=relative>1e-6",'output_reach_gt_1e6=True',
     'test_zero_or_dead_selected_edit_does_not_inherit_full_FFN_reach'),
    ('unselected_support_bypass','if not np.array_equal(patched[...,~mask],clean[...,~mask]):','if False:',
     'test_selected_support_rejects_collateral_or_wrong_reset')]


def run():
    """Each corrupted gate must yield a test failure without collection errors."""
    for name,before,after,test in PLANTS:
        code=f"""
import pathlib,pytest
from experiments.feature_search import ffn_selectivity_causal_audit as module
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
exec(compile(source.replace({before!r},{after!r}),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_causal_audit.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ,**{k:'1' for k in THREAD_VARS}),capture_output=True,text=True,timeout=60)
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Plant not detected: '+name)
        print(name+': detected',flush=True)
    print('3/3 selected-edit audit plants detected',flush=True)


if __name__=='__main__':
    run()
