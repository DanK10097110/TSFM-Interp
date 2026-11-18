"""Two toy-only regression plants for the explicit tighter execution cap."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.natural_validation_study import THREAD_VARS


def run():
    """Require assertion failures when the deadline or hold is bypassed."""
    plants=[('worker_deadline','study.SECONDS=SECONDS','study.SECONDS=900','test_amendment_requires300_and_sets_internal_worker_deadline'),
            ('hold_bypass','if not slot_granted:','if False:','test_amended_dispatcher_holds_without_grant')]
    for name,before,after,test in plants:
        code=f"""
import pathlib,pytest
from experiments.feature_search import ffn_selectivity_causal_replication_300s as module
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
exec(compile(source.replace({before!r},{after!r}),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_causal_replication_300s.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=60,
            env=dict(os.environ,**{k:'1' for k in THREAD_VARS}))
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Plant not detected: '+name)
        print(name+': detected',flush=True)
    print('2/2 amended-cap plants detected',flush=True)


if __name__=='__main__':
    run()
