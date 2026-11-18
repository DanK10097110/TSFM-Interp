"""Independent audit and local weight-energy regression plants, no native calls."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.natural_validation_study import THREAD_VARS


def run():
    """Each planted arithmetic/control failure must trigger the designated test."""
    plants=[('response_sign','ffn_selectivity_causal_replication_audit','change=prediction-changed','change=changed-prediction',
             'test_independent_event_reconstruction_detects_collateral_and_response_sign'),
            ('CI_gate_bypass','ffn_selectivity_causal_replication_audit','elif not np.allclose(actual[key],reported[key],rtol=1e-10,atol=1e-12):','elif False:',
             'test_independent_source_CI_rejects_token_counts_and_tampering'),
            ('ignored_projection_weight','ffn_selectivity_downprojection_diagnostic','return float(np.sum(value**2))','return float(np.sum(removed**2))',
             'test_equal_input_activity_does_not_imply_equal_projected_energy')]
    for name,module,before,after,test in plants:
        code=f"""
import pathlib,pytest,importlib
module=importlib.import_module('experiments.feature_search.{module}')
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
exec(compile(source.replace({before!r},{after!r}),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_causal_replication_audit.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=60,
            env=dict(os.environ,**{k:'1' for k in THREAD_VARS}))
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Plant not detected: '+name)
        print(name+': detected',flush=True)
    print('3/3 independent audit/weight plants detected',flush=True)


if __name__=='__main__':
    run()
