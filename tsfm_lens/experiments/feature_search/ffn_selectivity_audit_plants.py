"""In-memory audit regression plants; no native model construction."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS=[
    ('score_gate_bypass','if not np.allclose(value,saved,atol=2e-8,rtol=2e-6):','if False:',
     'test_audit_detects_score_and_feature_selection_tampering'),
    ('FIT_selection_gate_bypass',"if chosen.tolist()!=rule['indices']:",'if False:',
     'test_audit_detects_score_and_feature_selection_tampering'),
    ('TEST_PCA_leak','fit_data=delta[fit]','fit_data=delta',
     'test_audit_pca_ignores_TEST_and_balances_duplicate_variants'),
    ('variant_weight_bias','weights.extend([1/count]*delta.shape[2])','weights.extend([1.]*delta.shape[2])',
     'test_audit_pca_ignores_TEST_and_balances_duplicate_variants'),
    ('wrong_RMS_pool','else np.sqrt(np.sum(delta**2,axis=2)/delta.shape[2])','else np.abs(np.sum(delta,axis=2)/delta.shape[2])',
     'test_audit_pooling_distinguishes_canceling_tokens_and_rotation_geometry'),
    ('nonorthogonal_rotation',"norm='ortho'",'norm=None',
     'test_audit_pooling_distinguishes_canceling_tokens_and_rotation_geometry'),
    ('source_identity_bypass','if actual.get(key)!=saved.get(key):','if False:',
     'test_audit_bootstrap_rejects_source_and_CI_tampering'),
    ('CI_bypass',"elif not np.allclose(actual[key],saved[key],atol=1e-12,rtol=1e-10):",'elif False:',
     'test_audit_bootstrap_rejects_source_and_CI_tampering')]


def run():
    """Require a failed test, not a collection error, for every corrupted gate."""
    root=Path(__file__).resolve().parents[2]
    for name,before,after,test in PLANTS:
        code=f"""
import pathlib
import pytest
from experiments.feature_search import ffn_selectivity_audit as module
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
changed=source.replace({before!r},{after!r})
assert changed!=source
exec(compile(changed,module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_audit.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=root,env=dict(os.environ,**{k:'1' for k in THREAD_VARS}),
            capture_output=True,text=True,timeout=60)
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Plant not detected: '+name)
        print(name+': detected',flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} audit plants detected',flush=True)


if __name__=='__main__':
    run()
