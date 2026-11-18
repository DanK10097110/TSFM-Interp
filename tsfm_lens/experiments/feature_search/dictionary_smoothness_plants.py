"""Known-regression tests for saved-only dictionary smoothness instrumentation."""
from pathlib import Path
import os
import subprocess
import sys

PLANTS=[
    ('fixed_PC_becomes_adaptive',"if method=='identity':","if method=='pca_fixed': method='pca_adaptive'\n    if method=='identity':",
     'test_fixed_first_PC_is_linear_but_adaptive_signed_topk_has_boundary_chord'),
    ('actual_midpoint_replaced','actual=g[0]+g[1]-2*g[2]','actual=g[0]+g[1]-2*g[3]',
     'test_actual_midpoint_is_not_replaced_by_offmanifold_activation_midpoint'),
    ('refloor_ignored','np.maximum(denominator,multiplier*np.asarray(floor))','np.maximum(denominator,np.asarray(floor))',
     'test_refloor_controls_denominator_artifact_and_keeps_all_values'),
    ('decoder_bias_dropped',"output+=np.asarray(artifact['b_dec'],dtype=dtype)","output+=0",
     'test_positive_tie_zero_shortfall_and_frozen_learned_replay'),
    ('path_pseudoreplication',"mean=float(per_source.mean())","mean=float(np.mean(np.asarray(values)[sum(ids,[])]))",
     'test_source_aggregation_does_not_weight_repeated_paths_as_sources'),
    ('precision_control_disabled','h=np.asarray(h,dtype=dtype)','h=np.asarray(h,dtype=np.float64)',
     'test_FP32_FP64_precision_is_measured_not_native_precision_claim')]


def run():
    """Each mutant must fail its known-answer test, not collection/import."""
    root=Path(__file__).resolve().parents[2]
    for name,before,after,test in PLANTS:
        code=f"""
import pathlib,pytest
from experiments.feature_search import dictionary_smoothness_math as module
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
exec(compile(source.replace({before!r},{after!r}),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_dictionary_smoothness_study.py::{test}']))
"""
        env=dict(os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')})
        result=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=30)
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Plant was not detected: '+name)
        print(name+': detected',flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} smoothness plants detected',flush=True)


if __name__=='__main__':
    run()
