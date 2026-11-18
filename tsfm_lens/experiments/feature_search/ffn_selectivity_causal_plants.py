"""Causal-protocol regression plants; only small toy model execution is allowed."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS=[
    ('TEST_activity_leak','', 'np.asarray(raw)[np.asarray(fit,bool)]','np.asarray(raw)',
     'test_FIT_activity_matching_ignores_TEST_poison_and_variant_repetition'),
    ('selected_in_random_pool','', 'valid[selected]=False','valid[selected]=True',
     'test_FIT_activity_matching_ignores_TEST_poison_and_variant_repetition'),
    ('unit_zeroing_instead_of_delta','', 'result[...,units]=np.asarray(background)[...,units]','result[...,units]=0',
     'test_unit_delta_patch_preserves_unselected_coordinates_and_baseline'),
    ('wrong_response_direction','', 'removed=np.asarray(clean,float)-patched','removed=np.asarray(patched,float)-clean',
     'test_response_removal_distinguishes_direction_energy_offtarget_and_amplification'),
    ('wrong_orthogonal_energy','', 'np.mean((removed-projection*response)**2)/energy','np.mean(removed**2)/energy',
     'test_response_removal_distinguishes_direction_energy_offtarget_and_amplification'),
    ('offtarget_in_primary','_analysis',"if record['kind'] in kinds",'if True',
     'test_source_bootstrap_exclusions_and_equal_type_aggregation'),
    ('fresh_source_reuse','', 'seed=SEED+j;source_id=f\'syn:{seed}\'', 'seed=4961008+j;source_id=f\'syn:{seed}\'',
     'test_preparation_freezes384_cells_balanced_fresh_sources_and_no_native'),
    ('wrong_cell_budget','', 'cells=sum(9+10*len(r[\'budgets\']) for r in new_rows)', 'cells=sum(8+10*len(r[\'budgets\']) for r in new_rows)',
     'test_preparation_freezes384_cells_balanced_fresh_sources_and_no_native'),
    ('parameter_gate_bypass','', "if before_hash!=after_hash or count!=p['cells']:","if count!=p['cells']:",
     'test_toy_worker_exact_delta_support_parameter_hash_and_saved_analysis'),
    ('hidden_native_pass','', 'baseline,states=invoke(q[:1])','baseline,states=invoke(q[:1])\n            invoke(q[:1])',
     'test_toy_worker_exact_delta_support_parameter_hash_and_saved_analysis'),
    ('hold_bypass','_run','if not slot_granted:','if False:',
     'test_hold_and_Bolt_only_budget_gate_prevent_subprocess'),
    ('wrong_model_allowed','_run',"if p['model']!='bolt' or p['cells']!=CELL_CAP:","if p['cells']!=CELL_CAP:",
     'test_hold_and_Bolt_only_budget_gate_prevent_subprocess')]


def run():
    """Every mutation must trigger a failed test, not an import/collection error."""
    root=Path(__file__).resolve().parents[2]
    for name,suffix,before,after,test in PLANTS:
        code=f"""
import pathlib,importlib,pytest
module=importlib.import_module('experiments.feature_search.ffn_selectivity_causal{suffix}')
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
changed=source.replace({before!r},{after!r})
assert changed!=source
exec(compile(changed,module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_causal.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,timeout=60,
            env=dict(os.environ,**{k:'1' for k in THREAD_VARS},PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Expected test failure not detected: '+name)
        print(name+': detected',flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} causal plants detected; no cached native model loads',flush=True)


if __name__=='__main__':
    run()
