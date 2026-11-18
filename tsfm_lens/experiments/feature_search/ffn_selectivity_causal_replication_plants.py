"""Regression plants for crossed-control replication; no cached native loads."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS=[
    ('one_control_instead_of_mean','mean_control=controls.mean(axis=1)','mean_control=controls[:,0]',
     'test_crossed_control_mean_uses_sources_not_control_replicates'),
    ('mean_only_equivalence',"interval[0]>=-.02 and interval[1]<=.02","result[name]['mean']>=-.02 and result[name]['mean']<=.02",
     'test_practical_bound_requires_entire_CI_and_never_changes_primary'),
    ('wrong_control_sets',"controls=matching['draws'][:4]","controls=matching['draws'][4:]",
     'test_prepare_freezes32fresh1088cells_and_first4_controls_every_source'),
    ('phase_family_confound','indices=causal.condition_indices(i//4)','indices=causal.condition_indices(i)',
     'test_prepare_freezes32fresh1088cells_and_first4_controls_every_source'),
    ('old_source_seeds','SEED=4964000','SEED=4963000',
     'test_prepare_freezes32fresh1088cells_and_first4_controls_every_source'),
    ('missing_fourth_control',"zip(ARMS,[controls['selected']]+controls['controls'])","zip(ARMS,[controls['selected']]+controls['controls'][:3])",
     'test_toy_crossed_worker_saves_every_arm_and_float64_remaining_energy'),
    ('float32_energy_regression',"values=causal.response_metrics(data['clean_forecasts'][i,0].astype(float),data['clean_forecasts'][i,c+1].astype(float),\n                data['patched_forecasts'][j].astype(float))",
     "values=causal.response_metrics(data['clean_forecasts'][i,0],data['clean_forecasts'][i,c+1],\n                data['patched_forecasts'][j])",
     'test_toy_crossed_worker_saves_every_arm_and_float64_remaining_energy'),
    ('hold_bypass','if not slot_granted:','if False:',
     'test_replication_hold_wrong_model_and_cell_limit_prevent_native'),
    ('wrong_model_allowed',"if p['model']!='bolt' or p['cells']!=CELLS or p['cells']>1200:","if p['cells']!=CELLS or p['cells']>1200:",
     'test_replication_hold_wrong_model_and_cell_limit_prevent_native')]


def run():
    """Every corrupted implementation must fail its specified test, not collection."""
    for name,before,after,test in PLANTS:
        code=f"""
import pathlib,pytest
from experiments.feature_search import ffn_selectivity_causal_replication as module
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1
exec(compile(source.replace({before!r},{after!r}),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_causal_replication.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=60,
            env=dict(os.environ,**{k:'1' for k in THREAD_VARS},PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError('Expected plant failure missing: '+name)
        print(name+': detected',flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} replication plants detected',flush=True)


if __name__=='__main__':
    run()
