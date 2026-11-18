"""In-memory regression plants for FFN preparation, never cached native execution."""
from pathlib import Path
import os
import subprocess
import sys

from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS = [
    ('wrong_input_RMS','study','.1*x/np.sqrt(np.mean(x*x))','.2*x/np.sqrt(np.mean(x*x))',
     'test_equal_rms_phase_sign_variants_and_no_future_targets'),
    ('uncounted_reference','study','cells=len(rows)*(1+1+14+2+2)','cells=len(rows)*(1+14+2+2)',
     'test_prepare_freezes480_cells_no_sae_disjoint_groups_and_causal_hold'),
    ('future_read','study',"p = read(pilot/'protocol.json')","p = read(pilot/'protocol.json')\n    np.load(pilot/'targets.npz')",
     'test_reused_real_reads_only_past_roots_not_target_files'),
    ('nonorthogonal_rotation','analysis',"norm='ortho'","norm=None",
     'test_tokenwise_rotation_preserves_geometry_before_separate_pooling'),
    ('RMS_replaced_by_mean','analysis','else np.sqrt(np.mean(x*x,axis=2))','else abs(x.mean(axis=2))',
     'test_tokenwise_rotation_preserves_geometry_before_separate_pooling'),
    ('TEST_fitted_PCA','analysis','np.asarray(tokens,float)[np.asarray(fit,bool)]','np.asarray(tokens,float)',
     'test_test_source_poison_cannot_change_pca_or_fit_feature_selection'),
    ('TEST_fitted_features','analysis','np.asarray(features,float)[np.asarray(fit,bool)]','np.asarray(features,float)',
     'test_test_source_poison_cannot_change_pca_or_fit_feature_selection'),
    ('variant_weight_bias','analysis','1/frequency[c[\'kind\']]','1.',
     'test_pca_type_weights_do_not_reward_duplicate_phase_exposures'),
    ('purity_equals_accuracy','analysis','purity=purity,energy_coverage=coverage','purity=accuracy,energy_coverage=coverage',
     'test_readout_accuracy_is_not_the_same_as_unit_type_purity'),
    ('FIT_in_bootstrap','analysis','ix=np.asarray(indices,int)','ix=np.arange(len(rows))',
     'test_background_bootstrap_uses_only_heldout_source_groups'),
    ('exact_gate_bypass','study','if not np.array_equal(reference,actual):','if False:',
     'test_postactivation_capture_exact_replay_reach_and_normalizer_restoration'),
    ('reach_gate_bypass','study','if value<=1e-6:','if False:',
     'test_postactivation_capture_exact_replay_reach_and_normalizer_restoration'),
    ('wrong_FFN_capture','study',"observed[f'{tag}_ffn'].append(value.detach().cpu().numpy().copy())",
     "observed[f'{tag}_ffn'].append(value[..., :4].detach().cpu().numpy().copy())",
     'test_postactivation_capture_exact_replay_reach_and_normalizer_restoration'),
    ('hidden_reference_pass','study','            reference=invoke(anchor)\n        finally:',
     '            reference=invoke(anchor)\n            invoke(anchor)\n        finally:',
     'test_worker_counts_all_native_cells_and_keeps_causal_stage_out'),
    ('hold_bypass','run','if not slot_granted:','if False:',
     'test_native_hold_and_cellcap_are_enforced_without_model_loads'),
    ('cellcap_bypass','run',"if p['cells_per_model']>500:",'if False:',
     'test_native_hold_and_cellcap_are_enforced_without_model_loads'),
]


def run():
    """Each plant must cause a test assertion failure, never a collection error."""
    root=Path(__file__).resolve().parents[2]
    env=dict(os.environ,**{k:'1' for k in THREAD_VARS},PYTHONDONTWRITEBYTECODE='1')
    for name,suffix,before,after,test in PLANTS:
        code=f"""
import importlib
import pathlib
import pytest
module=importlib.import_module('experiments.feature_search.ffn_selectivity_{suffix}')
source=pathlib.Path(module.__file__).read_text()
assert source.count({before!r})==1, 'Plant site must be unique'
changed=source.replace({before!r},{after!r})
assert changed!=source
exec(compile(changed,module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_ffn_selectivity_study.py::{test}']))
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=60)
        if result.returncode!=1 or '1 failed' not in result.stdout:
            print(result.stdout,result.stderr)
            raise AssertionError(f'Expected failure not observed: {name}')
        print(f'{name}: detected',flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} plants detected; no cached native models loaded',flush=True)


if __name__=='__main__':
    run()
