"""Run in-memory planted regressions against every dictionary test's main guard."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.feature_search import dictionary_core as core
from experiments.feature_search import dictionary_native as native
from experiments.feature_search import dictionary_pilot as pilot


def run():
    """No source files are modified; failed assertions prove each plant was caught."""
    path = Path(__file__).resolve().parents[2]/'tests/test_dictionary_pilot.py'
    spec = importlib.util.spec_from_file_location('dictionary_tests', path)
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    topk, basis, encode, hook = core.signed_topk, core.fit_basis, core.encode, native.expansion_hook
    splits = core.split_indices
    plants = [
        ('source_overlap', core, 'split_indices', lambda rows: (np.array([0]), np.array([1])), 'test_split_source_leakage'),
        ('uniform_token_weights', core, 'source_weights', lambda s: np.ones(len(s))/len(s), 'test_source_weighted_train_scaler'),
        ('drop_negative_coordinates', core, 'signed_topk', lambda x, k: topk(np.maximum(x, 0), k), 'test_signed_and_actual_active_budget'),
        ('rotation_is_identity', core, 'fit_basis', lambda x, w, m, s: np.eye(x.shape[1]), 'test_single_feature_native_rotation_selectivity'),
        ('pca_ignores_weights', core, 'fit_basis', lambda x, w, m, s: basis(x, np.ones(len(x))/len(x), m, s), 'test_pca_uses_weighted_fit_covariance'),
        ('nested_full_loss_only', core.NestedTopK, 'reconstruction_loss', lambda self, x: ((self(x)[0]-x)**2).mean(), 'test_nested_shared_code_and_tiny_budget'),
        ('inference_batch_dependence', core, 'encode', lambda x, a, counts=None: encode(x, a, counts)+len(x)-1, 'test_frozen_inference_batch_independence'),
        ('many_to_one_stability', core, 'stability', lambda *a: dict(matched=2, mean_abs_cosine=1.), 'test_stability_one_to_one_and_dead_exclusion'),
        ('ffn_replacement_ignored', native, 'expansion_hook', lambda m, c, replacement=None: hook(m, c), 'test_ffn_hook_is_projection_input_and_replays'),
        ('relaxed_identity', native, 'exact_gate', lambda *a: None, 'test_exact_native_gate'),
        ('row_pseudoreplication', core, 'source_summary', lambda v, r: {'d': dict(n_sources=len(r), mean=float(np.mean(v)))}, 'test_source_summary_not_row_count_and_receiver_selection'),
        ('missing_slot_gate', pilot, 'launch', lambda a: None, 'test_slot_gate_precedes_launch'),
        ('proc_exit_race', pilot, 'rss_bytes', lambda lines: int(next(l.split()[1] for l in lines if l.startswith('VmRSS:')))*1024, 'test_resource_monitor_exit_race'),
        ('metadata_missing_patch_statistics', pilot, 'normalizer_metadata', lambda x, m: (np.column_stack([x.mean(1), x.std(1)]), (x-x.mean(1)[:, None])/x.std(1)[:, None]), 'test_normalizer_metadata_and_normalized_targets'),
        ('metadata_omitted_from_prediction', core, 'conditional_probe', lambda x, y, w, xt, b, m, mt: (np.zeros(len(xt)), np.zeros(len(xt)), [1]), 'test_metadata_conditioned_feature_increment'),
        ('test_states_used_for_fit', core, 'split_indices', lambda r: (np.arange(len(r)), splits(r)[1]), 'test_tiny_stage_test_states_cannot_change_fitted_dictionary'),
    ]
    results = []
    for name, obj, attribute, bad, test in plants:
        caught = None
        with patch.object(obj, attribute, bad):
            try:
                getattr(tests, test)()
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                caught = type(error).__name__
        if caught is None:
            raise AssertionError('Plant escaped: '+name)
        results.append(dict(plant=name, test=test, caught=caught))
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = run()
    pilot.write(a.out/'mutation_audits.json', result)
    print(f'{len(result)} / {len(result)} planted regressions caught')
