"""Run targeted in-memory regressions; never edit shared modules or load models."""
from pathlib import Path
import os
import subprocess
import sys

from experiments.feature_search.natural_validation_study import THREAD_VARS


PLANTS = [
    ('later_normalizer', 'study', 'normalize_history(value[origin-128:origin])',
     'normalize_history(value[128:256])', 'test_origins_normalize_only_own_history_and_exact_targets'),
    ('inherited_scale', 'study', 'return (x-center)/scale, center, scale, valid',
     'return x, center, scale, valid', 'test_helper_last_history_scale_is_removed_and_future_poison_is_isolated'),
    ('missing_zero_anchor', 'study', "forecasts[ids].sum(0)-depth*forecasts[plan['zero']]",
     'forecasts[ids].sum(0)', 'test_composition_keeps_dc_and_zero_anchor_and_tone_last_baseline'),
    ('forced_constant_eligibility', 'study', 'selector_usable=valid', 'selector_usable=True',
     'test_constant_hindcast_abstains_without_dropping_main_source'),
    ('roundoff_constant_scale', 'study', 'valid = bool(np.ptp(x) > 0 and scale > 0)', 'valid = scale > 0',
     'test_constant_hindcast_abstains_without_dropping_main_source'),
    ('ties_hybrid', 'analysis', 'loss[:, :, idx].mean(axis=1) < loss[:, :, 0].mean(axis=1)',
     'loss[:, :, idx].mean(axis=1) <= loss[:, :, 0].mean(axis=1)',
     'test_selector_requires_two_losses_ties_native_and_usable_history'),
    ('one_hindcast_only', 'analysis', 'loss[:, :, idx].mean(axis=1) < loss[:, :, 0].mean(axis=1)',
     'loss[:, 0, idx] < loss[:, 0, 0]', 'test_selector_requires_two_losses_ties_native_and_usable_history'),
    ('worker_reads_future', 'study', "p = read(out/'protocol.json')\n    for path, digest",
     "p = read(out/'protocol.json')\n    np.load(out/'TEST_targets.npz')\n    for path, digest",
     'test_worker_never_opens_targets_and_slot_is_required'),
    ('slot_bypass', 'study', 'if slot_granted is not True:', 'if False:',
     'test_worker_never_opens_targets_and_slot_is_required'),
    ('premature_TEST_read', 'analysis', "fit = bound_load(out, p, 'FIT_targets.npz')",
     "bound_load(out, p, 'TEST_targets.npz')\n    fit = bound_load(out, p, 'FIT_targets.npz')",
     'test_end_to_end_future_poison_preserves_choices_and_fit_rules'),
    ('peak_order_matching', 'analysis', 'ia, ib = linear_sum_assignment(cost)',
     'ia = ib = np.arange(min(cost.shape))', 'test_stability_matches_swapped_peaks_and_records_abstention'),
    ('source_reuse', 'study', 'if previous.intersection(ids):', 'if False:',
     'test_prepare_checks_fresh_source_ids_and_freezes_32_sources'),
    ('forecast_hash_bypass', 'analysis', "if worker_hashes[model] != record['sha256']:", 'if False:',
     'test_hash_and_execution_gates_reject_planted_damage[forecast]'),
    ('input_hash_bypass', 'analysis', "if sha(out/name) != p['hashes'][name]:", 'if False:',
     'test_hash_and_execution_gates_reject_planted_damage[input]'),
    ('execution_bypass', 'analysis', "not execution[model]['valid']", 'False',
     'test_hash_and_execution_gates_reject_planted_damage[invalid_execution]'),
    ('gain_sign', 'analysis', 'gain = native-other', 'gain = other-native',
     'test_source_bootstrap_counts_ties_and_has_known_gain'),
    ('resource_gate_bypass', 'study', "valid = valid and record.get('peak_rss', float('inf')) <= p['worker_cap']",
     'valid = valid', 'test_monitor_serializes_models_and_rejects_resource_overrun[True]'),
    ('missing_second_model', 'study', 'for model in MODELS:', "for model in ('bolt',):",
     'test_monitor_serializes_models_and_rejects_resource_overrun[False]'),
    ('lost_failure_history', 'study', 'events.append(entry)', 'events = [entry]',
     'test_knowledge_retains_failed_attempt_and_exact_analysis'),
]


def run():
    """Require assertion failures, not collection errors, for every planted bug."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, **{k: '1' for k in THREAD_VARS}, PYTHONDONTWRITEBYTECODE='1')
    for name, module, before, after, test in PLANTS:
        code = f"""
import importlib
import pathlib
import pytest
module = importlib.import_module('experiments.feature_search.natural_validation_{module}')
text = pathlib.Path(module.__file__).read_text()
assert text.count({before!r}) == 1, 'Plant must change exactly one site'
changed = text.replace({before!r}, {after!r})
assert changed != text
exec(compile(changed, module.__file__, 'exec'), module.__dict__)
raise SystemExit(pytest.main(['-q', 'tests/test_natural_validation_study.py::{test}']))
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 1 or '1 failed' not in result.stdout:
            print(result.stdout, result.stderr)
            raise AssertionError(f'Plant did not trigger expected test failure: {name}')
        print(f'{name}: detected by {test}', flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} regression plants detected; no native models loaded', flush=True)


if __name__ == '__main__':
    run()
