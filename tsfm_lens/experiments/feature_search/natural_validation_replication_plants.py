"""In-memory regression tests for the frozen natural replication design."""
from pathlib import Path
import os
import subprocess
import sys

from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS = [
    ('future_in_availability', '', 'values.slice(len(values)-288, 256)', 'values.slice(len(values)-288, 288)',
     'test_availability_never_materializes_future_values_and_ignores_poison'),
    ('availability_gate_bypass', '', "if not all(audit[d]['sufficient'] for d in DOMAINS):", 'if False:',
     'test_short_availability_blocks_before_any_helper_future_access'),
    ('old_ranks', '', 'values = helper(cache, OFFSET, COUNT)', 'values = helper(cache, 64, COUNT)',
     'test_original_helper_scope_and_target_errors_do_not_replace_sources'),
    ('swallowed_future_error', '', 'values = helper(cache, OFFSET, COUNT)',
     'try:\n            values = helper(cache, OFFSET, COUNT)\n        except ValueError:\n            return []',
     'test_original_helper_scope_and_target_errors_do_not_replace_sources'),
    ('wrong_primary_depth', '', 'primary_depth=1,', 'primary_depth=3,',
     'test_preparation_freezes192_test_sources_primary6_and_secondary'),
    ('fresh_sources_reused_for_FIT', '', "row.update(index=len(rows), split='TEST')", "row.update(index=len(rows), split='FIT')",
     'test_preparation_freezes192_test_sources_primary6_and_secondary'),
    ('source_reuse', '', 'or previous.intersection(ids):', 'or False:',
     'test_reused_source_fails_and_persists_negative_knowledge'),
    ('lost_failed_instrumentation', '', "knowledge(out, 'prepare', 'failed',", "knowledge(out, 'prepare', 'complete',",
     'test_reused_source_fails_and_persists_negative_knowledge'),
    ('later_scale_leak', '', 'value[origin-128:origin]', 'value[128:256]',
     'test_origin_normalization_and_main_future_poison_isolation'),
    ('lost_last_residual', '', "continuation(plan['details'], plan['primary_depth'])+bank[plan['residual1']][-1]",
     "continuation(plan['details'], plan['primary_depth'])",
     'test_fixed_depth1_controls_and_frozen_secondary_composition'),
    ('unadjusted_primary_interval', '_analysis', 'tail = .05/(2*family)', 'tail = .05/2',
     'test_primary_uncertainty_is_source_paired_adjusted6_and_ties_explicit'),
    ('unadjusted_sign_family', '_analysis', 'p[order]*np.arange(6, 0, -1)', 'p[order]',
     'test_holm_keeps_six_and_undefined_diagnostics_in_family'),
    ('one_hindcast_diagnostic', '_analysis', 'mean = past.mean(axis=1)', 'mean = past[:, 0]',
     'test_continuous_diagnostic_known_positive_negative_and_temporal_decoy'),
    ('ties_hybrid', '_analysis', 'losses[:, :, 1].mean(axis=1) < losses[:, :, 0].mean(axis=1)',
     'losses[:, :, 1].mean(axis=1) <= losses[:, :, 0].mean(axis=1)',
     'test_selector_uses_both_hindcasts_ties_native_and_no_future_threshold'),
    ('worker_target_read', '', "p = read(out/'protocol.json')\n    for path, digest",
     "p = read(out/'protocol.json')\n    np.load(out/'main_targets.npz')\n    for path, digest",
     'test_worker_target_blindness_and_fresh_hold'),
    ('hold_bypass', '', 'if not slot_granted:', 'if False:',
     'test_worker_target_blindness_and_fresh_hold'),
    ('forecast_hash_bypass', '_analysis', "or worker['sha256'] != sha(out/f'{model}_forecasts.npz')", 'or False',
     'test_forecast_tampering_fails_before_main_target_read'),
    ('main_before_freeze', '_analysis', "past = bound_targets(out, p, 'hindcast_targets.npz')['targets']",
     "past = bound_targets(out, p, 'hindcast_targets.npz')['targets']\n    bound_targets(out, p, 'main_targets.npz')",
     'test_analysis_future_poison_cannot_change_frozen_choices_or_pilot_rules'),
    ('budget_bypass', '', 'within_900_seconds=2*nominal <= SECONDS', 'within_900_seconds=True',
     'test_budget_has_conservative_allowance_and_rejects_over15min'),
    ('memory_gate_bypass', '', "record.get('peak_rss', float('inf')) <= MEMORY", 'True',
     'test_monitor_serial_and_resource_rejection[True]'),
    ('missing_second_model', '', '            for model in MODELS:', "            for model in ('bolt',):",
     'test_monitor_serial_and_resource_rejection[False]'),
]


def run():
    """Each mutant must cause the specified test to fail, not collection to error."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, **{k:'1' for k in THREAD_VARS}, PYTHONDONTWRITEBYTECODE='1')
    for name, suffix, before, after, test in PLANTS:
        code = f"""
import importlib
import pathlib
import pytest
module = importlib.import_module('experiments.feature_search.natural_validation_replication{suffix}')
source = pathlib.Path(module.__file__).read_text()
assert source.count({before!r}) == 1, 'Plant site must be unique'
changed = source.replace({before!r}, {after!r})
assert changed != source
exec(compile(changed, module.__file__, 'exec'), module.__dict__)
raise SystemExit(pytest.main(['-q', 'tests/test_natural_validation_replication.py::{test}']))
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 1 or '1 failed' not in result.stdout:
            print(result.stdout, result.stderr)
            raise AssertionError(f'Expected test failure not observed: {name}')
        print(f'{name}: detected', flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} replication plants detected; no native model loads', flush=True)


if __name__ == '__main__':
    run()
