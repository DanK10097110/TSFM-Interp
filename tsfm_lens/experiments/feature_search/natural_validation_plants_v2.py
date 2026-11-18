"""In-memory regression plants for the availability amendment; no model loads."""
from pathlib import Path
import os
import subprocess
import sys

from experiments.feature_search.natural_validation_study import THREAD_VARS

PLANTS = [
    ('wrong_domain_scope', 'v2', 'DOMAINS=(domain,)', 'DOMAINS=DOMAINS',
     'test_domain_helper_is_scoped_without_mutating_shared_globals'),
    ('ignoring_remaining_count', 'v2', 'count = min(requested, int(match.group(1))-64)', 'count = requested',
     'test_prospective_availability_uses_all_remaining_never_old_ranks'),
    ('swallowed_target_error', 'v2', 'if match is None:\n                raise', 'if match is None:\n                return [], {}',
     'test_target_error_or_fewer_than_two_never_substitutes[Selected target missing; refusing outcome-based replacement]'),
    ('accepting_one_remaining', 'v2', 'if count < 2:', 'if count < 1:',
     'test_target_error_or_fewer_than_two_never_substitutes[Only 65 eligible sources in monash_electricity_hourly]'),
    ('singleton_interval', 'analysis_v2', 'if len(gain) >= 2:', 'if len(gain) >= 1:',
     'test_singleton_has_no_ci_while_larger_sample_retains_descriptive_interval'),
    ('missing_amendment', 'v2', "write(out/'amendment.json', amendment)", 'pass',
     'test_amendment_precedes_data_and_preserves_failed_pilot'),
    ('premature_TEST_read', 'analysis_v2', "fit = bound_load(out, p, 'FIT_targets.npz')",
     "bound_load(out, p, 'TEST_targets.npz')\n    fit = bound_load(out, p, 'FIT_targets.npz')",
     'test_v2_future_poison_cannot_change_selectors_or_fit_rules'),
    ('changed_tie_rule', 'analysis_v2', 'loss[:, :, idx].mean(axis=1) < loss[:, :, 0].mean(axis=1)',
     'loss[:, :, idx].mean(axis=1) <= loss[:, :, 0].mean(axis=1)',
     'test_v2_selector_matches_frozen_original'),
]


def run():
    """Require a failing test for each isolated code regression."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, **{k: '1' for k in THREAD_VARS}, PYTHONDONTWRITEBYTECODE='1')
    for name, module, before, after, test in PLANTS:
        code = f"""
import importlib
import pathlib
import pytest
module = importlib.import_module('experiments.feature_search.natural_validation_{module}')
text = pathlib.Path(module.__file__).read_text()
assert text.count({before!r}) == 1
changed = text.replace({before!r}, {after!r})
assert changed != text
exec(compile(changed, module.__file__, 'exec'), module.__dict__)
raise SystemExit(pytest.main(['-q', 'tests/test_natural_validation_v2.py::{test}']))
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 1 or '1 failed' not in result.stdout:
            print(result.stdout, result.stderr)
            raise AssertionError(f'Plant was not detected by its test: {name}')
        print(f'{name}: detected', flush=True)
    print(f'{len(PLANTS)}/{len(PLANTS)} amendment plants detected', flush=True)


if __name__ == '__main__':
    run()
