"""Small regression plants for the same-weight saved-only gate controls."""
from pathlib import Path
import os
import subprocess
import sys
from experiments.feature_search.coherence_study import write

PLANTS = [
    ('wrong_midpoint', 'central = pre[2]', 'central = pre[3]',
     'test_actual_midpoint_support_not_activation_midpoint_or_first_k'),
    ('affine_reclips_relu', 'values = values * np.broadcast_to(gate,', 'values = np.maximum(values, 0) * np.broadcast_to(gate,',
     'test_preserved_relu_and_frozen_relu_are_distinct_with_zero_tie'),
    ('response_uses_states', 'decoded_response - raw_response', 'g[:2] - h[:2]',
     'test_smoothing_can_worsen_state_fidelity_while_improving_response'),
    ('source_pseudoreplication', 'mean=float(np.mean(means))', 'mean=float(np.mean(values))',
     'test_bias_scale_dense_replay_and_equal_source_aggregation'),
]


def run():
    """Require actual assertion failure, and preserve every plant outcome in child run."""
    root = Path(__file__).resolve().parents[2]
    result_file = root / 'runs/dictionary_smoothness_fixed_support4965100/plants.json'
    assert not result_file.exists()
    results = []
    for name, before, after, test in PLANTS:
        code = f"""
import pathlib, pytest
from experiments.feature_search import dictionary_smoothness_fixed_support_math as module
source=pathlib.Path(module.__file__).read_text(encoding='utf-8')
assert source.count({before!r})==1
exec(compile(source.replace({before!r},{after!r}),module.__file__,'exec'),module.__dict__)
raise SystemExit(pytest.main(['-q','tests/test_dictionary_smoothness_fixed_support.py::{test}']))
"""
        env = dict(os.environ, **{k:'1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')})
        result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env, capture_output=True, text=True, timeout=30)
        detected = result.returncode == 1 and '1 failed' in result.stdout
        results.append(dict(plant=name, test=test, detected=detected, returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        write(result_file, dict(valid=all(r['detected'] for r in results), complete=len(results)==4, results=results))
        assert detected, results[-1]
        print(name + ': detected', flush=True)


if __name__ == '__main__':
    run()
