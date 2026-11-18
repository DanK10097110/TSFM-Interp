"""Focused planted-regression audit; mutate in subprocess memory, never source.

Each plant must change its code and produce an actual pytest failure rather than
a collection error. Native source files stay byte-identical during and after it.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from experiments.feature_search.persistence_protocol import ENV, sha, write

PLANTS = [
    ('protocol', 'elapsed = np.maximum(t-onset, 0)', 'elapsed = np.maximum(t-onset+1, 0)', 'generator_changes_only_when_observable'),
    ('protocol', "split='fit' if j < 4 else 'test'", "split='fit' if j < 5 else 'test'", 'sources_never_split_worlds'),
    ('protocol', 'tone = continuation(detail, len(detail))', 'tone = continuation(detail, len(detail)) + history.mean()', 'plan_retains_level_in_residual_and_zero_really_means_zero'),
    ('protocol', 'if not np.array_equal(a, b):', 'if not np.allclose(a, b):', 'identity_gate_rejects_tiny_perturbations'),
    ('protocol', 'allow_nan=False', 'allow_nan=True', 'finite_artifacts_reject_nan'),
    ('analysis', 'weights = np.array([p]+[(1-p)/3]*3)', 'weights = np.array([.25]*4)', 'expected_loss_is_a_tradeoff_not_detection'),
    ('analysis', 'return loss(uncued_prediction, cued_target, scale)-loss(cued_prediction, cued_target, scale)', 'return loss(cued_prediction, cued_target, scale)-loss(uncued_prediction, cued_target, scale)', 'cue_gain_uses_same_target_and_correct_sign'),
    ('study', 'if path == root or root in path.parents:', 'if False:', 'target_guard_rejects_direct_and_symlink_open'),
    ('study', "if sha(out/'inference'/name) != digest:", 'if False:', 'frozen_inputs_cannot_change'),
    ('analysis', 'n_sources=len(values)', 'n_sources=len(values)*32', 'uncertainty_resamples_sources'),
]


def audit(out):
    """Write exact pytest evidence for each separately killed planted regression."""
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    test = here.parents[1]/'tests/test_persistence_pilot.py'
    records = []
    for module, old, new, test_name in PLANTS:
        name = 'experiments.feature_search.persistence_'+module
        source = here/f'persistence_{module}.py'
        original = source.read_text(encoding='utf-8')
        if original.count(old) != 1 or old == new:
            raise ValueError('Ambiguous or inert planted regression')
        mutated = original.replace(old, new)
        code = (f'import importlib,pytest; m=importlib.import_module({name!r}); '
                f'exec(compile({mutated!r},m.__file__,"exec"),m.__dict__); '
                f'raise SystemExit(pytest.main(["-q","-p","no:cacheprovider",{str(test)+"::test_"+test_name!r}]))')
        result = subprocess.run([sys.executable, '-c', code], env=dict(os.environ, **ENV),
                                capture_output=True, text=True, timeout=60)
        log = result.stdout+result.stderr
        (out/f'{test_name}.log').write_text(log, encoding='utf-8')
        killed = result.returncode == 1 and '1 failed' in log and 'ERROR collecting' not in log
        records.append(dict(test='test_'+test_name, module=name, old=old, new=new,
                            changed=mutated != original, returncode=result.returncode, killed=killed,
                            source_sha256=sha(source)))
        if not killed:
            write(out/'mutation_audit.json', records)
            raise RuntimeError('Plant not killed: '+test_name)
    write(out/'mutation_audit.json', records)
    print('Killed', len(records), 'planted regressions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
