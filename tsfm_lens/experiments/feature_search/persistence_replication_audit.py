"""Plant replication faults in subprocess memory, keeping frozen sources intact."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from experiments.feature_search.persistence_protocol import ENV, read, sha, write

PLANTS = [
    ('protocol', 'elapsed = np.maximum(t+lead, 0)', 'elapsed = np.maximum(t+lead+1, 0)', 'generator_matches_pilot_and_hides_future_regime'),
    ('protocol', "return 'fit' if seed < 4953032 else 'test'", "return 'fit' if seed < 4953033 else 'test'", 'source_split_and_cue_grid_are_frozen'),
    ('protocol', 'candidate(x[-32:])', 'candidate(x[-64:])', 'recent_refit_ignores_old_prefix_and_preserves_dc'),
    ('protocol', 'return float(max(np.linalg.norm(recent-full), np.linalg.norm(recent-earlier))/', 'return float(0*max(np.linalg.norm(recent-full), np.linalg.norm(recent-earlier))/', 'recent_error_detects_observed_change_at_shared_phase_origin'),
    ('protocol', 'threshold is None or error <= threshold', 'threshold is None or True', 'veto_is_observation_only_and_cannot_force_extraction'),
    ('analysis', "if row['split'] == 'fit']", "if row['split'] in ['fit', 'test']]", 'fit_selector_ignores_adversarial_test_targets_and_breaks_ties_native'),
    ('worker', 'if not path.exists():', 'if False:', 'worker_cannot_bypass_slot_hold'),
    ('analysis', 'return adjusted.tolist()', 'return values.tolist()', 'holm_retains_complete_six_test_family'),
    ('analysis', 'n_sources=len(values)', 'n_sources=len(values)*40', 'statistics_count_sources_and_do_not_promote_ties'),
]


def audit(out):
    """Every changed plant must yield one assertion/test failure, not collection error."""
    folder = out/'mutation_audit'; folder.mkdir(exist_ok=False)
    here = Path(__file__).resolve().parent
    test = here.parents[1]/'tests/test_persistence_replication.py'
    records = []
    for part, old, new, test_name in PLANTS:
        path = here/f'persistence_replication_{part}.py'
        original = path.read_text(encoding='utf-8')
        if original.count(old) != 1 or old == new:
            raise ValueError('Ambiguous or inert mutation')
        mutant = original.replace(old, new)
        module = 'experiments.feature_search.persistence_replication_'+part
        code = (f'import importlib,pytest; m=importlib.import_module({module!r}); '
                f'exec(compile({mutant!r},m.__file__,"exec"),m.__dict__); '
                f'raise SystemExit(pytest.main(["-q","-p","no:cacheprovider",{str(test)+"::test_"+test_name!r}]))')
        proc = subprocess.run([sys.executable, '-c', code], env=dict(os.environ, **ENV),
                              text=True, capture_output=True, timeout=60)
        log = proc.stdout+proc.stderr
        (folder/f'{test_name}.log').write_text(log, encoding='utf-8')
        killed = proc.returncode == 1 and '1 failed' in log and 'ERROR collecting' not in log
        records.append(dict(test='test_'+test_name, old=old, new=new, changed=mutant != original,
                            returncode=proc.returncode, killed=killed, source_sha256=sha(path)))
        write(folder/'audit.json', records)
        if not killed:
            raise RuntimeError('Plant survived: '+test_name)
    knowledge = read(out/'knowledge.json')
    knowledge['prelaunch_verification'] = dict(planted_faults=len(records), caught=sum(r['killed'] for r in records),native_launched=False)
    write(out/'knowledge.json', knowledge)
    print('Killed', len(records), 'replication plants; native remains HOLD', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
