"""Run isolated in-memory planted regressions without changing frozen sources."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiments.feature_search.persistence_protocol import read, write
from experiments.feature_search import postrelease_weather_data as data


def audit(out):
    """Demand assertion failures, not import errors, for each documented plant."""
    plants = [
        ('accept_bad_qc', "row['temperature_Quality_Code'] not in (None, '')", 'False', 'test_qc_units_and_identity'),
        ('wrong_units', 'value = float(value)', 'value = float(value)/10', 'test_qc_units_and_identity'),
        ('wrong_station', "row['STATION'] != station", 'False', 'test_qc_units_and_identity'),
        ('duplicate_weight', 'defaultdict(set)', 'defaultdict(list)', 'test_duplicate_weight_and_boundary'),
        ('future_fill', 'missing = ~np.isfinite(history)', 'future = np.nan_to_num(future)\n    missing = ~np.isfinite(history)', 'test_causal_history_and_untouched_future'),
        ('too_many_gaps', 'missing.sum() <= 2', 'missing.sum() <= 3', 'test_causal_history_and_untouched_future'),
        ('leading_fill', 'and not missing[0]', '', 'test_causal_history_and_untouched_future'),
        ('future_leak', 'history[i] = history[i-1]', 'history[i] = future[0]', 'test_causal_history_and_untouched_future'),
    ]
    source = Path(data.__file__).read_text(encoding='utf-8')
    results = []
    for name, old, new, test in plants:
        assert source.count(old) == 1
        mutant = source.replace(old, new)
        if name == 'duplicate_weight':
            mutant = mutant.replace('.add(value)', '.append(value)')
        assert mutant != source
        code = 'from experiments.feature_search import postrelease_weather_data as m\nimport pytest\nexec(compile('+repr(mutant)+',m.__file__,"exec"),m.__dict__)\nraise SystemExit(pytest.main(["-q",'+repr('tsfm_lens/tests/test_postrelease_weather_data.py::'+test)+']))'
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        (out/f'{name}.log').write_text(run.stdout+run.stderr, encoding='utf-8')
        caught = run.returncode == 1 and '1 failed' in run.stdout and '1 error' not in run.stdout
        results.append(dict(plant=name, test=test, returncode=run.returncode, caught=caught))
        write(out/'plants.json', results)
        assert caught, (name, run.stdout, run.stderr)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    audit(args.out)
