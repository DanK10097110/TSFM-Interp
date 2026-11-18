"""CPU-only calibration of actual production seed inference, with fixed null laws.

Symmetric correlated nulls check gross family error behavior. A skewed mean-zero
law deliberately violates sign-flip assumptions and is labeled accordingly.
Independent simulation replicates, not hypotheses inside them, supply binomial
uncertainty. This is not precise certification of a nominal five-percent level.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import time

import numpy as np
from scipy.stats import binomtest

from tsfm_lens.analysis.feature_interference import paired_seed_effect, holm

PROTOCOL_PATH = Path(__file__).with_name('statistics_calibration_protocol.json')


def load_protocol():
    """Read the frozen calibration design before controls or simulation are run."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def run_family(differences, *, seed, n_boot=199, n_permutations=999, alpha=.05):
    """Pass only genuine finite 1-D seed vectors to the actual production helper."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or differences.shape[1] != 4 or len(differences) < 3 or not np.isfinite(differences).all():
        raise ValueError('Need independent seed rows and exactly four finite hypothesis columns')
    effects = {f'h{j}': paired_seed_effect(differences[:, j], n_boot=n_boot,
        n_permutations=n_permutations, seed=seed) for j in range(4)}
    adjusted = holm({name: effect['p'] for name, effect in effects.items()})
    raw_rejections = np.array([effects[f'h{j}']['p'] < alpha for j in range(4)])
    adjusted_rejections = np.array([adjusted[f'h{j}'] < alpha for j in range(4)])
    ci_exclusions = np.array([effects[f'h{j}']['lo'] > 0 or effects[f'h{j}']['hi'] < 0 for j in range(4)])
    for name, effect in effects.items():
        effect['p_holm'] = adjusted[name]
    return dict(effects=effects, raw_rejections=raw_rejections,
                adjusted_rejections=adjusted_rejections, ci_exclusions=ci_exclusions)


def correlated_gaussian(rng, n=64, m=4):
    """Equicorrelation .5 with independent generating-seed rows."""
    common = rng.normal(size=(n, 1))
    independent = rng.normal(size=(n, m))
    return (common + independent) / np.sqrt(2.)


def controls(protocol=None):
    """Known nonzero answers, zero decoy and the omitted-Holm threshold fixture."""
    protocol = load_protocol() if protocol is None else protocol
    settings = protocol['resampling']
    kwargs = dict(n_boot=settings['n_boot'], n_permutations=settings['n_permutations'], alpha=settings['alpha'])
    n = protocol['n_seeds_per_family']
    zero = run_family(np.zeros((n, 4)), seed=113, **kwargs)
    clear_data = correlated_gaussian(np.random.default_rng(protocol['master_seed'] + 99), n=n) + 1.5
    clear = run_family(clear_data, seed=113, **kwargs)
    moderate_data = np.zeros((n, 4))
    moderate_data[:, 0] = np.r_[np.full(n // 2, -1.), np.full(n - n // 2, 1.)] + .28
    moderate = run_family(moderate_data, seed=113, **kwargs)
    if zero['adjusted_rejections'].any() or any(effect['p'] != 1 or effect['value'] != 0 or effect['lo'] != 0 or effect['hi'] != 0 for effect in zero['effects'].values()):
        raise AssertionError('All-zero decoy failed its exact known answer')
    if not clear['adjusted_rejections'].all() or any(effect['lo'] <= 0 for effect in clear['effects'].values()):
        raise AssertionError('Clear nonzero known answer was not detected')
    if not moderate['raw_rejections'][0] or moderate['adjusted_rejections'].any():
        raise AssertionError('Omitted-Holm guard needs a raw-only rejection and no adjusted rejection')
    return dict(all_zero=zero, clear_nonzero=clear, omitted_Holm_guard=moderate)


def replicate_data(protocol, scenario, replicate, phase):
    """Keep data and Monte Carlo randomization independent across replicate families."""
    code = {'symmetric_null': 0, 'skew_mean_zero_stress': 1}[scenario]
    sequence = np.random.SeedSequence([protocol['master_seed'], code, replicate, phase])
    data_sequence, permutation_sequence = sequence.spawn(2)
    rng = np.random.default_rng(data_sequence)
    gaussian = correlated_gaussian(rng, protocol['n_seeds_per_family'])
    data = gaussian if scenario == 'symmetric_null' else np.exp(1.5 * gaussian - 1.5**2 / 2) - 1
    permutation_seed = int(np.random.default_rng(permutation_sequence).integers(0, 2**31 - 1))
    return data, permutation_seed


def binomial_summary(count, total):
    """Use independent replicate families as the binomial uncertainty unit."""
    if total < 1:
        return dict(count=count, total=total, untestable='no completed simulation families')
    interval = binomtest(count, total).proportion_ci(confidence_level=.95, method='exact')
    return dict(count=int(count), total=int(total), rate=float(count / total),
                lo=float(interval.low), hi=float(interval.high), interval='95% exact Clopper-Pearson')


def summarize(records):
    """Retain complete family and per-hypothesis rejection counts without pooling dependence."""
    if not records:
        return dict(completed_replicates=0)
    adjusted = np.array([row['adjusted_rejections'] for row in records], dtype=bool)
    raw = np.array([row['raw_rejections'] for row in records], dtype=bool)
    exclusions = np.array([row['ci_exclusions'] for row in records], dtype=bool)
    n = len(records)
    return dict(completed_replicates=n,
        Holm_family_rejection=binomial_summary(int(adjusted.any(axis=1).sum()), n),
        raw_unadjusted_family_rejection=binomial_summary(int(raw.any(axis=1).sum()), n),
        per_hypothesis={f'h{j}': dict(Holm_rejections=binomial_summary(int(adjusted[:, j].sum()), n),
            raw_rejections=binomial_summary(int(raw[:, j].sum()), n),
            bootstrap_CI_excludes_population_mean_zero=binomial_summary(int(exclusions[:, j].sum()), n)) for j in range(4)})


def _serializable(value):
    """Serialize arrays while retaining all replicate outcomes and Monte Carlo seeds."""
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(out):
    """Pilot runtime first, freeze replicate count, then run at most five CPU minutes."""
    protocol = load_protocol()
    out.mkdir(parents=True, exist_ok=False)
    raw_protocol = PROTOCOL_PATH.read_bytes()
    (out / 'protocol.json').write_bytes(raw_protocol)
    snapshots, function_records = [], {}
    for function in (paired_seed_effect, holm):
        source = inspect.getsource(function)
        snapshots.append(f'# {function.__module__}.{function.__name__}\n{source}\n')
        function_records[function.__name__] = dict(module=function.__module__, source_file=inspect.getsourcefile(function),
            source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
            compiled_code_sha256=hashlib.sha256(marshal.dumps(function.__code__)).hexdigest())
    (out / 'production_function_snapshot.py').write_text('\n'.join(snapshots), encoding='utf-8')
    provenance = dict(protocol_sha256=hashlib.sha256(raw_protocol).hexdigest(), functions=function_records,
        worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), execution='actual imported production functions, not source-snapshot execution')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    started = time.perf_counter()
    known = controls(protocol)
    (out / 'known_controls.json').write_text(json.dumps(_serializable(known), indent=2), encoding='utf-8')
    settings = protocol['resampling']
    kwargs = dict(n_boot=settings['n_boot'], n_permutations=settings['n_permutations'], alpha=settings['alpha'])
    scenarios = ('symmetric_null', 'skew_mean_zero_stress')
    pilot = []
    for scenario in scenarios:
        for replicate in range(protocol['pilot_replicates_per_scenario']):
            data, permutation_seed = replicate_data(protocol, scenario, replicate, 0)
            block = time.perf_counter()
            result = run_family(data, seed=permutation_seed, **kwargs)
            pilot.append(dict(scenario=scenario, replicate=replicate, seconds=time.perf_counter() - block,
                permutation_seed=permutation_seed, **result))
    maximum = max(row['seconds'] for row in pilot)
    available = max(0., protocol['runtime']['main_budget_seconds'] - (time.perf_counter() - started))
    count = min(protocol['planned_main_replicates_per_scenario'], int(available // (2 * maximum)))
    count = count // 16 * 16
    decision = dict(pilot_max_family_seconds=maximum, elapsed_seconds=time.perf_counter() - started,
        selected_main_replicates_per_scenario=count, projected_main_seconds=2 * count * maximum,
        rule='runtime only;pilot outcomes excluded from main calibration counts')
    (out / 'pilot.json').write_text(json.dumps(_serializable(dict(decision=decision, outcomes=pilot)), indent=2), encoding='utf-8')
    print(json.dumps(decision), flush=True)
    if count < 32:
        (out / 'paused.json').write_text(json.dumps(dict(reason='runtime pilot cannot fit at least32replicates per null', decision=decision)), encoding='utf-8')
        return
    records = {scenario: [] for scenario in scenarios}
    for scenario in scenarios:
        for replicate in range(count):
            if time.perf_counter() - started >= protocol['runtime']['cap_seconds']:
                break
            data, permutation_seed = replicate_data(protocol, scenario, replicate, 1)
            result = run_family(data, seed=permutation_seed, **kwargs)
            records[scenario].append(dict(replicate=replicate, permutation_seed=permutation_seed,
                data_seed_sequence=[protocol['master_seed'], 0 if scenario == 'symmetric_null' else 1, replicate, 1], **result))
        print(json.dumps(dict(scenario=scenario, **summarize(records[scenario]))), flush=True)
    summaries = {scenario: summarize(value) for scenario, value in records.items()}
    report = dict(protocol=protocol, runtime_decision=decision, seconds=time.perf_counter() - started,
        summaries=summaries, calibration_scope='coarse empirical check,not precise5%certification',
        stress_caution='Skew mean-zero law violates sign-flip symmetry;its rejection frequency is not a required nominal pass',
        binomial_unit='independent simulation families,not four correlated hypotheses pooled',
        sign_sensitivity_caution='Exact sign sensitivity concerns P(positive|nonzero),not the mean;not substituted for the primary mean inference')
    (out / 'results.json').write_text(json.dumps(_serializable(report), indent=2), encoding='utf-8')
    (out / 'replicate_results.json').write_text(json.dumps(_serializable(records), indent=2), encoding='utf-8')
    print(json.dumps(_serializable(report), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
