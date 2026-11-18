"""Version2 prospective evaluation with source-limited domains explicitly descriptive.

All intervals resample sources within domain, never the overlapping origins.
Stability and hindcast success are descriptive, not independent confirmation.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.feature_search.coherence_study import read, sha, write
from experiments.feature_search.natural_validation_study import DEPTHS, DOMAINS, METHODS, MODELS, ORIGINS
from experiments.feature_search.natural_validation_v2 import record_knowledge


def past_choices(forecasts, targets, usable):
    """Choose hybrid at each fixed depth only for strict two-hindcast mean gain."""
    forecasts, targets = np.asarray(forecasts), np.asarray(targets)
    if forecasts.shape[1:] != (2, len(METHODS), 32) or targets.shape != (len(forecasts), 2, 32):
        raise ValueError('Selector accepts exactly two past targets, never main future')
    loss = np.mean((forecasts-targets[:, :, None])**2, axis=-1)
    choices = np.zeros((len(forecasts), 3), int)
    eligible = np.asarray(usable, bool).all(axis=1)
    for d in DEPTHS:
        idx = METHODS.index(f'hybrid{d}')
        choose = eligible & np.isfinite(loss[:, :, [0, idx]]).all(axis=(1, 2))
        choose &= loss[:, :, idx].mean(axis=1) < loss[:, :, 0].mean(axis=1)
        choices[choose, d-1] = idx
    return choices, loss


def fit_rule(fit_losses):
    """Select from native-first fixed candidates using an explicit FIT-only matrix."""
    fit_losses = np.asarray(fit_losses)
    if fit_losses.ndim != 2 or not len(fit_losses) or not np.isfinite(fit_losses).all():
        raise ValueError('Finite nonempty FIT matrix required')
    return int(np.argmin(fit_losses.mean(axis=0)))


def stability(plans):
    """Match extracted frequencies across origins; amplitudes are own-history SDs."""
    pairs = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a, b = plans[left]['details'], plans[right]['details']
        matches = []
        if a and b:
            cost = abs(np.asarray([v['frequency'] for v in a])[:, None]-np.asarray([v['frequency'] for v in b])[None])
            ia, ib = linear_sum_assignment(cost)
            for u, v in zip(ia, ib):
                amp_a, amp_b = np.linalg.norm(a[u]['coefficients']), np.linalg.norm(b[v]['coefficients'])
                physical_a, physical_b = amp_a*plans[left]['scale'], amp_b*plans[right]['scale']
                matches.append(dict(left_tone=int(u), right_tone=int(v),
                    frequency_difference=float(cost[u, v]),
                    standardized_amplitude_difference=float(amp_b-amp_a),
                    common_scale_amplitude_relative_change=float(abs(physical_b-physical_a)/max(physical_a, physical_b, 1e-12))))
        pairs.append(dict(origins=[ORIGINS[left], ORIGINS[right]], matched=matches,
            unmatched_left=len(a)-len(matches), unmatched_right=len(b)-len(matches)))
    return dict(abstained=[p['abstained'] for p in plans],
                depths=[p['extracted_depth'] for p in plans], pairs=pairs,
                scope='Descriptive potential cue; no learned threshold or TEST-selected rule')


def summary(native, other):
    """Source-paired mean MSE gain and conditional descriptive bootstrap interval."""
    native, other = np.asarray(native), np.asarray(other)
    gain = native-other
    if not len(gain):
        raise ValueError('Need at least one source')
    interval = None
    if len(gain) >= 2:
        rng = np.random.default_rng(4951000)
        draws = gain[rng.integers(len(gain), size=(1999, len(gain)))].mean(axis=1)
        interval = np.quantile(draws, [.025, .975]).tolist()
    return dict(n=len(gain), native_mse=float(native.mean()), mse=float(other.mean()),
        paired_mean_gain=float(gain.mean()),
        gain_ci95=interval,
        interval_status='not estimable: n=1; no CI or significance claim' if len(gain) == 1 else 'conditional descriptive bootstrap',
        wins=int((gain > 0).sum()), ties=int((gain == 0).sum()), losses=int((gain < 0).sum()))


def bound_load(out, protocol, name):
    """Verify a single artifact only at the phase authorized to open it."""
    if sha(out/name) != protocol['hashes'][name]:
        raise ValueError(f'Artifact changed: {name}')
    with np.load(out/name, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def analyze(out):
    """Persist exact findings or a failed analysis attempt alongside its artifacts."""
    try:
        result = _analyze(out)
    except Exception as exc:
        record_knowledge(out, 'analysis', 'failed', dict(error_type=type(exc).__name__, error=str(exc)),
                         'Resolve failed analysis gates; do not claim scientific results.')
        raise
    record_knowledge(out, 'analysis', 'complete', dict(rules=result['rules'],
        aggregates=result['aggregates'], continuation_failure=result['continuation_failure'],
        gates=result['gates'], analysis_sha256=sha(out/'analysis.json')),
        'Coordinator review of per-domain TEST gains and hindcast failure detection; no automatic expansion or TEST depth deployment.')
    return result


def _analyze(out):
    """Seal choices, open FIT, seal domain rules, then open TEST for evaluation."""
    if any((out/name).exists() for name in ('past_choices.json', 'fit_rules.json', 'analysis.json')):
        raise FileExistsError('Refuse repeated analysis in frozen run')
    p = read(out/'protocol.json')
    for path, digest in p['sources'].items():
        if sha(path) != digest:
            raise ValueError('Frozen analysis/source changed')
    for name in ('rows.json', 'plans.json'):
        if sha(out/name) != p['hashes'][name]:
            raise ValueError('Rows/plans changed')
    if p.get('version') == 2 and sha(out/'amendment.json') != p['hashes']['amendment.json']:
        raise ValueError('Frozen amendment changed')
    rows, plans = read(out/'rows.json'), read(out/'plans.json')
    limited = {d: sum(r['domain'] == d for r in rows) < 8 for d in DOMAINS}
    if [r['source'] for r in rows] != p['source_ids'] or len(set(p['source_ids'])) != len(rows):
        raise ValueError('Source ID binding/disjointness failed')
    if {r['split'] for r in rows} != {'FIT', 'TEST'}:
        raise ValueError('Source FIT/TEST split required')
    execution = read(out/'execution.json')
    forecasts, choices, hindloss = {}, {}, {}
    past = bound_load(out, p, 'hindcast_targets.npz')['targets']
    usable = [[v['selector_usable'] for v in plan[:2]] for plan in plans]
    worker_hashes = {}
    for model in MODELS:
        record = read(out/f'{model}_worker.json')
        if (not execution[model]['valid'] or not record['repeat_exact'] or not record['finite']
                or record['closure_fp32'] > p['gates']['closure_fp32']
                or record['seconds'] > p['seconds_per_model'] or record['peak_rss'] > p['worker_cap']
                or record['protocol_sha256'] != sha(out/'protocol.json')):
            raise ValueError('Native validity/resource gate failed')
        artifact = out/f'{model}_forecasts.npz'
        worker_hashes[model] = sha(artifact)
        if worker_hashes[model] != record['sha256']:
            raise ValueError('Forecast hash changed')
        with np.load(artifact, allow_pickle=False) as f:
            if f['source_ids'].tolist() != p['source_ids'] or f['methods'].tolist() != list(METHODS) or f['origins'].tolist() != list(ORIGINS):
                raise ValueError('Forecast identity/order binding failed')
            forecasts[model] = f['forecasts']
        if forecasts[model].shape != (len(rows), 3, len(METHODS), 32) or not np.isfinite(forecasts[model]).all():
            raise ValueError('Forecast shape/finite gate failed')
        choices[model], hindloss[model] = past_choices(forecasts[model][:, :2], past, usable)
    write(out/'past_choices.json', dict(rule=p['selector'], source_ids=p['source_ids'],
        forecast_hashes=worker_hashes, choices={k: v.tolist() for k, v in choices.items()},
        hindcast_losses={k: v.tolist() for k, v in hindloss.items()}))

    fit = bound_load(out, p, 'FIT_targets.npz')
    fit_ids = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'FIT'])
    test_ids = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'TEST'])
    if not np.array_equal(fit['indices'], fit_ids):
        raise ValueError('FIT target identity mismatch')
    names = list(METHODS)+[f'past_selector{d}' for d in DEPTHS]+['FIT_rule']
    candidates = ['native']+[f'hybrid{d}' for d in DEPTHS]+[f'past_selector{d}' for d in DEPTHS]
    matrix, rules = {}, {}
    for model in MODELS:
        f = forecasts[model][:, 2]
        selected = f[np.arange(len(rows))[:, None], choices[model]]
        matrix[model] = np.concatenate([f, selected, np.zeros((len(rows), 1, 32))], axis=1)
        fit_loss = np.mean((matrix[model][fit_ids]-fit['targets'][:, None])**2, axis=-1)
        rules[model] = {}
        for domain in DOMAINS:
            mask = [rows[i]['domain'] == domain for i in fit_ids]
            candidate_ids = [names.index(n) for n in candidates]
            choice = candidates[fit_rule(fit_loss[mask][:, candidate_ids])]
            rules[model][domain] = choice
            for i, row in enumerate(rows):
                if row['domain'] == domain:
                    matrix[model][i, -1] = matrix[model][i, names.index(choice)]
    write(out/'fit_rules.json', dict(rule=p['fit_rule'], candidates=candidates, rules=rules,
        fit_source_ids=[rows[i]['source'] for i in fit_ids],
        past_choices_sha256=sha(out/'past_choices.json')))

    test = bound_load(out, p, 'TEST_targets.npz')
    if not np.array_equal(test['indices'], test_ids):
        raise ValueError('TEST target identity mismatch')
    targets = np.zeros((len(rows), 32))
    targets[fit_ids], targets[test_ids] = fit['targets'], test['targets']
    records, aggregates, continuation_failure = [], [], []
    for model in MODELS:
        loss = np.mean((matrix[model]-targets[:, None])**2, axis=-1)
        for i, row in enumerate(rows):
            records.append(dict(row, model=model, source_limited=limited[row['domain']], losses=dict(zip(names, loss[i].tolist())),
                selected_methods=[METHODS[v] for v in choices[model][i]],
                fit_rule=rules[model][row['domain']], stability=stability(plans[i])))
        for domain in DOMAINS:
            split_scores = {}
            for split in ('FIT', 'TEST'):
                ix = [i for i, row in enumerate(rows) if row['domain'] == domain and row['split'] == split]
                split_scores[split] = {}
                for k, name in enumerate(names):
                    score = summary(loss[ix, 0], loss[ix, k])
                    split_scores[split][name] = score
                    aggregates.append(dict(model=model, domain=domain, split=split, method=name,
                        source_limited=limited[domain], evidence='source-limited descriptive' if limited[domain] else 'exploratory behavioral', **score))
                for depth in DEPTHS:
                    k = METHODS.index(f'hybrid{depth}')
                    past_harm = (hindloss[model][ix, :, k]-hindloss[model][ix, :, 0]).mean(axis=1)
                    actual_harm = loss[ix, k]-loss[ix, 0]
                    predicted = choices[model][ix, depth-1] == 0
                    failed = actual_harm > 0
                    continuation_failure.append(dict(model=model, domain=domain, split=split, depth=depth,
                        n=len(ix), source_limited=limited[domain], past_harm=past_harm.tolist(), future_harm=actual_harm.tolist(),
                        predicted_failure_actual_failure=int((predicted & failed).sum()),
                        predicted_failure_actual_nonfailure=int((predicted & ~failed).sum()),
                        predicted_success_actual_failure=int((~predicted & failed).sum()),
                        predicted_success_actual_nonfailure=int((~predicted & ~failed).sum()),
                        main_extraction_abstentions=sum(plans[i][2]['abstained'] for i in ix)))
            for item in aggregates:
                if item['model'] == model and item['domain'] == domain:
                    name = item['method']
                    item['generalization_eligible'] = False
                    item['positive_in_FIT_and_TEST'] = all(split_scores[s][name]['paired_mean_gain'] > 0 for s in ('FIT', 'TEST'))
    np.savez_compressed(out/'evaluated_forecasts.npz', **matrix, methods=np.asarray(names), source_ids=np.asarray(p['source_ids']))
    result = dict(scope=p['interpretation'], metric='MSE in each origin own-history SD units; main future in main-history units',
        uncertainty='1999 within-domain source bootstrap only for n>=2; n=1 has no CI or significance claim. Source-limited domains descriptive only. No multiple-testing confirmation.',
        availability=p.get('availability'),
        gates=dict(native_workers=True, source_disjoint=True, identities=True, choices_frozen_before_main_targets=True,
                   rules_frozen_before_TEST=True, no_TEST_depth_deployment=True),
        rules=rules, aggregates=aggregates, source_records=records, continuation_failure=continuation_failure,
        hashes={name: sha(out/name) for name in ('protocol.json', 'past_choices.json', 'fit_rules.json', 'evaluated_forecasts.npz')})
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/natural_validation_pilot4951001'))
    analyze(parser.parse_args().out)
