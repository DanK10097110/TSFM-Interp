"""Post-inference independent replay and exact replication handoff artifacts.

Not imported by the frozen native or selection workers. Reconstructs every
elementary score from raw arrays and checks primary means independently before
distilling results. Does not select a new method or change a hypothesis.
"""
import argparse
import math
from pathlib import Path
import numpy as np

from experiments.feature_search.persistence_protocol import read, sha, write


def replay(out):
    """Replay fixed forecasts, all score/cue/risk rows, and source-level endpoints."""
    data = np.load(out/'inference/inputs.npz')
    plans = read(out/'inference/plans.json'); selection = read(out/'selection.json')
    forecasts = {m: dict(np.load(out/f'{m}_forecasts.npz')) for m in ['bolt', 'timesfm']}
    for model, f in forecasts.items():
        np.testing.assert_array_equal(f['native'], f['bank_predictions'][[p['native'] for p in plans]])
        np.testing.assert_array_equal(f['hybrid'], data['tones']+f['bank_predictions'][[p['residual'] for p in plans]])
        np.testing.assert_array_equal(f['recent_hybrid'], data['recent_tones']+f['bank_predictions'][[p['recent_residual'] for p in plans]])
        np.testing.assert_array_equal(f['tones_zero'], data['tones'])
        np.testing.assert_array_equal(f['recent_zero'], data['recent_tones'])
        np.testing.assert_array_equal(f['seasonal'], data['seasonal'])
        threshold = selection[model]['threshold']
        mask = np.array([bool(p['accepted']) and (threshold is None or p['recent_error'] <= threshold) for p in plans])
        f['veto'] = np.where(mask[:, None], f['hybrid'], f['native'])
    target_lookup, meta, world = {}, {}, {}
    for split in ['fit', 'test']:
        rows = read(out/'evaluation'/split/'rows.json')
        targets = np.load(out/'evaluation'/split/'targets.npz')['future']
        for r, y in zip(rows, targets):
            target_lookup[r['row']] = y; meta[r['row']] = r
            world[r['seed'], r['period'], r['lead'], r['regime']] = r
    errors = []; count = dict(scores=0, cues=0, risks=0, primary_means=0, finite_arrays=0)
    for entry in read(out/'scores.json'):
        r = meta[entry['row']]
        pred = forecasts[entry['model']][entry['method']][r['query']].astype(float)
        expected = float(np.sum((pred-target_lookup[r['row']])**2)/(32*r['scale']))
        np.testing.assert_allclose(expected, entry['loss'], rtol=1e-12, atol=1e-13)
        errors.append(abs(expected-entry['loss'])); count['scores'] += 1
    for entry in read(out/'cue_rows.json'):
        r = meta[entry['row']]; old = world[r['seed'], r['period'], 0, 'persistent']
        f = forecasts[entry['model']][entry['method']].astype(float)
        target = target_lookup[r['row']]
        expected = float(np.mean((f[old['query']]-target)**2-(f[r['query']]-target)**2)/r['scale'])
        np.testing.assert_allclose(expected, entry['gain'], rtol=1e-11, atol=1e-13)
        errors.append(abs(expected-entry['gain'])); count['cues'] += 1
    for entry in read(out/'risk_rows.json'):
        group = [world[entry['seed'], entry['period'], 0, regime] for regime in
                 ['persistent', 'amplitude_decay', 'frequency_drift', 'abrupt_cessation']]
        pred = forecasts[entry['model']][entry['method']][group[0]['query']].astype(float)
        prob = entry['persistence_probability']; weights = [prob]+[(1-prob)/3]*3
        expected = sum(w*float(np.sum((pred-target_lookup[r['row']])**2))/(32*r['scale']) for w, r in zip(weights, group))
        np.testing.assert_allclose(expected, entry['expected'], rtol=1e-12, atol=1e-13)
        errors.append(abs(expected-entry['expected'])); count['risks'] += 1
    scores = read(out/'scores.json'); analysis = read(out/'analysis.json')
    test = [r for r in scores if r['split'] == 'test']; seeds = sorted({r['seed'] for r in test})
    sign_probabilities = []
    for endpoint in analysis['primary']:
        name = endpoint['hypothesis']; values = []
        for seed in seeds:
            if name.startswith('H3'):
                period = float(name.split('_')[1]); model = 'bolt'
                group = [r for r in test if r['seed'] == seed and r['model'] == model and
                         r['period'] == period and r['lead'] > 0 and r['regime'] != 'persistent']
                left, right = 'tones_zero', 'recent_zero'
            else:
                model = name.split('_')[1]
                group = [r for r in test if r['seed'] == seed and r['model'] == model]
                left, right = ('native' if name.startswith('H1') else 'hybrid'), 'veto'
            paired = {}
            for r in group:
                if r['method'] in [left, right]:
                    paired.setdefault((r['period'], r['lead'], r['regime']), {})[r['method']] = r['loss']
            values.append(float(np.mean([v[left]-v[right] for v in paired.values()])))
        np.testing.assert_allclose(values, endpoint['values'], rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(np.mean(values), endpoint['mean'], rtol=1e-11, atol=1e-13)
        declared = np.asarray(endpoint['values'])
        n = int(np.sum(declared != 0)); wins = int(np.sum(declared > 0))
        probability = sum(math.comb(n, k) for k in range(wins, n+1))/2**n if n else 1.
        np.testing.assert_allclose(probability, endpoint['one_sided_sign_p'], rtol=1e-13, atol=1e-16)
        assert endpoint['n_sources'] == len(seeds)
        sign_probabilities.append(probability)
        count['primary_means'] += 1
    order = sorted(range(len(sign_probabilities)), key=sign_probabilities.__getitem__)
    corrected = [0.]*len(order)
    for rank, index in enumerate(order):
        corrected[index] = min(1., max((len(order)-j)*sign_probabilities[order[j]] for j in range(rank+1)))
    for endpoint, adjusted in zip(analysis['primary'], corrected):
        np.testing.assert_allclose(endpoint['holm_p'], adjusted, rtol=1e-13, atol=1e-15)
        assert endpoint['passes'] == (endpoint['mean'] > .01 and adjusted <= .05)
    for model in forecasts:
        for suffix in ['forecasts', 'states']:
            with np.load(out/f'{model}_{suffix}.npz') as arrays:
                for name in arrays.files:
                    if not np.isfinite(arrays[name]).all():
                        raise ValueError('Nonfinite saved array')
                    count['finite_arrays'] += 1
    result = dict(passed=True, counts=count, largest_arithmetic_difference=max(errors), source_sha256=sha(__file__))
    write(out/'independent_replay.json', result)
    return result


def inference_notes(primary):
    """Keep sign-test estimand, empirical mean screen and downside risk distinct."""
    endpoints = {row['hypothesis']: row for row in primary}
    bolt = endpoints['H2_bolt']; timesfm = endpoints['H1_timesfm']
    losses = sorted(value for value in bolt['values'] if value < 0)
    return [
        'Inference scope: the one-sided sign test concerns P(source-level gain > 0 | nonzero gain) > 0.5. It does not test whether the population mean gain is positive.',
        'The preregistered pass flag combines an OBSERVED sample mean gain > 0.01 with a Holm-corrected sign test. The empirical mean threshold is not a significance test of the population mean. A pass must not be described as statistically significant population-mean improvement.',
        'Source-bootstrap intervals describe uncertainty in the mean separately; they are descriptive and uncorrected, and do not replace the preregistered sign endpoint.',
        f"Observed Bolt H2 downside risk: {bolt['positive_sources']}/{bolt['n_sources']} positive sources, but two observed losses {losses!r}; mean bootstrap95% interval {bolt['ci95']!r} crosses zero. Frequent wins coexist with costly losses; the sign-based screen passes while population-mean improvement is not established.",
        f"TimesFM H1: observed mean gain {timesfm['mean']!r}, but only {timesfm['positive_sources']}/{timesfm['n_sources']} positive sources; H1 fails. A positive descriptive mean interval does not rescue its failed sign endpoint.",
        'H3 passes at both periods with32/32 positive TEST sources each. Its sign-test significance concerns source-level win probability, not population-mean improvement; sample means and descriptive intervals remain separate evidence.'
    ]


def report(out):
    """Write full-precision tables, calibration counts and a short persistent synthesis."""
    check = replay(out)
    analysis = read(out/'analysis.json'); score_rows = read(out/'scores.json')
    cues = read(out/'cue_rows.json'); risk_rows = read(out/'risk_rows.json')
    methods = ['native', 'hybrid', 'veto', 'recent_hybrid', 'tones_zero', 'recent_zero', 'seasonal']
    summary = []
    for split in ['fit', 'test']:
        for model in ['bolt', 'timesfm']:
            for method in methods:
                rows = [r for r in score_rows if r['split'] == split and r['model'] == model and r['method'] == method]
                summary.append(dict(split=split, model=model, method=method, mean=float(np.mean([r['loss'] for r in rows]))))
    conditional = []
    for model in ['bolt', 'timesfm']:
        for period in [19.2, 24.]:
            for lead in [0, 8, 16, 32, 64]:
                group = [r for r in score_rows if r['split'] == 'test' and r['model'] == model and
                         r['period'] == period and r['lead'] == lead and r['regime'] != 'persistent']
                conditional.append(dict(model=model, period=period, lead=lead,
                    losses={method: float(np.mean([r['loss'] for r in group if r['method'] == method])) for method in methods},
                    veto_accept_fraction=float(np.mean([r['veto_accept'] for r in group if r['method'] == 'veto']))))
    mixture = []
    for model in ['bolt', 'timesfm']:
        for period in [19.2, 24.]:
            for method in methods:
                group = [r for r in risk_rows if r['split'] == 'test' and r['model'] == model and r['period'] == period and r['method'] == method and r['persistence_probability'] == .25]
                mixture.append(dict(model=model, period=period, method=method, equal_prior_loss=float(np.mean([r['expected'] for r in group]))))
    cue_summary = []
    for model in ['bolt', 'timesfm']:
        for method in ['native', 'hybrid', 'veto', 'recent_hybrid']:
            for regime in ['amplitude_decay', 'frequency_drift', 'abrupt_cessation']:
                group = [r for r in cues if r['split'] == 'test' and r['model'] == model and r['method'] == method and r['regime'] == regime and r['lead'] > 0]
                cue_summary.append(dict(model=model, method=method, regime=regime, mean_gain=float(np.mean([r['gain'] for r in group]))))
    executions = {m: read(out/f'{m}_execution.json') for m in ['bolt', 'timesfm']}
    workers = {m: read(out/f'{m}_worker.json') for m in ['bolt', 'timesfm']}
    result = dict(overall=summary, visible_change_by_lead=conditional, equal_prior_no_cue=mixture,
                  cue_gains=cue_summary, execution=executions, gates={m: w['gates'] for m, w in workers.items()}, independent_replay=check)
    write(out/'handoff.json', result)
    lines = ['Persistence replication:32 FIT/32 TEST source seeds; all values full precision.',
             'Fixed periods19.2/24, cue leads0/8/16/32/64. MSE divided by same-seed persistent-history variance.',
             'Native and native-residual outputs are medians (0.5 quantiles), not means. Hypothetical MSE risks do not identify training priors.',
             'Native calls use singleton geometry; unseen future regimes share exactly the same prediction.',
             '', 'FIT-only frozen selector thresholds:']
    for model in ['bolt', 'timesfm']:
        lines.append(model+': '+repr(analysis['selection'][model]['threshold']))
    lines += ['', 'Six preregistered primary screens (positive source gain means lower loss):']
    lines += inference_notes(analysis['primary'])
    lines.extend(str(row) for row in analysis['primary'])
    lines += ['', 'TEST all-condition mean losses:']
    lines.extend(str(row) for row in summary if row['split'] == 'test')
    lines += ['', 'TEST visible-changing cases by lead and period:']
    lines.extend(str(row) for row in conditional)
    lines += ['', 'TEST equal-prior expected loss, identical histories:']
    lines.extend(str(row) for row in mixture)
    lines += ['', 'Native execution: '+str(executions), 'Exact gates: '+str(result['gates']),
              'Independent arithmetic replay: '+str(check),
              'Limits: synthetic strong tones, fixed stationary recent32 control, six primary sign tests; other slices descriptive. No causal training or natural-data claim.']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    knowledge = read(out/'knowledge.json')
    knowledge['post_analysis'] = result
    knowledge['statistical_interpretation'] = inference_notes(analysis['primary'])
    knowledge['next_decision'] = 'Coordinator review of six primary gates and the recent-refit/residual controls; native slot released; no further expansion authorized'
    write(out/'knowledge.json', knowledge)
    print('\n'.join(lines[:15]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    report(parser.parse_args().out)
