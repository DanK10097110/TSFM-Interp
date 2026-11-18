"""Independent arithmetic replay and compact exact-number pilot handoff.

This post-inference module is not imported by any native worker. Plots describe
hypothetical expected-loss priors, never predictive detection of unseen futures.
"""
import argparse
import csv
import io
from pathlib import Path

import numpy as np

from experiments.feature_search.persistence_protocol import METHODS, REGIMES, read, sha, write


def replay(out):
    """Recompute every reported elementary score directly from saved raw arrays."""
    rows = read(out/'evaluation/rows.json')
    targets = np.load(out/'evaluation/targets.npz')['future']
    forecasts = {m: np.load(out/f'{m}_forecasts.npz') for m in ['bolt', 'timesfm']}
    lookup = {(r['seed'], r['arm'], r['regime']): r for r in rows}
    counts = dict(losses=0, expected_risks=0, cue_gains=0, finite_arrays=0, paired_world_identity=0)
    largest = 0.
    for row in read(out/'scores.json'):
        prediction = forecasts[row['model']][row['method']][row['query']].astype(float)
        expected = float(np.sum((prediction-targets[row['row']])**2)/(32*row['scale']))
        np.testing.assert_allclose(expected, row['loss'], rtol=1e-13, atol=1e-14)
        largest = max(largest, abs(expected-row['loss'])); counts['losses'] += 1
    for row in read(out/'risk_rows.json'):
        group = [lookup[row['seed'], 'identical', r] for r in REGIMES]
        pred = forecasts[row['model']][row['method']][group[0]['query']].astype(float)
        weights = [row['persistence_probability']]+[(1-row['persistence_probability'])/3]*3
        expected = sum(w*float(np.mean((pred-targets[r['row']])**2))/r['scale']
                       for w, r in zip(weights, group))
        np.testing.assert_allclose(expected, row['expected'], rtol=1e-13, atol=1e-14)
        largest = max(largest, abs(expected-row['expected'])); counts['expected_risks'] += 1
    for row in read(out/'cue_rows.json'):
        a = lookup[row['seed'], 'identical', 'persistent']
        b = lookup[row['seed'], 'cued', row['regime']]
        f = forecasts[row['model']][row['method']]
        expected = float(np.mean((f[a['query']].astype(float)-targets[b['row']])**2-
                                 (f[b['query']].astype(float)-targets[b['row']])**2)/b['scale'])
        np.testing.assert_allclose(expected, row['gain'], rtol=1e-12, atol=1e-14)
        largest = max(largest, abs(expected-row['gain'])); counts['cue_gains'] += 1
    for seed in sorted({r['seed'] for r in rows}):
        ids = [lookup[seed, 'identical', r]['query'] for r in REGIMES]
        assert len(set(ids)) == 1
        counts['paired_world_identity'] += 1
    for model in forecasts:
        for path in [out/f'{model}_forecasts.npz', out/f'{model}_states.npz']:
            with np.load(path) as arrays:
                for name in arrays.files:
                    assert np.isfinite(arrays[name]).all()
                    counts['finite_arrays'] += 1
    result = dict(passed=True, counts=counts, largest_abs_arithmetic_difference=largest,
                  source_sha256=sha(__file__), note='Independent formulas; no native inference or outcome-dependent selection')
    write(out/'arithmetic_replay.json', result)
    return result


def report(out):
    """Render exact held-out values plus resource/gate records and an exportable plot."""
    from experiments.feature_search.persistence_analysis import verify
    verify(out)
    check = replay(out)
    analysis = read(out/'analysis.json')
    lines = ['Persistence pilot: exploratory behavioral results',
             '8 independent synthetic seeds: 4 FIT, 4 TEST. No method or hyperparameter selection.',
             'All losses: mean squared error / same-seed persistent-history variance.',
             'Identical histories cannot identify future regimes. Risk curves use hypothetical priors.',
             'Cue gain compares old and updated forecasts on the SAME cued future; positive is beneficial.',
             '', 'Exact TEST mean losses (4 sources each):']
    for arm in ['identical', 'cued']:
        lines.append('\n'+arm)
        for model in ['bolt', 'timesfm']:
            lines.append(model)
            lines.append('regime | '+' | '.join(METHODS))
            for regime in REGIMES:
                values = [next(r['mean'] for r in analysis['summary'] if r['split'] == 'test' and
                               r['arm'] == arm and r['model'] == model and r['regime'] == regime and r['method'] == m)
                          for m in METHODS]
                lines.append(regime+' | '+' | '.join(repr(v) for v in values))
    lines.append('\nExact TEST equal-prior expected loss:')
    for row in analysis['expected_risk']:
        if row['split'] == 'test' and row['persistence_probability'] == .25:
            lines.append(f"{row['model']} {row['method']}: {row['mean']!r}; irreducible {row['irreducible_mean']!r}")
    lines.append('\nExact TEST cue gain (ignore-cue loss minus cued loss):')
    for row in analysis['cue_gain']:
        if row['split'] == 'test' and row['regime'] != 'persistent':
            lines.append(f"{row['model']} {row['method']} {row['regime']}: {row['mean']!r}; source signs {row['positive_sources']}/{row['n_sources']}; descriptive95% {row['ci95']!r}")
    lines.append('\nExact TEST native/method risk crossovers:')
    lines.extend(str(row) for row in analysis['crossings'] if row['split'] == 'test')
    lines.append('\nNative execution and exact gates:')
    lines.append(str(read(out/'execution.json')))
    for model in ['bolt', 'timesfm']:
        worker = read(out/f'{model}_worker.json')
        lines.append(model+' '+str(worker['gates']))
        lines.append(model+' captured block indices '+str(worker['layers']))
    lines.append('\nIndependent raw-array replay: '+str(check))
    lines.append('\nRecovery preserves original failed run and native source. Singleton geometry fixes measured batch-position arithmetic; exact gates were not loosened.')
    lines.append('Raw states: q{query}_full_block{index} and q{query}_residual_block{index}, FP32, all tokens. Use evaluation/rows.json query/source/split mappings. Alignment is deferred; no cross-model geometry claim.')
    lines.append('Limits: narrow strong-tone synthetic generator, one cue lead time/severity, fixed depth1, four TEST sources, no natural-data or training-causality claim. Bootstrap intervals are descriptive and uncorrected.')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    csvtext = io.StringIO()
    writer = csv.DictWriter(csvtext, fieldnames=['split', 'model', 'arm', 'regime', 'method', 'mean', 'n_sources'])
    writer.writeheader()
    for row in analysis['summary']:
        writer.writerow({k: row[k] for k in writer.fieldnames})
    (out/'losses.csv').write_text(csvtext.getvalue(), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, model in zip(axes, ['bolt', 'timesfm']):
        for method in METHODS:
            group = [r for r in analysis['expected_risk'] if r['split'] == 'test' and
                     r['model'] == model and r['method'] == method]
            ax.plot([r['persistence_probability'] for r in group], [r['mean'] for r in group], marker='o', label=method)
        ax.set(title=model, xlabel='Assumed probability of persistent future', xlim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Expected normalized squared loss (4 TEST sources)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Identical histories: hypothetical expected-loss tradeoff')
    fig.text(.5, .01, 'Other three futures equally weighted. No model can detect the unobserved future regime.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(out/'expected_risk.png', dpi=160); plt.close(fig)
    write(out/'report_manifest.json', {name: sha(out/name) for name in ['REPORT.txt', 'losses.csv', 'expected_risk.png', 'analysis.json', 'arithmetic_replay.json']})
    print('\n'.join(lines[:33]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    report(parser.parse_args().out)
