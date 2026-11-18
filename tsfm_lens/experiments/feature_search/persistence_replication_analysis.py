"""FIT-only veto freeze, then source-clustered held-out replication scoring."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binom

from experiments.feature_search.persistence_protocol import exact, finite, read, sha, write
from experiments.feature_search.persistence_analysis import risk
from experiments.feature_search.persistence_replication_protocol import METHODS, PERIODS, THRESHOLDS, binding, use_hybrid
from experiments.feature_search.persistence_study import target_guard


def choose(rows, targets, native, hybrid, plans):
    """Ignore every non-FIT row before touching targets; ties prefer earlier thresholds."""
    ids = [i for i, row in enumerate(rows) if row['split'] == 'fit']
    if not ids:
        raise ValueError('No FIT sources')
    targets = finite(np.asarray(targets)[ids])
    fit = [rows[i] for i in ids]
    candidates = []
    for threshold in THRESHOLDS:
        source_losses = {}
        for row, target in zip(fit, targets):
            q = row['query']; plan = plans[q]
            use = use_hybrid(plan['recent_error'], plan['accepted'], threshold)
            pred = hybrid[q] if use else native[q]
            value = float(np.mean((finite(pred)-target)**2)/row['scale'])
            source_losses.setdefault(row['source'], []).append(value)
        candidates.append(dict(threshold=threshold, fit_mean=float(np.mean([np.mean(v) for v in source_losses.values()]))))
    winner = int(np.argmin([r['fit_mean'] for r in candidates]))
    return dict(threshold=candidates[winner]['threshold'], candidates=candidates, n_fit_sources=len(source_losses))


def verify_native(out):
    """Check complete native executions and immutable source/input/artifact bindings."""
    binding(out)
    if sha(out/'protocol.json') != read(out/'seal.json')['sha256']:
        raise ValueError('Protocol seal changed')
    for model in ['bolt', 'timesfm']:
        if not read(out/f'{model}_execution.json')['valid']:
            raise ValueError('Incomplete native worker')
        worker = read(out/f'{model}_worker.json')
        if worker['protocol_sha256'] != sha(out/'inference/protocol.json'):
            raise ValueError('Native protocol mismatch')
        for name, digest in worker['files'].items():
            if sha(out/name) != digest:
                raise ValueError('Native artifact changed')


def select(out):
    """Physically deny TEST-file reads while selecting the global per-model threshold."""
    if (out/'selection.json').exists():
        raise ValueError('Refuse selector overwrite')
    sys.addaudithook(target_guard(out/'evaluation/test'))
    verify_native(out)
    protocol = read(out/'protocol.json')
    for name, digest in protocol['sealed'].items():
        if name.startswith('evaluation/fit/') and sha(out/name) != digest:
            raise ValueError('FIT artifact changed')
    rows = read(out/'evaluation/fit/rows.json')
    if any(r['split'] != 'fit' for r in rows):
        raise ValueError('Non-FIT metadata in FIT files')
    targets = np.load(out/'evaluation/fit/targets.npz')['future']
    plans = read(out/'inference/plans.json')
    result = {}
    for model in ['bolt', 'timesfm']:
        f = np.load(out/f'{model}_forecasts.npz')
        result[model] = choose(rows, targets, f['native'], f['hybrid'], plans)
    result['binding'] = {name: sha(out/name) for name in ['protocol.json', 'evaluation/fit/rows.json',
                         'evaluation/fit/targets.npz', 'bolt_forecasts.npz', 'timesfm_forecasts.npz']}
    write(out/'selection.json', result)
    write(out/'selection_seal.json', dict(sha256=sha(out/'selection.json')))
    knowledge = read(out/'knowledge.json')
    knowledge['fit_selection'] = result
    knowledge['next_decision'] = 'Selector frozen; score TEST once using this binding'
    write(out/'knowledge.json', knowledge)


def summarize(values):
    """One value per independent source; fixed paired bootstrap and exact sign test."""
    values = finite(values).astype(float)
    rng = np.random.default_rng(4953999)
    means = values[rng.integers(0, len(values), (4096, len(values)))].mean(1)
    nonzero = values[values != 0]
    positive = int(np.sum(nonzero > 0))
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                n_sources=len(values), positive_sources=positive, values=values.tolist(),
                one_sided_sign_p=float(binom.sf(positive-1, len(nonzero), .5)) if len(nonzero) else 1.)


def holm(pvalues):
    """Step-down adjusted probabilities over the complete preregistered family."""
    values = np.asarray(pvalues, float); order = np.argsort(values, kind='stable')
    adjusted = np.empty(len(values)); previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., float(values[index]*(len(values)-rank))))
        adjusted[index] = previous
    return adjusted.tolist()


def analyze(out):
    """Score every fixed cell and primary paired contrast after the selector seal."""
    if (out/'analysis.json').exists():
        raise ValueError('Refuse TEST analysis overwrite')
    verify_native(out)
    selected = read(out/'selection.json')
    if sha(out/'selection.json') != read(out/'selection_seal.json')['sha256']:
        raise ValueError('Selector seal changed')
    protocol = read(out/'protocol.json')
    for name, digest in dict(protocol['sealed'], **selected['binding']).items():
        if sha(out/name) != digest:
            raise ValueError('Bound artifact changed: '+name)
    plans = read(out/'inference/plans.json')
    score_rows, cue_rows, risks = [], [], []
    all_methods = METHODS+('veto',)
    for model in ['bolt', 'timesfm']:
        f = dict(np.load(out/f'{model}_forecasts.npz'))
        f['veto'] = np.array([f['hybrid'][q] if use_hybrid(p['recent_error'], p['accepted'], selected[model]['threshold'])
                              else f['native'][q] for q, p in enumerate(plans)])
        for split in ['fit', 'test']:
            rows = read(out/'evaluation'/split/'rows.json')
            y = np.load(out/'evaluation'/split/'targets.npz')['future']
            lookup = {(r['seed'], r['period'], r['lead'], r['regime']): (r, target) for r, target in zip(rows, y)}
            for method in all_methods:
                for row, target in zip(rows, y):
                    q = row['query']; pred = finite(f[method][q])
                    loss = float(np.mean((pred-target)**2)/row['scale'])
                    score_rows.append(dict(row, model=model, method=method, loss=loss,
                                           extraction=plans[q]['accepted'],
                                           veto_accept=use_hybrid(plans[q]['recent_error'], plans[q]['accepted'], selected[model]['threshold'])))
                    old, _ = lookup[row['seed'], row['period'], 0, 'persistent']
                    if row['lead'] == 0:
                        exact(pred, f[method][old['query']], 'identical future-world predictions')
                    gain = float(np.mean((f[method][old['query']]-target)**2)/row['scale'])-loss
                    cue_rows.append(dict(row, model=model, method=method, gain=gain))
                for seed in sorted({r['seed'] for r in rows}):
                    for period in PERIODS:
                        world = [lookup[seed, period, 0, reg] for reg in protocol['regimes']]
                        prediction = f[method][world[0][0]['query']]
                        targets = np.stack([item[1] for item in world])
                        for p in protocol['expected_priors']:
                            risks.append(dict(seed=seed, split=split, model=model, method=method, period=period,
                                              persistence_probability=p, **risk(prediction, targets, world[0][0]['scale'], p)))
    summary = []
    for split in ['fit', 'test']:
        for model in ['bolt', 'timesfm']:
            for method in all_methods:
                for period in PERIODS:
                    for lead in protocol['leads']:
                        for regime in protocol['regimes']:
                            group = [r for r in score_rows if r['split'] == split and r['model'] == model and
                                     r['method'] == method and r['period'] == period and r['lead'] == lead and r['regime'] == regime]
                            summary.append(dict(split=split, model=model, method=method, period=period,
                                                lead=lead, regime=regime, **summarize([r['loss'] for r in group])))
    test = [r for r in score_rows if r['split'] == 'test']
    seeds = sorted({r['seed'] for r in test})
    primary = []
    for model in ['bolt', 'timesfm']:
        for label, comparator in [('H1', 'native'), ('H2', 'hybrid')]:
            gains = [float(np.mean([r['loss'] for r in test if r['model'] == model and r['method'] == comparator and r['seed'] == s]))-
                     float(np.mean([r['loss'] for r in test if r['model'] == model and r['method'] == 'veto' and r['seed'] == s])) for s in seeds]
            primary.append(dict(hypothesis=label+'_'+model, **summarize(gains)))
    for period in PERIODS:
        gains = []
        for seed in seeds:
            group = [r for r in test if r['model'] == 'bolt' and r['seed'] == seed and r['period'] == period and r['lead'] > 0 and r['regime'] != 'persistent']
            gains.append(float(np.mean([r['loss'] for r in group if r['method'] == 'tones_zero']))-
                         float(np.mean([r['loss'] for r in group if r['method'] == 'recent_zero'])))
        primary.append(dict(hypothesis='H3_'+str(period), **summarize(gains)))
    for row, p in zip(primary, holm([r['one_sided_sign_p'] for r in primary])):
        row.update(holm_p=p, passes=bool(row['mean'] > .01 and p <= .05))
    write(out/'scores.json', score_rows); write(out/'cue_rows.json', cue_rows); write(out/'risk_rows.json', risks)
    result = dict(primary=primary, summary=summary, selection=selected,
                  evidence='prospective source-held-out synthetic replication; six corrected primary screens; other slices descriptive')
    write(out/'analysis.json', result)
    knowledge = read(out/'knowledge.json')
    knowledge['exact_findings'] = dict(primary=primary, selection=selected, all_cells='analysis.json', source_scores='scores.json', cue_gains='cue_rows.json', expected_risk='risk_rows.json')
    knowledge['evidence_class'] = result['evidence']
    knowledge['next_decision'] = 'Review primary gates and cellwise tradeoffs with coordinator; do not expand autonomously'
    write(out/'knowledge.json', knowledge)
    print(primary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        select(args.out)
    else:
        analyze(args.out)
