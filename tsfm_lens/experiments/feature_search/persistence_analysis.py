"""Source-paired expected-loss and visible-cue diagnostics, after native inference.

All candidate recipes and priors are frozen. Four TEST sources are an exploratory
screen, not confirmation. Counterfactual worlds and horizon points are never
treated as independent observations. Oracle mixture means are diagnostics only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from experiments.feature_search.persistence_protocol import METHODS, REGIMES, exact, finite, read, sha, write


def loss(prediction, target, scale):
    """Score horizon-mean squared error in fixed source-history variance units."""
    return float(np.mean((finite(prediction).astype(float)-finite(target))**2)/scale)


def risk(prediction, targets, scale, persistence_probability):
    """Expected squared loss and retrospective mixture-mean lower bound."""
    p = persistence_probability
    weights = np.array([p]+[(1-p)/3]*3)
    targets = finite(targets)
    mean = weights@targets
    floor = float(weights@np.mean((targets-mean)**2, axis=1)/scale)
    excess = loss(prediction, mean, scale)
    expected = float(weights@np.mean((targets-prediction)**2, axis=1)/scale)
    if not np.isclose(expected, floor+excess, atol=1e-12, rtol=1e-12):
        raise ValueError('Expected-loss decomposition failed')
    return dict(expected=expected, irreducible=floor, excess=excess)


def cue_gain(uncued_prediction, cued_prediction, cued_target, scale):
    """Use the same cued target on both sides; positive means useful adaptation."""
    return loss(uncued_prediction, cued_target, scale)-loss(cued_prediction, cued_target, scale)


def interval(values):
    """Descriptive source bootstrap; no inferential promotion at n=4 or8."""
    values = finite(values).astype(float)
    if values.ndim != 1 or not len(values):
        raise ValueError('Need one scalar per independent source')
    rng = np.random.default_rng(4951999)
    draw = rng.integers(0, len(values), (4096, len(values)))
    means = values[draw].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                n_sources=len(values), positive_sources=int(np.sum(values > 0)), values=values.tolist())


def verify(out):
    """Reject altered protocols, evaluation artifacts, forecasts, states or inputs."""
    protocol = read(out/'protocol.json')
    if sha(out/'protocol.json') != read(out/'seal.json')['protocol_sha256']:
        raise ValueError('Protocol seal changed')
    for name, digest in protocol['sealed'].items():
        if sha(out/name) != digest:
            raise ValueError('Sealed artifact changed: '+name)
    from experiments.feature_search.persistence_study import bound
    bound(out)
    execution = read(out/'execution.json')
    for model in ['bolt', 'timesfm']:
        if not execution[model]['valid']:
            raise ValueError('Invalid native execution')
        worker = read(out/f'{model}_worker.json')
        if worker['protocol_sha256'] != sha(out/'inference/protocol.json'):
            raise ValueError('Forecast protocol binding changed')
        for name, digest in worker['files'].items():
            if sha(out/name) != digest:
                raise ValueError('Native artifact changed: '+name)
    return protocol


def analyze(out):
    """Persist complete row scores, source contrasts, risk curves and cue effects."""
    protocol = verify(out)
    if (out/'analysis.json').exists():
        raise ValueError('Refuse analysis overwrite')
    rows = read(out/'evaluation/rows.json')
    targets = np.load(out/'evaluation/targets.npz')['future']
    plans = read(out/'inference/plans.json')
    lookup = {(r['seed'], r['arm'], r['regime']): r for r in rows}
    forecasts = {model: np.load(out/f'{model}_forecasts.npz') for model in ['bolt', 'timesfm']}
    scores, risk_rows, cue_rows = [], [], []
    for model, preds in forecasts.items():
        for method in METHODS:
            if preds[method].shape != (protocol['unique_histories'], 32):
                raise ValueError('Forecast shape mismatch')
            finite(preds[method])
            for row in rows:
                pred = preds[method][row['query']]
                scores.append(dict(row, model=model, method=method,
                                   loss=loss(pred, targets[row['row']], row['scale']),
                                   accepted=plans[row['query']]['accepted']))
            for seed in protocol['seeds']:
                same = [lookup[seed, 'identical', r] for r in REGIMES]
                pred = preds[method][same[0]['query']]
                for row in same[1:]:
                    exact(pred, preds[method][row['query']], 'identical-world forecasts')
                future = np.stack([targets[r['row']] for r in same])
                for p in protocol['priors']:
                    risk_rows.append(dict(model=model, method=method, seed=seed,
                                          split=same[0]['split'], persistence_probability=p,
                                          **risk(pred, future, same[0]['scale'], p)))
                for regime in REGIMES:
                    cued = lookup[seed, 'cued', regime]
                    cued_pred = preds[method][cued['query']]
                    target = targets[cued['row']]
                    cue_rows.append(dict(model=model, method=method, seed=seed, split=cued['split'],
                                         regime=regime, gain=cue_gain(pred, cued_pred, target, cued['scale']),
                                         ignore_loss=loss(pred, target, cued['scale']),
                                         cued_loss=loss(cued_pred, target, cued['scale']),
                                         forecast_change=float(np.sqrt(np.mean((pred-cued_pred)**2)/cued['scale']))))
    summaries, risks, cues, contrasts, crossings = [], [], [], [], []
    for split in ['fit', 'test', 'all']:
        def included(row):
            return split == 'all' or row['split'] == split
        for model in forecasts:
            for arm in ['identical', 'cued']:
                for regime in REGIMES:
                    by_method = {}
                    for method in METHODS:
                        selected = [r for r in scores if included(r) and r['model'] == model and
                                    r['method'] == method and r['arm'] == arm and r['regime'] == regime]
                        by_method[method] = np.array([r['loss'] for r in selected])
                        summaries.append(dict(split=split, model=model, arm=arm, regime=regime,
                                              method=method, **interval(by_method[method])))
                    for method in METHODS[1:]:
                        contrasts.append(dict(split=split, model=model, arm=arm, regime=regime,
                                              contrast='native_minus_'+method,
                                              **interval(by_method['native']-by_method[method])))
                    contrasts.append(dict(split=split, model=model, arm=arm, regime=regime,
                                          contrast='zero_minus_native_residual',
                                          **interval(by_method['tones_zero_residual']-by_method['tones_native_residual'])))
            for method in METHODS:
                for p in protocol['priors']:
                    selected = [r for r in risk_rows if included(r) and r['model'] == model and
                                r['method'] == method and r['persistence_probability'] == p]
                    risks.append(dict(split=split, model=model, method=method, persistence_probability=p,
                                      irreducible_mean=float(np.mean([r['irreducible'] for r in selected])),
                                      **interval([r['expected'] for r in selected])))
                for regime in REGIMES:
                    selected = [r for r in cue_rows if included(r) and r['model'] == model and
                                r['method'] == method and r['regime'] == regime]
                    cues.append(dict(split=split, model=model, method=method, regime=regime,
                                     ignore_loss=float(np.mean([r['ignore_loss'] for r in selected])),
                                     cued_loss=float(np.mean([r['cued_loss'] for r in selected])),
                                     **interval([r['gain'] for r in selected])))
            for method in METHODS[1:]:
                endpoint = []
                for p in [0., 1.]:
                    def mean(m):
                        return next(r['mean'] for r in risks if r['split'] == split and
                                    r['model'] == model and r['method'] == m and r['persistence_probability'] == p)
                    endpoint.append(mean('native')-mean(method))
                slope = endpoint[1]-endpoint[0]
                crossing = -endpoint[0]/slope if slope != 0 else None
                crossings.append(dict(split=split, model=model, method=method,
                                      native_minus_method_at_p0=endpoint[0], native_minus_method_at_p1=endpoint[1],
                                      crossing_probability=crossing if crossing is not None and 0 <= crossing <= 1 else None))
    write(out/'scores.json', scores); write(out/'risk_rows.json', risk_rows); write(out/'cue_rows.json', cue_rows)
    write(out/'analysis.json', dict(evidence=protocol['evidence'], summary=summaries,
                                   expected_risk=risks, cue_gain=cues, contrasts=contrasts, crossings=crossings,
                                   extraction=[dict(row=r['row'], seed=r['seed'], split=r['split'],
                                                    arm=r['arm'], regime=r['regime'], **plans[r['query']]) for r in rows]))
    print('Scored', len(scores), 'rows;', len(risk_rows), 'source risks;', len(cue_rows), 'source cue comparisons')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    analyze(parser.parse_args().out)
