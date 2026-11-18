"""Independent NumPy replay of published Fourier losses and paired responses.

No producer metric/decomposition helpers, native model imports or new inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(run, report):
    """Recompute all reported model losses/known-response errors from bound arrays."""
    protocol = read(run/'protocol.json')
    rows, plans = read(run/'rows.json'), read(run/'plans.json')
    with np.load(run/'corpus.npz') as f: x, y = f['history'], f['future']
    with np.load(run/'bank.npz') as f: bank = f['inputs'].astype(float)
    assert len({r['source'] for r in rows}) == 40
    for source in {r['source'] for r in rows}:
        assert len({r['split'] for r in rows if r['source'] == source}) == 1
    result = {}
    for model in ['bolt', 'timesfm']:
        with (run/f'{model}_forecasts.npz').open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == read(run/f'{model}_execution.json')['prediction_sha256']
        with np.load(run/f'{model}_forecasts.npz') as f: forecasts = f['predictions'].astype(float)
        published = read(report/f'{model}_measurements.json')
        zeros = forecasts[protocol['zero_id']]
        composed = {}; max_error = 0.; losses_checked = 0
        for index, (row, tree) in enumerate(zip(rows, plans)):
            schemes = dict(tree['schemes'])
            if 'linear_detrend' in tree: schemes['linear_detrend'] = tree['linear_detrend']['schemes']['contiguous']
            for method, levels in schemes.items():
                expected = published['rows'][index]['methods'][method]
                computed_losses = {}
                for k, nodes in levels.items():
                    ids = [node['bank_id'] for node in nodes]
                    assert sorted(leaf for node in nodes for leaf in node['leaves']) == list(range(8))
                    residual = np.max(abs(bank[ids].sum(0)-x[index]))/max(1., np.max(abs(x[index])))
                    assert residual < 2e-6
                    raw = np.sum(forecasts[ids], axis=0)
                    prediction = raw-(int(k)-1)*zeros
                    computed_losses[k] = float(np.dot(prediction-y[index], prediction-y[index])/32/row['scale']**2)
                    raw_loss = float(np.dot(raw-y[index], raw-y[index])/32/row['scale']**2)
                    np.testing.assert_allclose(computed_losses[k], expected['loss'][k], atol=1e-10, rtol=1e-12)
                    np.testing.assert_allclose(raw_loss, expected['raw_loss'][k], atol=1e-10, rtol=1e-12)
                    max_error = max(max_error, abs(computed_losses[k]-expected['loss'][k]))
                    composed[index, method, k] = prediction
                    losses_checked += 2
                crossed = [int(k) for k, loss in computed_losses.items() if loss <= .25]
                assert expected['first_crossing'] == (min(crossed) if crossed else None)
                for node in expected['nodes']:
                    k = str(node['level'])
                    parent = next(n for n in levels[k] if n['leaves'] == node['leaves'])
                    children = [n for n in levels[str(2*int(k))] if set(n['leaves']).issubset(node['leaves'])]
                    delta = sum(forecasts[n['bank_id']] for n in children)-forecasts[parent['bank_id']]-zeros
                    hybrid = composed[index, method, k]+delta
                    gain = computed_losses[k]-np.mean((hybrid-y[index])**2)/row['scale']**2
                    np.testing.assert_allclose(gain, node['split_loss_gain'], atol=1e-10)
        def pairkey(row): return (row['source'], row.get('distractor_bin'), row.get('distractor_amplitude'))
        base = {pairkey(row): i for i, row in enumerate(rows) if row.get('sign') == 0}
        response_count = 0
        for record in published['responses']:
            i = record['index']; row = rows[i]; j = base[pairkey(row)]
            target = row['sign']*.5*np.sin(2*np.pi*5*np.arange(32)/128+.3)
            for level, metrics in record['methods'].items():
                delta = composed[i, 'contiguous', level]-composed[j, 'contiguous', level]
                value = float(np.sum((delta-target)**2)/np.sum(target**2))
                np.testing.assert_allclose(value, metrics['error'], atol=1e-10)
                np.testing.assert_allclose(float(delta@target/(target@target)), metrics['gain'], atol=1e-10)
                response_count += 1
        for key, curve in read(report/f'{model}_summary.json')['curves'].items():
            domain, family, split = key.split('/')
            ids = [i for i, row in enumerate(rows) if (row['domain'], row['family'], row['split']) == (domain, family, split) and row.get('sign', 0) == 0]
            for level in ['1', '2', '4', '8']:
                loss = [np.mean((composed[i, 'contiguous', level]-y[i])**2)/rows[i]['scale']**2 for i in ids]
                np.testing.assert_allclose(np.mean(loss), curve['levels'][level]['loss']['mean'], atol=1e-10)
        result[model] = dict(passed=True, forecast_losses_checked=losses_checked, response_scores_checked=response_count,
                             max_loss_replay_error=max_error, source_units_verified=40)
    return dict(passed=True, models=result,
                scope='Independent all-node raw/corrected losses, threshold crossings, split-loss changes, response NMSE/gain and aggregate curve means; bootstrap interval coverage and pretraining novelty are not certified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.report)
    (args.report/'independent_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
