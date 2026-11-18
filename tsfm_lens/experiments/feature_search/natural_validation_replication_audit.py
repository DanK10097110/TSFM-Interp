"""Post-analysis numerical audit and explicitly exploratory paired contrasts.

Adds neither native calls nor fitting. Between-model effects subtract hybrid
gains within the same source, preserving covariance. Original primary artifacts
are immutable; this file writes a separate audit with its own source hash.
"""
from pathlib import Path
import argparse

import numpy as np

from experiments.feature_search.natural_validation_replication import DOMAINS, MODELS, METHODS, ORIGINS, read, write, sha, knowledge


def paired_difference(left_ids, left, right_ids, right):
    """Align unique source identities before differencing model-specific gains."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.shape != (len(left_ids),) or right.shape != (len(right_ids),)
            or len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids)
            or set(left_ids) != set(right_ids)):
        raise ValueError('One paired gain per identical unique source ID required')
    lookup = dict(zip(right_ids, right))
    return left-np.asarray([lookup[s] for s in left_ids])


def paired_interval(delta, family, n_boot=23999):
    """Resample already paired source differences; report exploratory intervals."""
    delta = np.asarray(delta, float)
    if delta.ndim != 1 or not np.isfinite(delta).all() or len(delta) < 2 or family not in (3, 6):
        raise ValueError('Finite paired source differences, n>=2 and declared family3/6 required')
    rng = np.random.default_rng(4955002)
    means = []
    for start in range(0, n_boot, 1024):
        ix = rng.integers(len(delta), size=(min(1024, n_boot-start), len(delta)))
        means.extend(delta[ix].mean(axis=1))
    tail = .05/(2*family)
    return dict(n=len(delta), paired_mean_difference=float(delta.mean()),
        paired_ci95=np.quantile(means, [.025, .975]).tolist(),
        exploratory_bonferroni_family=family,
        exploratory_adjusted_interval=np.quantile(means, [tail, 1-tail]).tolist(),
        positive_sources=int((delta > 0).sum()), ties=int((delta == 0).sum()), negative_sources=int((delta < 0).sum()),
        source_differences=delta.tolist(),
        scope='Post-primary exploratory source-paired bootstrap; approximate intervals, not an additional frozen primary claim')


def hybrid_vs_seasonal(hybrid_loss, seasonal_loss, n_boot=23999):
    """Positive difference means hybrid beats seasonal, using the same targets."""
    hybrid, seasonal = np.asarray(hybrid_loss, float), np.asarray(seasonal_loss, float)
    if hybrid.shape != seasonal.shape:
        raise ValueError('Matched source losses required')
    return paired_interval(seasonal-hybrid, family=6, n_boot=n_boot)


def audit(out):
    """Independently re-evaluate saved point forecasts and source-paired contrasts."""
    if (out/'audit.json').exists():
        raise FileExistsError('Refuse audit overwrite')
    p, a, rows, plans = [read(out/name) for name in ('protocol.json', 'analysis.json', 'rows.json', 'plans.json')]
    original_analysis_sha = sha(out/'analysis.json')
    original_protocol_sha = sha(out/'protocol.json')
    for path, digest in {**p['sources'], **p['pilot_bindings']}.items():
        if sha(path) != digest:
            raise ValueError('Frozen source/pilot changed')
    for name, digest in p['hashes'].items():
        if sha(out/name) != digest:
            raise ValueError('Frozen artifact changed')
    for name, digest in a['hashes'].items():
        if sha(out/name) != digest:
            raise ValueError('Analysis artifact changed')
    if len(rows) != 192 or len(set(p['source_ids'])) != 192:
        raise ValueError('Source count/uniqueness failed')
    with np.load(out/'main_targets.npz', allow_pickle=False) as f:
        if f['source_ids'].tolist() != p['source_ids']:
            raise ValueError('Target identities changed')
        target = f['targets']
    with np.load(out/'hindcast_targets.npz', allow_pickle=False) as f:
        hind = f['targets']
    choices = read(out/'past_choices.json')
    execution = read(out/'execution.json')
    loss, gains, seasonal_reference, workers = {}, {}, None, {}
    numeric_error = 0.
    for model in MODELS:
        record = read(out/f'{model}_worker.json')
        workers[model] = record
        if (not execution[model]['valid'] or execution[model]['returncode'] != 0
                or not record['finite'] or not record['repeat_exact'] or record['closure_fp32'] > 2e-6
                or record['seconds'] > 900 or record['peak_rss'] > 3*2**30
                or record['protocol_sha256'] != original_protocol_sha
                or record['sha256'] != sha(out/f'{model}_forecasts.npz')):
            raise ValueError('Native worker gates failed')
        with np.load(out/f'{model}_forecasts.npz', allow_pickle=False) as f:
            if f['source_ids'].tolist() != p['source_ids'] or f['origins'].tolist() != list(ORIGINS) or f['methods'].tolist() != list(METHODS):
                raise ValueError('Forecast source/method/origin alignment failed')
            pred = f['forecasts']
        loss[model] = np.square(pred[:, 2]-target[:, None]).mean(axis=2)
        gains[model] = loss[model][:, 0]-loss[model][:, 1]
        hl = np.square(pred[:, :2]-hind[:, :, None]).mean(axis=3)
        usable = np.asarray([[q['selector_usable'] for q in source[:2]] for source in plans]).all(axis=1)
        depth1 = usable & ((hl[:, 0, 0]+hl[:, 1, 0]) > (hl[:, 0, 1]+hl[:, 1, 1]))
        secondary = usable & ((hl[:, 0, 0]+hl[:, 1, 0]) > (hl[:, 0, 4]+hl[:, 1, 4]))
        np.testing.assert_array_equal(depth1, choices['choices'][model]['depth1'])
        np.testing.assert_array_equal(secondary, choices['choices'][model]['secondary'])
        with np.load(out/'evaluated_forecasts.npz', allow_pickle=False) as f:
            evaluated = f[model]
            np.testing.assert_array_equal(evaluated[:, :5], pred[:, 2])
            np.testing.assert_array_equal(evaluated[:, 5], np.where(depth1[:, None], pred[:, 2, 1], pred[:, 2, 0]))
            expected = pred[:, 2, 4].copy()
            for i, row in enumerate(rows):
                if p['frozen_pilot_rules'][model][row['domain']] == 'past_selector3' and not secondary[i]:
                    expected[i] = pred[i, 2, 0]
            np.testing.assert_array_equal(evaluated[:, 6], expected)
        if seasonal_reference is None:
            seasonal_reference = pred[:, 2, 2]
        else:
            np.testing.assert_array_equal(seasonal_reference, pred[:, 2, 2])
        for item in a['primary']:
            if item['model'] != model:
                continue
            ix = np.asarray([i for i, row in enumerate(rows) if row['domain'] == item['domain']])
            assert len(ix) == 64
            observed = (loss[model][ix, 0].mean(), loss[model][ix, 1].mean(), gains[model][ix].mean())
            saved = (item['native_mse'], item['mse'], item['paired_mean_gain'])
            numeric_error = max(numeric_error, float(abs(np.asarray(observed)-saved).max()))
            np.testing.assert_allclose(observed, saved, rtol=0, atol=1e-12)
            assert item['wins'] == int((gains[model][ix] > 0).sum())
            assert item['ties'] == int((gains[model][ix] == 0).sum())
            assert item['losses'] == int((gains[model][ix] < 0).sum())
    between, vs_seasonal = [], []
    for domain in DOMAINS:
        ix = np.asarray([i for i, row in enumerate(rows) if row['domain'] == domain])
        ids = [rows[i]['source'] for i in ix]
        delta = paired_difference(ids, gains['timesfm'][ix], ids, gains['bolt'][ix])
        between.append(dict(domain=domain, source_ids=ids,
            contrast='(native-hybrid1) TimesFM minus (native-hybrid1) Bolt; positive means larger TimesFM gain',
            **paired_interval(delta, family=3)))
        for model in MODELS:
            vs_seasonal.append(dict(domain=domain, model=model, source_ids=ids,
                hybrid_mse=float(loss[model][ix, 1].mean()), seasonal_mse=float(loss[model][ix, 2].mean()),
                contrast='seasonal24 loss minus hybrid1 loss; positive means hybrid better',
                **hybrid_vs_seasonal(loss[model][ix, 1], loss[model][ix, 2])))
    assert sha(out/'analysis.json') == original_analysis_sha
    assert sha(out/'protocol.json') == original_protocol_sha
    report = dict(status='passed', native_slot2='released; transferred to Carson', additional_native_launched=False,
        paired_between_model_gain=between, paired_hybrid_vs_seasonal=vs_seasonal,
        audit=dict(all_frozen_bindings=True, worker_hashes_and_caps=True, source_order=True,
            independently_recomputed_MSE_max_absolute_error=numeric_error, wins_ties_losses=True,
            independently_recomputed_hindcast_choices=True, frozen_secondary_replayed=True,
            identical_seasonal_control_across_models=True, primary_protocol_and_analysis_unchanged=True),
        execution=execution, worker_gates=workers,
        analysis_sha256=original_analysis_sha, protocol_sha256=original_protocol_sha, audit_source_sha256=sha(__file__),
        evidence_class='Exploratory post-primary paired contrasts; original primary family unchanged',
        caveats=['Difference between models is tested by paired gain differences, never inferred from differing significance',
            'All intervals resample the same64 source pairs within domain; no independent model resampling',
            'Conditional source bootstrap cannot remove shared-calendar dependence; median point outputs evaluated using MSE',
            'Post-primary comparisons do not select new thresholds, depths or deployments'])
    write(out/'audit.json', report)
    knowledge(out, 'analysis', 'complete', dict(primary=a['primary'], diagnostic=a['diagnostic'],
        comparisons=a['comparisons'], conditional=a['conditional'], audit=report),
        'Slot2 released to Carson. Review primary effects alongside seasonal baseline and paired model differences; no natural expansion or FFN native work without a new grant.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/natural_validation_replication4955000'))
    audit(parser.parse_args().out)
