"""Six frozen depth1 contrasts and continuous hindcast/main association.

Source is the only resampling unit. Primary mean-effect intervals use a fixed
Bonferroni family of six; sign tests describe a different directional estimand.
Diagnostic permutation tests form their own six-comparison exploratory family.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import binomtest, rankdata

from experiments.feature_search.natural_validation_replication import (
    DOMAINS, MODELS, METHODS, ORIGINS, read, write, sha, knowledge)

BOOTSTRAPS = 23999
PERMUTATIONS = 9999


def paired_summary(native, other, family=1, n_boot=None):
    """Paired source bootstrap; ties retained in means and all-source win rates."""
    native, other = np.asarray(native, float), np.asarray(other, float)
    if native.ndim != 1 or native.shape != other.shape or not np.isfinite(native).all() or not np.isfinite(other).all():
        raise ValueError('Finite paired source-loss vectors required')
    if family not in (1, 6):
        raise ValueError('Only prespecified primary6 or descriptive1 families')
    n = len(native)
    if n == 0:
        return dict(n=0, status='no eligible sources', paired_mean_gain=None, gain_interval=None,
                    wins=0, ties=0, losses=0, winrate=None, non_tie_sign_p=1.)
    gain = native-other
    wins, ties, losses = int((gain > 0).sum()), int((gain == 0).sum()), int((gain < 0).sum())
    interval, win_interval = None, None
    if n >= 2:
        rng = np.random.default_rng(4955000)
        count = BOOTSTRAPS if n_boot is None else n_boot
        means, rates = [], []
        for start in range(0, count, 1024):
            ix = rng.integers(n, size=(min(1024, count-start), n))
            draw = gain[ix]
            means.extend(draw.mean(axis=1))
            rates.extend((draw > 0).mean(axis=1))
        tail = .05/(2*family)
        interval = np.quantile(means, [tail, 1-tail]).tolist()
        win_interval = np.quantile(rates, [.025, .975]).tolist()
    return dict(n=n, status='ok' if n >= 2 else 'singleton descriptive only', native_mse=float(native.mean()),
        mse=float(other.mean()), paired_mean_gain=float(gain.mean()), gain_interval=interval,
        interval_family=family, interval_marginal_coverage=1-.05/family,
        interval_scope='paired source bootstrap; approximate familywise95% for primary6, descriptive95% otherwise',
        wins=wins, ties=ties, losses=losses, winrate=wins/n,
        winrate_ci95_descriptive=win_interval, non_tie_winrate=wins/(wins+losses) if wins+losses else None,
        non_tie_sign_p=float(binomtest(wins, wins+losses, .5, alternative='greater').pvalue) if wins+losses else 1.)


def holm6(pvalues):
    """Keep all six declared comparisons, including null/undefined diagnostics."""
    p = np.asarray(pvalues, float)
    if p.shape != (6,) or np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError('Exactly six valid p values required')
    order = np.argsort(p, kind='stable')
    corrected = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(6, 0, -1)))
    result = np.empty(6)
    result[order] = corrected
    return result.tolist()


def correlation(x, y):
    """Return an undefined marker for a constant vector, never a spurious zero."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = x-x.mean(), y-y.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return None if norm == 0 else float(np.clip(np.dot(a, b)/norm, -1, 1))


def continuous_diagnostic(past_benefits, main_benefit, n_perm=None):
    """Positive rank association, no threshold fit; retain temporal disagreement."""
    past, main = np.asarray(past_benefits, float), np.asarray(main_benefit, float)
    if past.shape != (len(main), 2) or not np.isfinite(past).all() or not np.isfinite(main).all():
        raise ValueError('Two past benefits and one main benefit per source required')
    n = len(main)
    if n == 0:
        return dict(n=0, status='no eligible sources', spearman=None, positive_association_p=1.)
    mean = past.mean(axis=1)
    x, y = rankdata(mean), rankdata(main)
    rho = correlation(x, y)
    p = 1.
    if rho is not None:
        x, y = x-x.mean(), y-y.mean()
        denominator = np.linalg.norm(x)*np.linalg.norm(y)
        rng = np.random.default_rng(4955001)
        count, exceed = PERMUTATIONS if n_perm is None else n_perm, 0
        for _ in range(count):
            exceed += np.dot(x, rng.permutation(y))/denominator >= rho-1e-12
        p = (1+int(exceed))/(count+1)
    return dict(n=n, status='ok' if rho is not None else 'constant benefit; association not identifiable',
        spearman=rho, pearson=correlation(mean, main), positive_association_p=p,
        past_mean_benefit=mean.tolist(), main_benefit=main.tolist(),
        per_hindcast_pearson_with_main=[correlation(past[:, j], main) for j in range(2)],
        hindcast_benefit_correlation=correlation(past[:, 0], past[:, 1]),
        mean_abs_between_hindcasts=float(abs(past[:, 0]-past[:, 1]).mean()),
        opposite_sign_hindcasts=int((past[:, 0]*past[:, 1] < 0).sum()),
        either_hindcast_tied=int(((past[:, 0] == 0) | (past[:, 1] == 0)).sum()),
        caveat='Overlapping historical origins and source/calendar heterogeneity; association does not guarantee future temporal stability or independent prospective replication')


def past_choices(forecasts, targets, usable):
    """Strict depth1/native mean loss comparison across exactly two hindcasts."""
    forecasts, targets = np.asarray(forecasts), np.asarray(targets)
    if forecasts.shape[1:] != (2, len(METHODS), 32) or targets.shape != (len(forecasts), 2, 32):
        raise ValueError('Only two past targets accepted')
    losses = ((forecasts-targets[:, :, None])**2).mean(axis=-1)
    valid = np.asarray(usable, bool).all(axis=1)
    choices = valid & (losses[:, :, 1].mean(axis=1) < losses[:, :, 0].mean(axis=1))
    secondary = valid & (losses[:, :, 4].mean(axis=1) < losses[:, :, 0].mean(axis=1))
    return choices, secondary, losses


def bound_targets(out, protocol, name):
    """Open one target artifact only at its authorized analysis phase."""
    if sha(out/name) != protocol['hashes'][name]:
        raise ValueError('Target binding changed')
    with np.load(out/name, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def analyze(out):
    """Persist exact results and failures; never fit a rule on fresh future losses."""
    try:
        result = _analyze(out)
    except Exception as exc:
        knowledge(out, 'analysis', 'failed', dict(error_type=type(exc).__name__, error=str(exc)),
                  'Inspect invalid instrumentation; do not claim replication results.')
        raise
    knowledge(out, 'analysis', 'complete', dict(primary=result['primary'], comparisons=result['comparisons'],
        diagnostic=result['diagnostic'], conditional=result['conditional'], gates=result['gates'],
        analysis_sha256=sha(out/'analysis.json')),
        'Coordinator review of the six adjusted primary effects and source variability; no TEST tuning or automatic expansion.')
    return result


def _analyze(out):
    """Freeze choices from past targets, then evaluate all fresh sources once."""
    if any((out/name).exists() for name in ('past_choices.json', 'analysis.json')):
        raise FileExistsError('Refuse repeated analysis')
    p = read(out/'protocol.json')
    for path, digest in {**p['sources'], **p['pilot_bindings']}.items():
        if sha(path) != digest:
            raise ValueError('Frozen source/pilot binding changed')
    for name in ('rows.json', 'plans.json', 'declaration.json', 'availability.json'):
        if sha(out/name) != p['hashes'][name]:
            raise ValueError('Frozen input changed')
    rows, plans = read(out/'rows.json'), read(out/'plans.json')
    if len(rows) != 192 or [r['source'] for r in rows] != p['source_ids'] or len(set(p['source_ids'])) != 192:
        raise ValueError('Expected192 uniquely bound source IDs')
    if any(r['split'] != 'TEST' for r in rows) or any(sum(r['domain'] == d for r in rows) != 64 for d in DOMAINS):
        raise ValueError('All64/domain must remain fresh evaluation sources')
    execution, predictions, selections, hindloss = read(out/'execution.json'), {}, {}, {}
    past = bound_targets(out, p, 'hindcast_targets.npz')['targets']
    usable = [[q['selector_usable'] for q in source[:2]] for source in plans]
    for model in MODELS:
        worker = read(out/f'{model}_worker.json')
        if (not execution[model]['valid'] or not worker['repeat_exact'] or not worker['finite']
                or worker['closure_fp32'] > 2e-6 or worker['peak_rss'] > p['worker_cap']
                or worker['seconds'] > p['seconds_per_model'] or worker['protocol_sha256'] != sha(out/'protocol.json')
                or worker['sha256'] != sha(out/f'{model}_forecasts.npz')):
            raise ValueError('Native worker/resource/hash gate failed')
        with np.load(out/f'{model}_forecasts.npz', allow_pickle=False) as f:
            if f['source_ids'].tolist() != p['source_ids'] or f['methods'].tolist() != list(METHODS) or f['origins'].tolist() != list(ORIGINS):
                raise ValueError('Forecast identity mismatch')
            pred = f['forecasts']
        if pred.shape != (192, 3, len(METHODS), 32) or not np.isfinite(pred).all():
            raise ValueError('Forecast shape/finite gate failed')
        chosen, secondary, loss = past_choices(pred[:, :2], past, usable)
        selections[model] = dict(depth1=chosen.tolist(), secondary=secondary.tolist())
        predictions[model], hindloss[model] = pred, loss
    write(out/'past_choices.json', dict(source_ids=p['source_ids'], choices=selections,
        frozen_pilot_rules=p['frozen_pilot_rules'], protocol_sha256=sha(out/'protocol.json'),
        forecast_hashes={m:sha(out/f'{m}_forecasts.npz') for m in MODELS}))
    main = bound_targets(out, p, 'main_targets.npz')
    if main['source_ids'].tolist() != p['source_ids']:
        raise ValueError('Main target source order mismatch')
    names = list(METHODS)+['past_selector1', 'pilot_rule']
    primary, comparisons, diagnostic, conditional, records, full_forecasts = [], [], [], [], [], {}
    for model in MODELS:
        f = predictions[model][:, 2]
        choose = np.asarray(selections[model]['depth1'])
        selected = np.where(choose[:, None], f[:, 1], f[:, 0])
        pilot = f[:, 4].copy()
        for i, row in enumerate(rows):
            if p['frozen_pilot_rules'][model][row['domain']] == 'past_selector3' and not selections[model]['secondary'][i]:
                pilot[i] = f[i, 0]
        matrix = np.concatenate([f, selected[:, None], pilot[:, None]], axis=1)
        full_forecasts[model] = matrix
        losses = ((matrix-main['targets'][:, None])**2).mean(axis=-1)
        past_benefit = hindloss[model][:, :, 0]-hindloss[model][:, :, 1]
        main_benefit = losses[:, 0]-losses[:, 1]
        for i, row in enumerate(rows):
            records.append(dict(row, model=model, losses=dict(zip(names, losses[i].tolist())),
                past_benefits=past_benefit[i].tolist(), main_benefit=float(main_benefit[i]),
                selector_chose_hybrid=bool(choose[i]), main_eligible=not plans[i][2]['abstained'],
                all_origins_eligible=all(not q['abstained'] for q in plans[i]),
                extracted_depths=[len(q['details']) for q in plans[i]],
                origin_tone_details=[q['details'] for q in plans[i]]))
        for domain in DOMAINS:
            ix = np.asarray([i for i, row in enumerate(rows) if row['domain'] == domain])
            effect = paired_summary(losses[ix, 0], losses[ix, 1], family=6)
            primary.append(dict(model=model, domain=domain, method='hybrid1', **effect))
            for k, name in enumerate(names):
                if name not in ('native', 'hybrid1', 'pilot_candidate'):
                    comparisons.append(dict(model=model, domain=domain, method=name,
                        scope='secondary/descriptive, outside six-primary mean family',
                        **paired_summary(losses[ix, 0], losses[ix, k])))
            diagnostic.append(dict(model=model, domain=domain,
                **continuous_diagnostic(past_benefit[ix], main_benefit[ix])))
            for label, mask in [('main_extraction', [not plans[i][2]['abstained'] for i in ix]),
                                ('all_origins_extraction', [all(not q['abstained'] for q in plans[i]) for i in ix])]:
                selected_ids = ix[mask]
                conditional.append(dict(model=model, domain=domain, eligibility=label,
                    total_sources=len(ix), eligible_sources=len(selected_ids), eligibility_rate=len(selected_ids)/len(ix),
                    scope='Prespecified past-only conditioning; descriptive, outside primary family',
                    fixed=paired_summary(losses[selected_ids, 0], losses[selected_ids, 1]),
                    selector=paired_summary(losses[selected_ids, 0], losses[selected_ids, 5]),
                    continuous=continuous_diagnostic(past_benefit[selected_ids], main_benefit[selected_ids])))
    for item, adjusted in zip(primary, holm6([v['non_tie_sign_p'] for v in primary])):
        item['sign_holm6_p'] = adjusted
        item['positive_adjusted_mean_interval'] = item['gain_interval'][0] > 0
        item['sign_scope'] = 'Supportive non-tie majority endpoint, not a population mean-effect p value'
    for item, adjusted in zip(diagnostic, holm6([v['positive_association_p'] for v in diagnostic])):
        item['diagnostic_holm6_p'] = adjusted
        item['family_scope'] = 'Separate exploratory association family; not extra primary confirmation'
    np.savez_compressed(out/'evaluated_forecasts.npz', **full_forecasts,
        methods=np.asarray(names), source_ids=np.asarray(p['source_ids']))
    result = dict(primary=primary, comparisons=comparisons, diagnostic=diagnostic,
        conditional=conditional, source_records=records,
        metric='Own-history standardized point-forecast MSE; native outputs are medians (Bolt quantile .5, TimesFM aridx5). Positive gain = native loss minus method loss',
        uncertainty='Source resampling within domain; approximate Bonferroni95% family intervals across six primary mean effects. Secondary/conditional intervals descriptive. Winrate/sign endpoint differs from mean effect.',
        temporal_caveat='Two hindcasts overlap main history; source/calendar dependence and temporal instability limit transfer of retrospective association. No TEST threshold tuning.',
        gates=dict(native_valid=True, source_ids=True, all192_evaluation=True, primary_family6=True,
            choices_before_main_targets=True, pilot_rules_frozen=True, no_future_fit=True),
        hashes={name:sha(out/name) for name in ('protocol.json', 'past_choices.json', 'evaluated_forecasts.npz')})
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/natural_validation_replication4955000'))
    analyze(parser.parse_args().out)
