"""Read-only arithmetic reduction of completed dictionary pilot stage records.

No model loading, retraining, TEST selection or population-significance claim.
Write derived tables under a new child directory; preserve all raw stage files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def read(path):
    """Read exact saved stage results."""
    return json.loads(path.read_text(encoding='utf-8'))


def mean(values):
    """Keep missing/dead endpoints missing, not silently zero."""
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def review(root, dest):
    """Produce domain-specific causal tables and prespecified spectral probe rows."""
    dest.mkdir(exist_ok=False)
    fits, forecasts, probes, failures = [], [], [], []
    for stage in sorted(root.glob('fit_*')):
        if not (stage/'execution.json').exists() or read(stage/'execution.json')['returncode']:
            failures.append(stage.name)
            continue
        r = read(stage/'results.json')
        meta = {key: r[key] for key in ('model', 'cut', 'method', 'k', 'seed')}
        fits.append(dict(meta, **{key: r['metrics']['test'][key] for key in ('nmse', 'mean_l0', 'dead_fraction')},
                         parameters=r['budget']['trainable_parameters'],
                         seconds=read(stage/'execution.json')['seconds'],
                         peak_rss=read(stage/'execution.json')['peak_rss']))
        for view in ('code_mean', 'code_rms', 'input_generic', 'metadata_only'):
            for row in r['probes'][view]:
                if 'skipped' in row:
                    continue
                for domain, item in row['test'].items():
                    extra = row.get('metadata_conditioned', {})
                    probes.append(dict(meta, view=view, target=row['target'], budget=row['budget'],
                        domain=domain, n_sources=item['n_sources'], nmse=item['mean'],
                        metadata_conditioned_nmse=extra.get('test', {}).get(domain, {}).get('mean'),
                        metadata_only_nmse=extra.get('metadata_only_test', {}).get(domain, {}).get('mean')))
        native = root/stage.name.replace('fit_', 'native_', 1)
        if not (native/'execution.json').exists() or read(native/'execution.json')['returncode']:
            failures.append(native.name)
            continue
        rows = read(native/'results.json')
        for domain in sorted({row['domain'] for row in rows}):
            subset = [row for row in rows if row['domain'] == domain]
            recs = [row['measurements']['reconstruction'] for row in subset]
            forecasts.append(dict(meta, domain=domain, n_sources=len(subset),
                history_variance_nmse=mean([row['history_variance_nmse'] for row in recs]),
                matched_native_history_variance_nmse=mean([row['history_variance_nmse'] for row in recs if row['random_valid']]),
                random_history_variance_nmse=mean([mean([v['history_variance_nmse'] for v in row['random_metrics']]) for row in recs if row['random_valid']]),
                random_valid=sum(row['random_valid'] for row in recs),
                exact_gates=all(row['exact_gates'] for row in subset),
                sinusoid_reach=sum(row['causal_selectivity']['sinusoid']['reach'] for row in subset),
                noise_reach=sum(row['causal_selectivity']['noise']['reach'] for row in subset),
                ablate1_sinusoid_response_change_nmse=mean([row['causal_selectivity']['sinusoid']['budgets']['1']['response_change_nmse'] for row in subset]),
                ablate1_noise_response_change_nmse=mean([row['causal_selectivity']['noise']['budgets']['1']['response_change_nmse'] for row in subset])))
            for direction in ('sinusoid', 'noise'):
                for budget in ('1', '4'):
                    effects = [row['causal_selectivity'][direction]['budgets'][budget] for row in subset]
                    stem = f'ablate{budget}_{direction}'
                    forecasts[-1][stem+'_valid_random_count'] = sum(row['random_valid'] for row in effects)
                    forecasts[-1][stem+'_matched_response_change_nmse'] = mean([row['response_change_nmse'] for row in effects if row['random_valid']])
                    forecasts[-1][stem+'_random_response_change_nmse'] = mean([row['random_response_change_nmse'] for row in effects if row['random_valid']])
            forecasts[-1]['causal_selectivity_claim_scorable'] = False
            forecasts[-1]['response_null_status'] = 'Independent +/- edits are not a calibrated smooth-map null; zero effects do not certify zero activation edits. See validity audit.'
    for name, rows in [('fit', fits), ('forecast', forecasts), ('probe', probes)]:
        (dest/f'{name}.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
        if rows:
            with (dest/f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = dict(question='What do the completed matched dictionary pilots establish?',
        exact_results=dict(completed_fits=len(fits), forecast_domain_rows=len(forecasts), probe_domain_rows=len(probes),
                           incomplete=failures, fit_table='fit.csv', forecast_table='forecast.csv', probe_table='probe.csv'),
        controls=['frozen FIT-only representations and probes; input/normalizer metadata controls',
                  'source-level domain scores; actual L0; exact native replay and matched random controls'],
        limitations=['No TEST model selection; no general monosemanticity or learned-concept claim',
                     'Four TEST real sources/domain and four synthetic receivers in native pilot',
                     'Random controls compare only against the matching valid receiver subset; unmatched all-receiver preservation remains separate',
                     'Response-change NMSE describes intervention sensitivity, not true-signal accuracy',
                     'Causal response cells require validity/edit-reach audit; fallback nulls are invalid and zero effects are not harmlessness evidence',
                     'Earlier conservative-null wording is superseded: independently sampled +/- edits are not a calibrated smooth-response null',
                     'Native preservation concerns the point forecast; CA15/MN47/CA16 late-head or encoder-output multi-quantile steering are different sites/endpoints',
                     'PCA stores a fitted d-by-d basis; zero neural optimizer parameters is not zero fitted capacity',
                     'FFN and response-aware SAE follow-ups are parked until review'],
        next_step='Review full three-seed/k comparisons, preservation and specific perturbation effects jointly.', status='review_tables')
    (dest/'knowledge.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['exact_results']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    review(args.root, args.dest)
