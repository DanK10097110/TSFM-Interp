"""Model-free replay and static report for the five real-background pilot screens.

This module reads saved arrays, never invokes a checkpoint and never promotes a
failed screen. Source-level paired contrasts remain descriptive within datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def read(path):
    """Read a persisted record."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """Hash files independently of the experiment runner."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def analyze(run):
    """Reconstruct losses, gates and source pairing without invoking native models."""
    protocol = read(run/'protocol.json')
    metadata = read(run/'metadata.json')
    execution = read(run/'execution.json')
    data = np.load(run/'corpus.npz')['series'].astype(float)
    fit = np.array([i for i, r in enumerate(metadata) if r['split'] == 'fit'])
    test = np.array([i for i, r in enumerate(metadata) if r['split'] == 'evaluation'])
    ids = [(r['dataset'], r['id']) for r in metadata]
    assert len(set(ids)) == len(ids)
    assert not set(fit).intersection(test)
    assert len(set(r['window_sha256'] for r in metadata)) == len(metadata)
    assert protocol['corpus_sha256'] == digest(run/'corpus.npz')
    assert protocol['metadata_sha256'] == digest(run/'metadata.json')
    datasets = list(dict.fromkeys(r['dataset'] for r in metadata))
    a = .5*np.sin(2*np.pi*np.arange(32)/24+.3)
    b = .5*np.sin(2*np.pi*np.arange(32)/40-.7)
    energy = float(np.mean(a*a))
    series_metrics, results, checks = {}, {}, {}
    for model in protocol['checkpoints']:
        assert execution[model]['valid']
        result = read(run/f'{model}.json')
        assert result['peak_rss_bytes'] <= protocol['budget']['resident_bytes']
        assert result['elapsed_seconds'] <= protocol['budget']['model_seconds']
        assert result['arrays_sha256'] == digest(run/f'{model}_arrays.npz')
        with np.load(run/f'{model}_arrays.npz') as saved:
            arrays = {k: saved[k] for k in saved.files}
        assert all(np.isfinite(value).all() for value in arrays.values())
        f = arrays['forecast_base'].astype(float)
        x = data[:, 128:256]
        e = result['experiments']
        losses = {}
        for suffix, sign in [('plus', 1), ('minus', -1)]:
            fa = arrays['forecast_'+suffix].astype(float)
            fb = arrays['forecast_b'+suffix].astype(float)
            both = arrays['forecast_both'+suffix].astype(float)
            interaction = both-fa-fb+f
            np.testing.assert_allclose(interaction, arrays['interaction_'+suffix], atol=2e-6)
            target = data[:, 256:]+sign*(a+b)
            losses[suffix] = np.mean((both-target)**2, axis=1)-np.mean((fa+fb-f-target)**2, axis=1)
            response_error = np.mean((fa-f-sign*a)**2, axis=1)/energy
            np.testing.assert_allclose(response_error, result['component_response'][suffix]['error'], atol=2e-6, rtol=1e-5)
            tr = e['3']['polarities'][suffix]
            def loss(key):
                return np.mean((arrays[f'transfer_{suffix}_{key}'].astype(float)-fa[test])**2, axis=1)/energy
            for key in ['near', 'far']:
                np.testing.assert_allclose(loss(key), tr[key+'_loss'], atol=2e-6)
            np.testing.assert_allclose(loss('oracle'), tr['oracle_error'], atol=2e-6)
            assert tr['checks']['oracle'] == bool(np.all(loss('oracle') <= .05))
            assert tr['checks']['natural'] == bool(np.all(response_error[test] <= .5))
            assert tr['checks']['geometry'] == (not any(tr['invalid_geometry']))
            improvements = [np.mean((f[test]-fa[test])**2, axis=1)/energy-loss('near'),
                            (loss('random0')+loss('random1'))/2-loss('near')]
            for label, vector in zip(['improvement', 'specificity'], improvements):
                means = {dataset: float(vector[[j for j, ix in enumerate(test) if metadata[ix]['dataset'] == dataset]].mean()) for dataset in datasets}
                np.testing.assert_allclose(list(means.values()), list(tr[label]['dataset_means'].values()), atol=2e-6)
                assert tr[label]['passed'] == all(v >= tr[label]['threshold'] for v in means.values())
        harm = (losses['plus']+losses['minus'])/2
        np.testing.assert_allclose(harm, e['2']['harmful_interaction'], atol=2e-6)
        predictions = e['2']['predictions']
        for key, by_dataset in e['2']['improvements'].items():
            for dataset in datasets:
                ix = [i for i in test if metadata[i]['dataset'] == dataset]
                internal = np.mean((np.asarray(predictions['internal_ridge'])[ix]-harm[ix])**2)
                baseline = np.mean((np.asarray(predictions[key])[ix]-harm[ix])**2)
                np.testing.assert_allclose(1-internal/max(baseline, 1e-12), by_dataset[dataset], atol=1e-4)
        predicted_pass = all(v >= .2 for row in e['2']['improvements'].values() for v in row.values()) and np.mean(harm[test]) >= .05
        assert e['2']['passed'] == bool(predicted_pass)
        assert e['3']['passed'] == all(all(row['checks'].values()) and row['improvement']['passed'] and row['specificity']['passed'] for row in e['3']['polarities'].values())
        scale_errors = []
        for index, (scale, shift) in enumerate(protocol['transforms']):
            fp, fa, fm = arrays[f'transform_{index}'].astype(float)
            errors = [np.sqrt(np.mean(((fp-shift)/scale-f)**2, axis=1)),
                      np.sqrt(np.mean(((fa-fp)/scale-(arrays['forecast_plus']-f))**2, axis=1)),
                      np.sqrt(np.mean(((fm-fp)/scale-(arrays['forecast_minus']-f))**2, axis=1))]
            scale_errors.append(np.max(errors, axis=0))
        scale_error = np.max(scale_errors, axis=0)
        for dataset in datasets:
            ix = [i for i in test if metadata[i]['dataset'] == dataset]
            np.testing.assert_allclose(scale_error[ix].mean(), e['4']['dataset_means'][dataset], atol=3e-6)
        assert e['4']['passed'] == (all(v >= .1 for v in e['4']['dataset_means'].values()) and e['4']['classical_max_error'] < 1e-5)
        for label, control in [('short_advantage', 'short'), ('irrelevant_advantage', 'irrelevant')]:
            vector = np.asarray(e['5']['errors'][control])-np.asarray(e['5']['errors']['long'])
            means = {dataset: float(vector[[i for i in test if metadata[i]['dataset'] == dataset]].mean()) for dataset in datasets}
            np.testing.assert_allclose(list(means.values()), list(e['5'][label]['dataset_means'].values()))
            assert e['5'][label]['passed'] == all(v >= .1 for v in means.values())
        assert e['5']['boundary_valid'] == bool(np.all(np.asarray(e['5']['prefix_step_energy_ratios'])[:, test] <= 2))
        assert e['5']['passed'] == (e['5']['short_advantage']['passed'] and e['5']['irrelevant_advantage']['passed'] and e['5']['boundary_valid'])
        oracle = arrays['transfer_plus_oracle'].astype(float)
        for refit in e['6']['refits']:
            pred = arrays[f"sae_{refit['seed']}_forecast"].astype(float)
            target = arrays['forecast_plus'][test].astype(float)
            ferror = np.mean((pred-target)**2)/np.mean((target-f[test])**2)
            np.testing.assert_allclose(ferror, refit['forecast_response_nmse'], rtol=1e-5)
        fidelity = all(r['heldout_reconstruction_nmse'] <= .25 and r['heldout_delta_nmse'] <= .25 and r['forecast_response_nmse'] <= .1 and max(r['permutation_max_error'], r['duplicate_max_error']) < 1e-4 for r in e['6']['refits'])
        assert e['6']['fidelity_valid'] == fidelity
        assert e['6']['passed'] == (fidelity and all(r['aligned_support_jaccard'] <= .5 and r['decoded_forecast_disagreement'] <= .1 for r in e['6']['pairs']))
        series_metrics[model] = dict(harm=harm, scaling_rms=scale_error,
                                    history_gain=np.asarray(e['5']['errors']['short'])-np.asarray(e['5']['errors']['long']))
        checks[model] = dict(identity=result['identity'], peak_rss_bytes=result['peak_rss_bytes'],
                             short_history_and_transfer_waveform_replay=True, finite_arrays=True,
                             oracle_response_nmse=float(np.mean((oracle-arrays['forecast_plus'][test])**2)/energy))
        results[model] = result
    contrasts = {}
    for measure in ['harm', 'scaling_rms', 'history_gain']:
        difference = series_metrics['timesfm'][measure]-series_metrics['bolt'][measure]
        contrasts[measure] = {dataset: dict(mean=float(difference[[i for i in test if metadata[i]['dataset'] == dataset]].mean()),
                                           source_min=float(difference[[i for i in test if metadata[i]['dataset'] == dataset]].min()),
                                           source_max=float(difference[[i for i in test if metadata[i]['dataset'] == dataset]].max()))
                              for dataset in datasets}
    return dict(passed=True, checks=checks, paired_timesfm_minus_bolt=contrasts,
                scope='Independent model-free short-history/transfer/scaling/SAE-functional losses and all decision gates. History gates use stored per-source losses: long/control unmodified forecast arrays were not retained in this pilot. SAE reconstruction/geometry diagnostics are read, not independently regenerated. Source min/max are descriptive ranges, not confidence intervals'), results


def render(run, output):
    """Publish lightweight records, a figure, and an HTML account of closed screens."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit, results = analyze(run)
    output.mkdir(parents=True, exist_ok=True)
    for name in ['summary.json', 'protocol.json', 'metadata.json', 'execution.json', 'mutation_checks.json', 'bolt.json', 'timesfm.json']:
        record = read(run/name)
        if name == 'protocol.json':
            for checkpoint in record['checkpoints'].values():
                checkpoint.pop('path', None)
        if name == 'metadata.json':
            for row in record:
                row.pop('source_file', None)
        if name in ['bolt.json', 'timesfm.json']:
            record['checkpoint'].pop('path', None)
        (output/name).write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    (output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    summary = read(run/'summary.json')
    if any(e['passed'] for result in results.values() for e in result['experiments'].values()):
        raise ValueError('This fixed negative-pilot narrative must be revised before reporting a positive screen')
    names = {'2': 'Four-input interaction', '3': 'Transfer across backgrounds', '4': 'Units and scaling',
             '5': 'Useful versus irrelevant history', '6': 'SAE measurement stability'}
    rows = []
    for key, title in names.items():
        outcomes = []
        for model in ['bolt', 'timesfm']:
            e = results[model]['experiments'][key]
            if e['passed']:
                text = 'Qualifies for fresh-source replication'
            elif key == '3':
                failures = sorted({k for v in e['polarities'].values() for k, passed in v['checks'].items() if not passed})
                text = 'Stop; failed '+', '.join(failures)+' validity gates'
            elif key == '4':
                text = f"No practical scaling defect (largest dataset mean {max(e['dataset_means'].values()):.3g})"
            elif key == '5':
                text = 'Stop; effect gates fail'+('; history-control discontinuity fails' if not e['boundary_valid'] else '')
            elif key == '6':
                text = 'Stop; response reconstruction fails fidelity gate'
            else:
                text = 'Stop; internal predictor does not beat input controls consistently'
            outcomes.append(html.escape(text))
        rows.append(f'<tr><td>{key}. {title}</td><td>{outcomes[0]}</td><td>{outcomes[1]}</td></tr>')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    labels = ['Electricity', 'M4 hourly', 'Pedestrians']
    colors = {'bolt': '#177e89', 'timesfm': '#c65d32'}
    for model, result in results.items():
        axes[0].plot(labels, list(result['experiments']['4']['dataset_means'].values()), 'o-', color=colors[model], label=model)
        axes[1].plot(labels, list(result['experiments']['5']['short_advantage']['dataset_means'].values()), 'o-', color=colors[model], label=model)
        refits = result['experiments']['6']['refits']
        axes[2].scatter([r['heldout_reconstruction_nmse'] for r in refits], [r['heldout_delta_nmse'] for r in refits], color=colors[model], label=model)
    axes[0].set(yscale='log', title='Scaling is consistent', ylabel='Worst transform RMS, source SD units')
    axes[1].axhline(.1, color='gray', linestyle='--', label='Screen gate')
    axes[1].axhline(0, color='gray', linewidth=.5)
    axes[1].set(title='Longer history: uneven benefit', ylabel='Short minus long response loss')
    axes[2].axhline(.25, color='gray', linestyle='--')
    axes[2].axvline(.25, color='gray', linestyle='--')
    axes[2].set(title='Good state fit can miss the response', xlabel='Heldout state reconstruction NMSE', ylabel='Heldout response reconstruction NMSE')
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.savefig(output/'pilot_results.png', dpi=160)
    plt.close(fig)
    links = ' · '.join(f'<a href="{name}">{name}</a>' for name in ['summary.json', 'protocol.json', 'audit.json', 'bolt.json', 'timesfm.json', 'mutation_checks.json'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Section 49: real-background pilots</title><style>body{{max-width:1150px;margin:35px auto;padding:0 20px;font:17px/1.6 system-ui;color:#20303b}}h1{{line-height:1.2}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc9ce;padding:12px;text-align:left;vertical-align:top}}th{{background:#edf4f5}}img{{width:100%}}small{{color:#52616b}}a{{color:#126477}}</style>
<h1>Five small real-background experiments</h1><p>Two checkpoints, 24 hourly source series, 12 discovery and 12 heldout sources. <strong>No pilot meets its frozen expansion gate.</strong> These are bounded exploratory screens, not evidence that the broader research ideas are false.</p>
<table><thead><tr><th>Experiment</th><th>Chronos-Bolt small</th><th>TimesFM 2.5 200M</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<figure><img src="pilot_results.png" alt="Scaling errors remain tiny, history benefits differ by dataset, and SAE response errors exceed the fidelity gate."><figcaption>Source-paired measurements, grouped by dataset. Each SAE point is one matched-budget refit. No confidence or significance claims are made.</figcaption></figure>
<h2>What these measurements establish</h2><p>Changing units leaves forecasts and component responses nearly unchanged on these inputs. Interaction features do not consistently improve heldout prediction beyond input controls. Raw hidden-state transfer is not interpretable without natural-response and normalization closure checks. The old-history control has a discontinuity failure, so this screen cannot isolate useful-history effects. Small SAEs reconstruct states better than they reconstruct the controlled change; unstable feature IDs therefore cannot establish unstable model semantics.</p>
<h2>Scope and next allocation</h2><p>The same sources, hourly sampling, horizon32, history128/256, phase and relative amplitude were used for both models. Native tokenization, training and normalization differ. Last-block output is the declared intervention role; equal architectural depth is not asserted. Real sources are separate from the sealed benchmark. Classical seasonal-naive and Fourier/trend baselines and separate forecast-and-sum controls are included in the records. Only three datasets are represented; pretraining overlap is unknown.</p>
<p>No broader native sweep, larger SAE, rank search or training run follows these negative screens. IDs1/7/8 remain parked; IDs7/8 still lack the positive mechanism or diagnostic needed to justify their compute.</p>
<h2>Reproducibility</h2><p>{links}</p><p>Protocol frozen before inference. All six planted regressions are detected. Independent model-free replay verifies saved losses, source pairing, finite arrays and resource bounds. Captured short-history states, intervention forecasts and SAE weights remain in the run directory. The audit replays short-history, transfer, scaling and SAE functional losses. History gates use stored source losses because the unmodified long/control forecasts were not retained; SAE reconstruction and geometry diagnostics are not independently regenerated. Only lightweight records are published here.</p><p><small>Evidence: exploratory behavioral and within-model interventions. Failed validity and fidelity gates are distinguished from scientific negatives. ROADMAP §49.24–49.25.</small></p></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.out), indent=2))
