"""Model-free analysis, numerical audit and scientific report for Fourier pilots.

All fixed cells remain visible. Signed improvements and degradations may nominate
future work; the check half is never used to choose decompositions or thresholds.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

import numpy as np
from experiments.feature_search import fourier_adapter as adapter
from experiments.feature_search.fourier_poc import read, write, digest, MODELS


def label_positive_counts(value):
    """Positive losses are not favorable outcomes; label generic bootstrap counts."""
    if isinstance(value, dict):
        return {('positive_sources' if key == 'favorable_sources' else key): label_positive_counts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [label_positive_counts(item) for item in value]
    return value


def response_score(response, sign=1):
    """Known-addition fidelity, including signed gain and off-target energy."""
    t = np.arange(32)
    expected = sign*.5*np.sin(2*np.pi*5*t/128+.3)
    basis = np.stack([np.sin(2*np.pi*5*t/128), np.cos(2*np.pi*5*t/128)], axis=1)
    coefficient = np.linalg.lstsq(basis, response, rcond=None)[0]
    target = np.linalg.lstsq(basis, expected, rcond=None)[0]
    energy = float(np.mean(expected**2))
    phase = float(np.angle(np.exp(1j*(np.arctan2(coefficient[1], coefficient[0])-np.arctan2(target[1], target[0])))))
    return dict(error=float(np.mean((response-expected)**2)/energy),
                gain=float(np.dot(response, expected)/np.dot(expected, expected)),
                absolute_phase_error=abs(phase),
                off_target=float(np.mean((response-basis@coefficient)**2)/energy))


def key_for(row):
    """Pair additions only within their exact source/distractor condition."""
    return (row['source'], row.get('distractor_bin'), row.get('distractor_amplitude'))


def evaluate(run, model):
    """Recombine saved predictions; check every tree reconstruction and split closure."""
    protocol, rows, plans = [read(run/name) for name in ['protocol.json', 'rows.json', 'plans.json']]
    execution = read(run/'execution.json')[model]
    if not execution['valid']:
        raise ValueError(f'{model} execution invalid: {execution}')
    for name, expected in protocol['artifact_hashes'].items():
        if digest(run/name) != expected:
            raise ValueError('Frozen artifact changed: '+name)
    record = read(run/f'{model}_execution.json')
    if digest(run/f'{model}_forecasts.npz') != record['prediction_sha256']:
        raise ValueError('Forecast binding changed')
    predictions = np.load(run/f'{model}_forecasts.npz')['predictions'].astype(float)
    bank = np.load(run/'bank.npz')['inputs'].astype(float)
    with np.load(run/'corpus.npz') as corpus:
        history, future = corpus['history'], corpus['future']
    if predictions.shape != (len(bank), 32) or not np.isfinite(predictions).all():
        raise ValueError('Nonfinite/missing forecasts')
    zero = predictions[protocol['zero_id']]
    metrics, waveforms, classical_predictions = [], [], []
    maximum_reconstruction, maximum_telescoping = 0., 0.
    for index, (row, plan) in enumerate(zip(rows, plans)):
        datum = dict(index=index, source=row['source'], domain=row['domain'], family=row['family'],
                     split=row['split'], scale=row['scale'], methods={})
        waves = {}
        schemes = dict(plan['schemes'])
        if 'linear_detrend' in plan:
            schemes['linear_detrend'] = plan['linear_detrend']['schemes']['contiguous']
        for name, scheme in schemes.items():
            losses, raw_losses, current, active_groups, component_rms = {}, {}, {}, {}, {}
            for level, nodes in scheme.items():
                ids = [node['bank_id'] for node in nodes]
                rebuilt = bank[ids].sum(0)
                rms = np.sqrt(np.mean(bank[ids]**2, axis=1))
                component_rms[level] = rms.tolist()
                active_groups[level] = int(np.sum(rms > 1e-10*max(float(np.sqrt(np.mean(history[index]**2))), 1e-12)))
                reconstruction = float(np.max(abs(rebuilt-history[index]))/max(1, np.max(abs(history[index]))))
                maximum_reconstruction = max(maximum_reconstruction, reconstruction)
                if reconstruction > 2e-6:
                    raise ValueError('Injected contexts do not reconstruct the root')
                corrected = adapter.combine(predictions[ids], zero)
                raw = adapter.combine(predictions[ids], zero, corrected=False)
                independent = sum(predictions[i]-zero for i in ids)+zero
                np.testing.assert_allclose(corrected, independent, atol=1e-12)
                if level == '1' and not np.array_equal(corrected, predictions[ids[0]]):
                    raise ValueError('K1 native identity failed')
                current[level] = corrected
                losses[level] = adapter.error(corrected, future[index], row['scale'])
                raw_losses[level] = adapter.error(raw, future[index], row['scale'])
            nodescores = []
            for level, nodes in scheme.items():
                nextlevel = str(int(level)*2)
                if nextlevel not in scheme:
                    continue
                changes = []
                for node in nodes:
                    children = [n for n in scheme[nextlevel] if set(n['leaves']).issubset(node['leaves'])]
                    if len(children) != 2:
                        raise ValueError('Tree does not have exactly two children')
                    p = predictions[node['bank_id']]
                    left, right = [predictions[c['bank_id']] for c in children]
                    delta = left+right-p-zero
                    changes.append(delta)
                    hybrid = current[level]+delta
                    nodescores.append(dict(level=int(level), leaves=node['leaves'],
                                           interaction_energy=float(np.mean(delta**2)/row['scale']**2),
                                           split_loss_gain=losses[level]-adapter.error(hybrid, future[index], row['scale'])))
                closure = float(np.max(abs(current[nextlevel]-current[level]-sum(changes))))
                maximum_telescoping = max(maximum_telescoping, closure)
                if closure > 1e-10:
                    raise ValueError('Split changes fail forecast telescoping')
            datum['methods'][name] = dict(loss=losses, raw_loss=raw_losses,
                                          first_crossing=adapter.first_crossing(losses), nodes=nodescores,
                                          active_groups=active_groups, component_rms=component_rms)
            waves[name] = current
        if 'matched_random' in schemes:
            from scipy.optimize import linear_sum_assignment
            first = bank[[n['bank_id'] for n in schemes['contiguous']['4']]]
            second = bank[[n['bank_id'] for n in schemes['matched_random']['4']]]
            cost = np.mean((first[:, None]-second[None])**2, axis=-1)/row['scale']**2
            a, b = linear_sum_assignment(cost)
            datum['random_grouping_effective_change'] = float(np.sqrt(cost[a, b].sum()))
            datum['random_grouping_nontrivial'] = datum['random_grouping_effective_change'] > 1e-6
        direct = adapter.classical(history[index])
        seasonal = adapter.classical(history[index], 'seasonal')
        leaves = adapter.decompose(history[index])
        decomposed = sum(adapter.classical(leaf) for leaf in leaves)
        datum['classical'] = dict(spectral=adapter.error(direct, future[index], row['scale']),
                                  seasonal=adapter.error(seasonal, future[index], row['scale']),
                                  decomposed_spectral=adapter.error(decomposed, future[index], row['scale']))
        classical_predictions.append(direct)
        metrics.append(datum); waveforms.append(waves)
    base_index = {key_for(row): i for i, row in enumerate(rows) if row.get('sign') == 0}
    responses = []
    for index, row in enumerate(rows):
        if row.get('sign') not in (-1, 1):
            continue
        baseline = base_index[key_for(row)]
        if rows[baseline]['split'] != row['split']:
            raise ValueError('Paired sources crossed evaluation split')
        response = dict(index=index, source=row['source'], domain=row['domain'], family=row['family'],
                        split=row['split'], sign=row['sign'], distractor_bin=row.get('distractor_bin'),
                        distractor_amplitude=row.get('distractor_amplitude'), methods={})
        for level in waveforms[index]['contiguous']:
            value = waveforms[index]['contiguous'][level]-waveforms[baseline]['contiguous'][level]
            response['methods'][level] = response_score(value, row['sign'])
        response['spectral'] = response_score(classical_predictions[index]-classical_predictions[baseline], row['sign'])
        response['periodic128_posthoc'] = response_score(history[index, :32]-history[baseline, :32], row['sign'])
        responses.append(response)
    return dict(model=model, rows=metrics, responses=responses,
                audit=dict(max_relative_input_reconstruction=maximum_reconstruction,
                           max_forecast_telescoping=maximum_telescoping,
                           native_identity_exact=True, finite_bound_forecasts=True),
                execution=record)


def aggregate(result, metadata):
    """Summaries retain each model/domain/family and direct paired contrasts."""
    output = dict(curves={}, grouping={}, phase={}, responses={}, distractor_contrasts={}, candidates=[])
    rows = result['rows']
    for domain, family in sorted({(r['domain'], r['family']) for r in rows if r['family'] != 'distractor'}):
        for split in ['discovery', 'check']:
            selected = [r for r in rows if (r['domain'], r['family'], r['split']) == (domain, family, split)
                        and metadata[r['index']].get('sign', 0) == 0]
            if not selected:
                continue
            ids = [r['source'] for r in selected]
            key = '/'.join([domain, family, split])
            curves = {}
            for level in ['1', '2', '4', '8']:
                loss = [r['methods']['contiguous']['loss'][level] for r in selected]
                gain = [r['methods']['contiguous']['loss']['1']-v for r, v in zip(selected, loss)]
                curves[level] = dict(loss=adapter.conditional_interval(loss, ids), gain=adapter.conditional_interval(gain, ids))
            output['curves'][key] = dict(levels=curves,
                classical={kind: adapter.conditional_interval([r['classical'][kind] for r in selected], ids)
                           for kind in ['spectral', 'seasonal', 'decomposed_spectral']},
                crossings={str(k): sum(r['methods']['contiguous']['first_crossing'] == k for r in selected) for k in [1, 2, 4, 8, None]},
                nonmonotone_sources=sum(any(r['methods']['contiguous']['loss'][str(k*2)] > r['methods']['contiguous']['loss'][str(k)] for k in [1, 2, 4]) for r in selected))
            if all('interleaved' in r['methods'] for r in selected):
                output['grouping'][key] = {method: adapter.conditional_interval(
                    [r['methods']['contiguous']['loss']['4']-r['methods'][method]['loss']['4'] for r in selected], ids)
                    for method in ['interleaved', 'matched_random', 'linear_detrend']}
    for split in ['discovery', 'check']:
        stationary = {r['source']: r for r in rows if r['family'] == 'stationary' and r['split'] == split}
        for family in ['phase_coherent', 'phase_matched']:
            selected = [r for r in rows if r['family'] == family and r['split'] == split]
            output['phase'][family+'/'+split] = {level: adapter.conditional_interval(
                [r['methods']['contiguous']['loss'][level]-stationary[r['source']]['methods']['contiguous']['loss'][level] for r in selected],
                [r['source'] for r in selected]) for level in ['1', '4', '8']}
    responses = result['responses']
    cells = sorted({(r['domain'], r['family'], r['distractor_bin'] or 0, r['distractor_amplitude'] or 0) for r in responses})
    for domain, family, frequency, amplitude in cells:
        for split in ['discovery', 'check']:
            selected = [r for r in responses if (r['domain'], r['family'], r['distractor_bin'] or 0, r['distractor_amplitude'] or 0, r['split']) == (domain, family, frequency, amplitude, split)]
            key = '/'.join(map(str, [domain, family, frequency, amplitude, split]))
            ids = [r['source'] for r in selected]
            output['responses'][key] = {level: {metric: adapter.conditional_interval([r['methods'][level][metric] for r in selected], ids)
                                               for metric in ['error', 'gain', 'absolute_phase_error', 'off_target']}
                                       for level in selected[0]['methods']}
            output['responses'][key]['spectral'] = adapter.conditional_interval([r['spectral']['error'] for r in selected], ids)
            output['responses'][key]['periodic128_posthoc'] = adapter.conditional_interval([r['periodic128_posthoc']['error'] for r in selected], ids)
            output['responses'][key]['paired_K4_gain'] = adapter.conditional_interval([r['methods']['1']['error']-r['methods']['4']['error'] for r in selected], ids)
    for frequency in [6, 17]:
        for split in ['discovery', 'check']:
            low = {(r['source'], r['sign']): r for r in responses if r['family'] == 'distractor' and r['distractor_bin'] == frequency and r['distractor_amplitude'] == .5 and r['split'] == split}
            high = {(r['source'], r['sign']): r for r in responses if r['family'] == 'distractor' and r['distractor_bin'] == frequency and r['distractor_amplitude'] == 8. and r['split'] == split}
            assert set(low) == set(high)
            output['distractor_contrasts'][f'{frequency}/{split}'] = {level: adapter.conditional_interval(
                [high[k]['methods'][level]['error']-low[k]['methods'][level]['error'] for k in low],
                [k[0] for k in low]) for level in ['1', '4']}
    for section in ['grouping', 'phase']:
        for key in output[section]:
            if not key.endswith('/discovery'): continue
            check_key = key.replace('/discovery', '/check')
            for contrast, estimate in output[section][key].items():
                discovery = estimate['mean']
                check = output[section][check_key][contrast]['mean']
                if abs(discovery) >= .1 and abs(check) >= .1 and discovery*check > 0:
                    output['candidates'].append(dict(contrast=section+'/'+contrast, cell=key.rsplit('/', 1)[0],
                                                      discovery=discovery, check=check,
                                                      direction='positive contrast' if check > 0 else 'negative contrast'))
    for key in output['curves']:
        if not key.endswith('/discovery'):
            continue
        discovery = output['curves'][key]['levels']['4']['gain']['mean']
        check = output['curves'][key.replace('/discovery', '/check')]['levels']['4']['gain']['mean']
        if abs(discovery) >= .1 and abs(check) >= .1 and discovery*check > 0:
            output['candidates'].append(dict(contrast='K4 minus native fidelity gain', cell=key.rsplit('/', 1)[0],
                                              discovery=discovery, check=check, direction='improvement' if check > 0 else 'degradation'))
    return label_positive_counts(output)


def paired_models(results, metadata):
    """Differences are paired on source and condition, never compared p-values."""
    output = {}
    by_model = {model: {r['index']: r for r in result['rows']} for model, result in results.items()}
    for domain, family in sorted({(r['domain'], r['family']) for r in metadata if r['family'] != 'distractor'}):
        selected = [i for i, r in enumerate(metadata) if (r['domain'], r['family'], r['split']) == (domain, family, 'check') and r.get('sign', 0) == 0]
        ids = [metadata[i]['source'] for i in selected]
        difference = []
        for index in selected:
            gains = {model: value[index]['methods']['contiguous']['loss']['1']-value[index]['methods']['contiguous']['loss']['4'] for model, value in by_model.items()}
            difference.append(gains['timesfm']-gains['bolt'])
        output[domain+'/'+family] = adapter.conditional_interval(difference, ids)
    return label_positive_counts(output)


def render(run, out):
    """Create portable results and figures without launching any native inference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metadata = read(run/'rows.json')
    results = {model: evaluate(run, model) for model in MODELS}
    summaries = {model: aggregate(result, metadata) for model, result in results.items()}
    paired = paired_models(results, metadata)
    out.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        portable = json.loads(json.dumps(results[model]))
        portable['execution']['checkpoint'].pop('path', None)
        write(out/f'{model}_measurements.json', portable)
        write(out/f'{model}_summary.json', summaries[model])
    write(out/'paired_models.json', paired)
    for name in ['protocol.json', 'preflight.json', 'mutation_checks.json', 'execution.json']:
        value = read(run/name)
        if name == 'protocol.json':
            value['source_bindings'] = {Path(k).name: v for k, v in value['source_bindings'].items()}
            for checkpoint in value['checkpoints'].values(): checkpoint.pop('path', None)
        write(out/name, value)
    write(out/'audit.json', {model: result['audit'] for model, result in results.items()})
    colors = dict(bolt='#127b85', timesfm='#bf592f')
    cells = ['synthetic/stationary/check', 'synthetic/off_grid/check', 'synthetic/chirp/check',
             'monash_electricity_hourly/real/check', 'm4_hourly/real/check', 'monash_pedestrian_counts/real/check']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for axis, cell in zip(axes.flat, cells):
        for model in MODELS:
            curve = summaries[model]['curves'][cell]
            axis.plot([1, 2, 4, 8], [curve['levels'][str(k)]['loss']['mean'] for k in [1, 2, 4, 8]], 'o-', color=colors[model], label=model)
        axis.axhline(summaries['bolt']['curves'][cell]['classical']['spectral']['mean'], color='gray', linestyle=':', label='Fourier/trend')
        axis.set(title=cell.replace('/check', '').replace('monash_', ''), xlabel='Number of groups', ylabel='Normalized forecast MSE')
        axis.set_xticks([1, 2, 4, 8]); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(out/'decomposition_curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for axis, frequency in zip(axes, [6, 17]):
        for model in MODELS:
            for level, style in [('1', '-'), ('4', '--')]:
                errors = [summaries[model]['responses'][f'synthetic/distractor/{frequency}/{amplitude}/check'][level]['error']['mean'] for amplitude in [.5, 2., 8.]]
                axis.plot([.5, 2., 8.], errors, 'o'+style, color=colors[model], label=f'{model} K{level}')
        axis.set(title=f'Target bin5, distractor bin{frequency}', xlabel='Distractor amplitude (target0.5)', ylabel='Known-component response NMSE', xscale='log')
        axis.set_xticks([.5, 2, 8], labels=['0.5', '2', '8'])
        axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(out/'distractor_response.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for model in MODELS:
        phase = summaries[model]['phase']['phase_matched/check']
        means = [phase[k]['mean'] for k in ['1', '4', '8']]
        axes[0].plot([1, 4, 8], means, 'o-', color=colors[model], label=model)
        grouping = summaries[model]['grouping']['synthetic/stationary/check']
        values = [grouping[k]['mean'] for k in ['interleaved', 'matched_random', 'linear_detrend']]
        axes[1].plot(['Interleaved', 'Matched random', 'Linear detrend'], values, 'o-', color=colors[model], label=model)
    axes[0].set(title='Same spectrum, crest-matched phases', xlabel='Number of groups', ylabel='Matched-phase minus original loss')
    axes[0].set_xticks([1, 4, 8])
    axes[1].set(title='Grouping / detrending sensitivity at K4', ylabel='Contiguous mean-detrend loss minus alternative')
    for axis in axes:
        axis.axhline(0, color='gray', linewidth=.8); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(out/'phase_grouping.png', dpi=160); plt.close(fig)
    response_rows = []
    for domain in ['monash_electricity_hourly', 'm4_hourly', 'monash_pedestrian_counts']:
        fields = []
        for model in MODELS:
            cell = summaries[model]['responses'][domain+'/real/0/0/check']
            fields.append(f"{cell['1']['error']['mean']:.4f} → {cell['4']['error']['mean']:.4f}")
        response_rows.append(f'<tr><td>{domain}</td><td>{fields[0]}</td><td>{fields[1]}</td></tr>')
    rows = []
    for cell in sorted(summaries['bolt']['curves']):
        if not cell.endswith('/check'): continue
        fields = []
        for model in MODELS:
            c = summaries[model]['curves'][cell]
            gain = c['levels']['4']['gain']
            fields.append(f"{gain['mean']:+.4f} [{gain['low']:+.4f}, {gain['high']:+.4f}]")
        rows.append(f"<tr><td>{html.escape(cell[:-6])}</td><td>{fields[0]}</td><td>{fields[1]}</td></tr>")
    candidates = [dict(model=model, **candidate) for model in MODELS for candidate in summaries[model]['candidates']]
    write(out/'candidates.json', candidates)
    links = ' · '.join(f'<a href="{name}">{name}</a>' for name in ['protocol.json', 'audit.json', 'independent_audit.json', 'bolt_summary.json', 'timesfm_summary.json', 'paired_models.json', 'candidates.json'])
    far_timesfm = summaries['timesfm']['responses']['synthetic/distractor/17/8.0/check']
    far_bolt = summaries['bolt']['responses']['synthetic/distractor/17/8.0/check']
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fourier decomposition: small comparative study</title><style>body{{max-width:1160px;margin:35px auto;padding:0 22px;font:17px/1.6 system-ui;color:#20303b}}h1{{line-height:1.2}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc9ce;padding:10px;text-align:left}}th{{background:#edf4f5}}a{{color:#126477}}.notice{{background:#eef5f7;padding:15px}}</style>
<h1>When does splitting a signal help—or hurt?</h1>
<p>A past-only Fourier adapter, two frozen models, 16 synthetic seeds and 24 new cached real sources. Every decomposition preserves the observed signal. The adapter has no access to true generator frequencies or future values.</p>
<p class="notice"><strong>Bounded proof of concept.</strong> Half the sources form a fixed check set. Both beneficial and harmful changes are retained. These real sources have unknown pretraining overlap. Results are exploratory; intervals are unadjusted, descriptive source bootstraps, not population or multiple-testing claims.</p>
<h2>Observed candidate findings</h2><ul><li><strong>Distractor sensitivity differs by model.</strong> At the strongest distant distractor, TimesFM response NMSE changes from {far_timesfm['1']['error']['mean']:.4f} native to {far_timesfm['4']['error']['mean']:.4f} after splitting; Bolt changes from {far_bolt['1']['error']['mean']:.4f} to {far_bolt['4']['error']['mean']:.4f}. Separation exposes a response TimesFM can forecast accurately; it does not make Bolt equally accurate.</li><li><strong>Response fidelity and forecast accuracy can move in opposite directions.</strong> On electricity and pedestrian sources, TimesFM's mean response gain has the same positive sign in discovery and check, while total forecasting deteriorates. M4 response improvement appears only in check, so it is weaker evidence.</li><li><strong>More splitting is not a universal remedy.</strong> Curves are nonmonotone. Both models' check-set total forecasts worsen at K4 in all three real datasets. Off-grid and chirp cases also deteriorate; decomposition quality is part of the explanation to investigate.</li></ul>
<h2>Forecast fidelity across the decomposition tree</h2><img src="decomposition_curves.png" alt="Forecast error versus number of spectral groups for two models and six input regimes">
<p>K1 is native forecasting; K2/K4/K8 sum forecasts of progressively separated spectral groups with duplicate F(0) bias removed. Frequency groups retain all input energy. More groups need not improve fidelity. Linear-detrend sensitivity, raw sums, per-node interactions and first threshold crossings are retained in the records. A threshold crossing is not proof of semantic interpretability.</p>
<table><tr><th>Fixed check condition</th><th>Bolt K4 gain</th><th>TimesFM K4 gain</th></tr>{''.join(rows)}</table>
<p>Positive gain means lower error after decomposition; negative means harm. Brackets show conditional, unadjusted95% source-bootstrap intervals. Synthetic conditions have8 independent check seeds; real datasets have4 check sources each. Different conditions from one seed are dependent.</p>
<h2>A known weak component under a distractor</h2><img src="distractor_response.png" alt="Known-component response error under increasing nearby or distant distractor amplitude">
<p>Target amplitude0.5 and noise0.05 remain fixed while distractor amplitude changes. Scores compare the actual forecast change under paired +/- additions with their known future contribution. This is behavioral evidence; neither native normalization nor internal interference is isolated causally. For the distant on-grid distractor, K4 puts the target and distractor into different groups; identical target-group inputs across amplitudes make the adapter response amplitude-invariant by construction. Its actual response accuracy, and the difference from native behavior, remain measured outcomes. The nearby pair shares one group and is not separated by this tree.</p>
<h2>Response fidelity on real backgrounds</h2><table><tr><th>Fixed check dataset</th><th>Bolt native → K4</th><th>TimesFM native → K4</th></tr>{''.join(response_rows)}</table><p>Known-addition response NMSE, averaging signs within source. Lower is better. These are controlled additions to real signals, not estimates of their unknown latent components. In this pilot, better TimesFM component-response fidelity coexists with worse total forecasting: response consistency is not a proxy for forecast quality. A post-hoc model-free periodic128 continuation control is also saved: it reproduces the on-grid addition by construction and diagnoses the three-peak baseline’s selection failure under the strongest nearby distractor; it is not evidence of strong real-data forecasting.</p>
<h2>Phase and grouping controls</h2><img src="phase_grouping.png" alt="Paired phase and grouping contrasts for synthetic check seeds"><p>The matched-phase comparison preserves Fourier magnitudes and approximately preserves crest factor. Mean effects can cancel heterogeneous seed effects; full paired intervals are retained in the summaries. Grouping controls are input interventions, not internal causal localization. Equal nominal K need not mean equal numbers of nonzero groups on sparse synthetic signals; active group counts and component RMS are retained for every row, so that confound must be checked before interpreting a grouping effect.</p>
<h2>What is retained for follow-up</h2><p>{len(candidates)} prespecified model/condition contrasts exceed0.10 in absolute magnitude with the same sign in discovery and check. These nominate later replication; none triggers additional compute here. All conditions, including those missing that criterion, remain in the report. Grouping, phase and response comparisons are in the complete source records.</p>
<p>The phase controls preserve spectral magnitudes; matched-phase histories also match crest factor within3%. Alternative grouping trees reuse the exact same leaves and root. Their random grouping controls pass the fixed energy-matching gate, but on sparse stationary synthetic inputs they can merely move zero-energy leaves between groups. Effective group-input change is separately recorded; a degenerate random control cannot establish grouping specificity. On the real rows, effective changes must be checked individually. Off-grid frequencies, chirps and coupled components challenge the Fourier representation itself. The history-selected Fourier/trend and seasonal baselines use no oracle frequencies.</p>
<h2>Reproduction and audit</h2><p>{links}</p><p>All node inputs and forecasts are retained in the run. Model-free replay checks hashes, finite forecasts, exact K1 identity, input reconstruction and telescoping of node forecast changes. No new SAE, internal patch, larger grid or additional native model is trained or run. See README for commands and limitations.</p></html>'''
    (out/'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(audit={m: results[m]['audit'] for m in MODELS}, candidates=candidates), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.out)
