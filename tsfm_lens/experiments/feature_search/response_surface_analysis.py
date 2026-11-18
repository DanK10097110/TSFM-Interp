"""Recompute the pilot from immutable raw predictions and full-token hidden states.

Reports source-level descriptive surfaces, scale-normalized local differences,
four-point interactions, and cheap grouped Fourier-coefficient decodability.
No selection or population significance is inferred from these eight backgrounds.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha
from experiments.feature_search.response_surface_metrics import (
    rms, difference, ratio, known_response, local_response, interaction, summarize)


def decoder_cv(features, labels, groups, alpha=.01):
    """Leave one whole source out, train-only global scaling, fixed dual ridge.

The scalar feature scale preserves hidden geometry and is learned independently
inside each fold. A training-mean predictor and shuffled-training-label control
share exactly the same held-out rows. No hyperparameter search uses these rows.
"""
    features, labels, groups = np.asarray(features, float), np.asarray(labels, float), np.asarray(groups)
    predictions = np.zeros_like(labels); mean_control = np.zeros_like(labels); shuffled = np.zeros_like(labels)
    folds = []
    for group in sorted(set(groups)):
        test = groups == group; train = ~test
        x, xt = features[train], features[test]
        mean = x.mean(0); x = x-mean; xt = xt-mean
        scale = rms(x)
        if scale == 0: scale = 1.
        x, xt = x/scale, xt/scale
        gram, cross = x@x.T/x.shape[1], xt@x.T/x.shape[1]
        y = labels[train]; ym = y.mean(0)
        matrix = gram+alpha*np.eye(len(x))
        predictions[test] = cross@np.linalg.solve(matrix, y-ym)+ym
        mean_control[test] = ym
        shuffled[test] = cross@np.linalg.solve(matrix, y[np.random.default_rng(4951900+int(group)).permutation(len(y))]-ym)+ym
        folds.append(dict(source_id=int(group), train_sources=sorted(set(int(g) for g in groups[train])),
                          test_rows=int(test.sum()), train_rows=int(train.sum()), feature_scale=scale))
    denominator = float(np.mean((labels-mean_control)**2))
    return dict(nmse=ratio(float(np.mean((predictions-labels)**2)), denominator),
                shuffled_nmse=ratio(float(np.mean((shuffled-labels)**2)), denominator),
                mean_control_mse=denominator, folds=folds), predictions, shuffled, mean_control


def analyze(out):
    """Verify bindings before analyzing and save exact arithmetic and grouped summaries."""
    from experiments.feature_search.response_surface_study import validate, CENTERS, tone
    p = validate(out); execution = read(out/'execution.json')
    inputs = np.load(out/'inputs.npy', mmap_mode='r'); rows = read(out/'rows.json')
    signals, paths, pairs = [read(out/f'{name}.json') for name in ('signals', 'paths', 'interactions')]
    response_rows, local_rows, pair_rows, adjacent_rows, decoders = [], [], [], [], []
    summary = dict(evidence='Descriptive instrumentation, hidden geometry and behavioral response separately',
                   limitations=p['limitations'], inference=p['inference'], models={})
    center_signals = [s for s in signals if s['offset'] == 0]
    decode_rows = []
    for s in center_signals:
        coefficient = np.zeros(4); k = list(CENTERS).index(s['center'])*2
        coefficient[k:k+2] = s['amplitude']*np.array([np.cos(s['phase']), np.sin(s['phase'])])
        decode_rows.append(dict(index=s['index'], base=s['base'], source_id=rows[s['index']]['source_id'],
                                coefficient=coefficient.tolist(), scale=s['scale']))
    for pair in pairs:
        phase = rows[pair['base']]['phase']
        coefficient = np.outer([pair['amplitude_a'], pair['amplitude_b']], [np.cos(phase), np.sin(phase)]).ravel()
        decode_rows.append(dict(index=pair['ab'], base=pair['base'], source_id=rows[pair['base']]['source_id'],
                                coefficient=coefficient.tolist(), scale=pair['scale']))
    labels = np.asarray([r['coefficient'] for r in decode_rows]); groups = [r['source_id'] for r in decode_rows]
    input_features = np.asarray([difference(inputs[r['index']], inputs[r['base']])/r['scale'] for r in decode_rows])
    baseline, pred, shuffled, mean = decoder_cv(input_features, labels, groups)
    decoders.append(dict(model='input', policy='physical', cut='input', **baseline))
    np.savez_compressed(out/'decoder_input.npz', labels=labels, prediction=pred, shuffled=shuffled, mean=mean,
                        source_ids=groups)
    write(out/'decoder_rows.json', decode_rows)
    for model in ('bolt', 'timesfm'):
        if not execution.get(model, {}).get('valid'): raise ValueError(f'Invalid or missing worker: {model}')
        worker = read(out/f'{model}_worker.json')
        for file, digest in worker['hashes'].items():
            if sha(out/file) != digest: raise ValueError(f'Changed native artifact {file}')
        summary['models'][model] = dict(execution=execution[model], worker=worker)
        for policy in ('native', 'locked'):
            for cut in ['forecast']+worker['cuts']:
                file = f'{model}_{policy}_forecast.npy' if cut == 'forecast' else f'{model}_{policy}_hidden_{cut}.npy'
                values = np.load(out/file, mmap_mode='r')
                for s in signals:
                    if cut != 'forecast': continue
                    effect = difference(values[s['index']], values[s['base']])
                    truth = tone(s['frequency'], s['amplitude']*s['scale'], s['phase'], np.arange(32))
                    response_rows.append(dict(model=model, policy=policy, **s, **known_response(effect, truth)))
                for path in paths:
                    result = local_response(values[path['minus']], values[path['zero']], values[path['plus']],
                        values[path['midpoint']], inputs[path['minus']], inputs[path['zero']], inputs[path['plus']],
                        path['step'], path['scale'])
                    result['chord_input_rounding'] = rms(difference(inputs[path['plus']], -inputs[path['minus']])-2*inputs[path['midpoint']].astype(float))/path['scale']
                    local_rows.append(dict(model=model, policy=policy, cut=cut, **path, **result))
                for pair in pairs:
                    result = interaction(values[pair['base']], values[pair['a']], values[pair['b']], values[pair['ab']],
                        difference(inputs[pair['a']], inputs[pair['base']]),
                        difference(inputs[pair['b']], inputs[pair['base']]), pair['scale'])
                    pair_rows.append(dict(model=model, policy=policy, cut=cut, **pair, **result))
                grouped = {}
                for s in signals: grouped.setdefault((s['source'], s['center'], s['amplitude']), []).append(s)
                for group in grouped.values():
                    ordered = sorted(group, key=lambda s: s['frequency'])
                    for a, b in zip(ordered[:-1], ordered[1:]):
                        dx = rms(difference(inputs[b['index']], inputs[a['index']]))/a['scale']
                        dy = rms(difference(values[b['index']], values[a['index']]))
                        adjacent_rows.append(dict(model=model, policy=policy, cut=cut, source=a['source'], domain=a['domain'],
                            center=a['center'], amplitude=a['amplitude'], left=a['index'], right=b['index'],
                            input_distance=dx, output_distance=dy, distance_sensitivity=ratio(dy, dx)))
                if cut != 'forecast':
                    features = np.asarray([difference(values[r['index']], values[r['base']]).ravel() for r in decode_rows])
                    decoded, pred, shuffled, mean = decoder_cv(features, labels, groups)
                    decoders.append(dict(model=model, policy=policy, cut=cut, **decoded))
                    np.savez_compressed(out/f'decoder_{model}_{policy}_{cut}.npz', labels=labels, prediction=pred,
                                        shuffled=shuffled, mean=mean, source_ids=groups)
                else:
                    np.savez_compressed(out/f'{model}_{policy}_arithmetic.npz',
                        response=np.asarray([difference(values[s['index']], values[s['base']]) for s in signals]),
                        known_signal=np.asarray([tone(s['frequency'], s['amplitude']*s['scale'], s['phase'], np.arange(32)) for s in signals]),
                        local_first=np.asarray([difference(values[r['plus']], values[r['minus']])/(2*r['step']) for r in paths]),
                        local_path_second=np.asarray([values[r['plus']].astype(float)+values[r['minus']]-2*values[r['zero']] for r in paths]),
                        local_chord_second=np.asarray([values[r['plus']].astype(float)+values[r['minus']]-2*values[r['midpoint']] for r in paths]),
                        interaction=np.asarray([values[r['ab']].astype(float)-values[r['a']]-values[r['b']]+values[r['base']] for r in pairs]))
    for name, data in [('response_rows', response_rows), ('local_rows', local_rows), ('interaction_rows', pair_rows),
                        ('adjacent_rows', adjacent_rows), ('decodability', decoders)]: write(out/f'{name}.json', data)
    summary['center_response'] = summarize([r for r in response_rows if r['offset'] == 0],
        ['model', 'policy', 'domain', 'center', 'amplitude'], ['nmse', 'gain'])
    summary['local'] = summarize(local_rows, ['model', 'policy', 'cut', 'domain', 'axis', 'relative_step'],
        ['distance_sensitivity', 'chord_relative', 'chord_curvature_per_distance2', 'input_radius', 'chord_input_rounding'])
    summary['interaction'] = summarize(pair_rows, ['model', 'policy', 'cut', 'domain', 'amplitude_a', 'amplitude_b'],
        ['relative_sum', 'relative_a', 'relative_b', 'per_input_product'])
    summary['decodability'] = decoders
    write(out/'analysis.json', summary)
    plot(out, response_rows, local_rows)
    report(out, summary)
    outputs = ['analysis.json', 'response_rows.json', 'local_rows.json', 'interaction_rows.json', 'adjacent_rows.json',
               'decodability.json', 'decoder_rows.json', 'report.html', 'response_surface.png', 'response_curvature.png']
    outputs += [p.name for p in out.glob('*_arithmetic.npz')]+[p.name for p in out.glob('decoder_*.npz')]
    write(out/'analysis_manifest.json', dict(protocol_sha256=sha(out/'protocol.json'), hashes={f: sha(out/f) for f in outputs}))
    print(dict(run=str(out.resolve()), response_rows=len(response_rows), local_rows=len(local_rows),
               interaction_rows=len(pair_rows), decoder_fits=len(decoders)), flush=True)


def plot(out, responses, locals_):
    """Standalone scientific plots retain all eight source trajectories rather than only averages."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for mi, model in enumerate(('bolt', 'timesfm')):
        for ai, amplitude in enumerate((.1, .5, 2.)):
            ax = axes[mi, ai]
            selected = [r for r in responses if r['model'] == model and r['amplitude'] == amplitude]
            sources = sorted(set(r['source'] for r in selected))
            for si, source in enumerate(sources):
                for policy in ('native', 'locked'):
                    for center in sorted(set(r['center'] for r in selected)):
                        points = sorted([r for r in selected if r['source'] == source and r['policy'] == policy and r['center'] == center], key=lambda r:r['frequency'])
                        ax.plot([128/r['frequency'] for r in points], [r['nmse'] for r in points],
                                color=f'C{si}', linestyle='-' if policy == 'native' else '--', alpha=.7,
                                label=source if policy == 'native' and center == min(r['center'] for r in selected) else None)
            ax.axhline(1, color='black', linewidth=.8); ax.set_yscale('symlog', linthresh=.1)
            ax.set_title(f'{model}: peak amplitude {amplitude} background SD'); ax.set_xlabel('Period (hours)')
            ax.set_ylabel('Known-signal incremental forecast NMSE')
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('Fixed source phase; solid native, dashed background-locked. Eight instrumentation sources.')
    fig.tight_layout(); fig.savefig(out/'response_surface.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for mi, model in enumerate(('bolt', 'timesfm')):
        for ai, axis in enumerate(('frequency', 'amplitude')):
            ax = axes[mi, ai]
            for pi, policy in enumerate(('native', 'locked')):
                for cut in sorted(set(r['cut'] for r in locals_ if r['model'] == model)):
                    subset = [r for r in locals_ if r['model'] == model and r['axis'] == axis and r['policy'] == policy and r['cut'] == cut]
                    steps = sorted(set(r['relative_step'] for r in subset))
                    medians = [np.median([r['chord_relative'] for r in subset if r['relative_step'] == step]) for step in steps]
                    ax.plot(steps, medians, marker='o', linestyle='-' if policy == 'native' else '--', label=f'{cut} {policy}')
            ax.set_title(f'{model}: {axis}'); ax.set_xlabel('Cycles per128' if axis == 'frequency' else 'Fractional amplitude step')
            ax.set_ylabel('Median chord curvature / sum of marginal effects'); ax.set_xscale('log'); ax.set_yscale('log')
            ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'response_curvature.png', dpi=160); plt.close(fig)


def report(out, summary):
    """Portable HTML index states estimands and links exact raw arithmetic and summaries."""
    import html
    content = '<!doctype html><meta charset="utf-8"><title>Response surface instrumentation pilot</title>'
    content += '<style>body{max-width:1200px;margin:30px auto;font:16px system-ui}img{width:100%}td,th{padding:6px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>'
    content += '<h1>Response surface instrumentation pilot</h1><p>Eight fixed backgrounds: four synthetic mixtures and four explicitly reused real FIT sources. Descriptive evidence only. Hidden-state geometry and forecast semantics are distinct endpoints.</p>'
    content += '<p>'+html.escape(summary['limitations'])+'</p>'
    content += '<p>All input paths, raw FP32 forecasts and full-token states are saved. FP64 local derivatives use measured input distances. Chord-midpoint controls separate frequency-path bending from model curvature. Four-point interactions are divided by the sum of the two individual effect RMS values; individual denominators remain available.</p>'
    content += '<p><a href="protocol.json">Frozen protocol</a> · <a href="analysis.json">Exact source-grouped analysis</a> · <a href="response_rows.json">Known-signal responses</a> · <a href="local_rows.json">Local differences</a> · <a href="interaction_rows.json">Interactions</a> · <a href="decodability.json">Grouped Fourier decodability</a></p>'
    content += '<img src="response_surface.png" alt="Frequency surfaces by model and amplitude"><p>Each color is a source; solid native and dashed locked. NMSE1 is the zero-response reference. These are controlled incremental responses, not natural-future prediction errors.</p>'
    content += '<img src="response_curvature.png" alt="Chord curvature at shrinking radii"><p>Curvature relative to marginal effects normally shrinks for differentiable maps. These medians do not establish global smoothness or exclude localized irregularities; every source and input distance is retained in the linked records.</p>'
    content += '<h2>Fourier-coefficient decoding</h2><p>Leave one source out, with all of its dependent rows excluded from training. Fixed ridge, train-only scaling. NMSE uses held-out error of the training-mean predictor as denominator. Input is the accessible linear baseline; decodability alone is not causal interpretation.</p><table><tr><th>Model</th><th>Policy</th><th>Cut</th><th>NMSE</th><th>Shuffled-label NMSE</th></tr>'
    for row in summary['decodability']:
        content += '<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in ['model', 'policy', 'cut', 'nmse', 'shuffled_nmse'])+'</tr>'
    content += '</table><p>Pilot complete; expansion requires orchestration. Smooth context dependence, normalization effects, amplitude priors and mixture interference remain competing, potentially simultaneous explanations. Failures do not establish that the model is uninterpretable.</p>'
    (out/'report.html').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', required=True, type=Path)
    analyze(parser.parse_args().out)
