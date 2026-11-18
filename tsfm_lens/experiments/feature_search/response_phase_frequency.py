"""Saved-only output-frequency hypothesis, with OLD-source CV and fixed TEST evaluation.

A response map can rotate phase while preserving frequency, or change the output
waveform's frequency. Finite32-step windows confound those explanations with
offset/trend and mixed oscillations. We retain all curves and a matched nuisance
variant; only OLD sources select candidate output periods and fit coefficients.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha

PERIODS = (16., 18., 19.2, 20., 22., 24., 26., 28., 32.)
INPUT_PERIODS = (19.2, 24.)
VARIANTS = ('tone_only', 'tone_dc_trend')
PHASE = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])


def design(period, variant):
    """Exact continuous output basis with phase at forecast origin, optional matched nuisance."""
    t = np.arange(32, dtype=float)
    tone = np.column_stack([np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period)])
    if variant == 'tone_only': return tone
    if variant == 'tone_dc_trend': return np.column_stack([tone, np.ones(32), (t-15.5)/15.5])
    raise ValueError('Unknown nuisance variant')


def normalize_delta(forecasts, backgrounds, amplitudes):
    """Physical responses normalized by each history-only stimulus scale, equal source weight."""
    a = np.asarray(amplitudes, float)
    if np.any(a <= 0): raise ValueError('Positive stimulus scales required')
    y = (np.asarray(forecasts, float)-np.asarray(backgrounds, float)[:, None, :])/a[:, None, None]
    if y.ndim != 3 or y.shape[1:] != (4, 32) or not np.isfinite(y).all():
        raise ValueError('Need finite source x4phase x32 forecast deltas')
    return y


def odd_waveforms(y):
    """32x2 odd response per source; even terms remain in the scored error."""
    return np.stack([(y[:, 0]-y[:, 2])/2, (y[:, 1]-y[:, 3])/2], axis=-1)


def loss_per_source(y, waveform, input_period):
    """Predict all four phases; denominator is INPUT tone energy for every output candidate."""
    prediction = np.einsum('pc,...tc->...pt', PHASE, waveform)
    ideal = PHASE @ design(input_period, 'tone_only').T
    return np.sum((y-prediction)**2, axis=(1, 2))/np.sum(ideal**2)


def fit_map(y, output_period, variant):
    """Fit a shared output-basis coefficient map to mean old-source odd waveform."""
    d = design(output_period, variant)
    coefficients = np.linalg.lstsq(d, odd_waveforms(y).mean(axis=0), rcond=None)[0]
    return coefficients, d @ coefficients


def cv_predictions(y, output_period, variant):
    """Leave out the entire receiver source, including all four phases."""
    return np.stack([fit_map(y[np.arange(len(y)) != i], output_period, variant)[1] for i in range(len(y))])


def fit_curve(y, input_period, variant):
    """Select once by old-source CV; TEST outcomes are neither accepted nor accessed."""
    coefficients = []; waves = []; cv_waves = []; train_losses = []; cv_losses = []
    for period in PERIODS:
        coef, waveform = fit_map(y, period, variant)
        cv = cv_predictions(y, period, variant)
        coefficients.append(coef); waves.append(waveform); cv_waves.append(cv)
        train_losses.append(loss_per_source(y, waveform, input_period))
        cv_losses.append(loss_per_source(y, cv, input_period))
    cv_losses = np.stack(cv_losses)
    selected = int(np.argmin(cv_losses.mean(axis=1)))
    mean_wave = odd_waveforms(y).mean(axis=0)
    cv_mean_wave = np.stack([odd_waveforms(y[np.arange(len(y)) != i]).mean(axis=0) for i in range(len(y))])
    return dict(selected_index=selected, selected_period=PERIODS[selected],
                coefficients=np.stack(coefficients), waveforms=np.stack(waves), cv_waveforms=np.stack(cv_waves),
                train_losses=np.stack(train_losses), cv_losses=cv_losses,
                mean_waveform=mean_wave, cv_mean_waveforms=cv_mean_wave,
                mean_waveform_train_losses=loss_per_source(y, mean_wave, input_period),
                mean_waveform_cv_losses=loss_per_source(y, cv_mean_wave, input_period))


def bootstrap_sources(types, count=16384):
    """Fixed balanced-type source bootstrap; conditional on frozen old predictors at TEST."""
    types = np.asarray(types); rng = np.random.default_rng(4954400)
    return np.concatenate([rng.choice(np.flatnonzero(types == kind), size=(count, np.sum(types == kind)))
                           for kind in sorted(set(types))], axis=1)


def paired_summary(difference, draws):
    """Descriptive paired-source intervals, with family2 adjustment for primary model pair."""
    difference = np.asarray(difference, float); means = difference[draws].mean(axis=1)
    return dict(mean=float(difference.mean()), wins=int(np.sum(difference > 0)),
                source_differences=difference.tolist(), percentile95=np.quantile(means, [.025, .975]).tolist(),
                primary_family2_percentile=np.quantile(means, [.0125, .9875]).tolist())


def geometry(period, variant, input_period):
    """Design conditioning, tone-subspace overlap and nuisance separation on32 samples."""
    d = design(period, variant); b = design(input_period, 'tone_only'); tone = d[:, :2]
    nuisance = design(period, 'tone_dc_trend')[:, 2:]
    residual_tone = tone-nuisance@np.linalg.lstsq(nuisance, tone, rcond=None)[0]
    residual_input = b-nuisance@np.linalg.lstsq(nuisance, b, rcond=None)[0]
    q = np.linalg.qr(tone)[0]; qi = np.linalg.qr(b)[0]
    qr = np.linalg.qr(residual_tone)[0]; qir = np.linalg.qr(residual_input)[0]
    singular = np.linalg.svd(d, compute_uv=False)
    return dict(design_rank=int(np.linalg.matrix_rank(d)), design_singular_values=singular.tolist(),
                condition_number=float(singular[0]/singular[-1]),
                tone_principal_cosines=np.linalg.svd(q.T@qi, compute_uv=False).tolist(),
                nuisance_residualized_principal_cosines=np.linalg.svd(qr.T@qir, compute_uv=False).tolist(),
                residual_tone_singular_values=np.linalg.svd(residual_tone, compute_uv=False).tolist())


def validate(out):
    """Refuse changed code, cohorts or saved source forecasts before any fitting/scoring."""
    p = read(out/'protocol.json')
    for file, digest in p['bindings'].items():
        if sha(file) != digest: raise ValueError(f'Frozen binding changed: {file}')
    return p


def load_cohort(p, cohort, model, period):
    """Extract center-only rows in explicit source order, with source identities attached."""
    cfg = p['cohorts'][cohort]; groups = read(cfg['groups'])
    selected = [g for g in groups if abs(g['frequency']-128/period) < 1e-12 and g.get('offset', 0.) == 0.]
    selected.sort(key=lambda g: g['source_id'])
    if len(selected) != 16 or [g['source_id'] for g in selected] != list(range(16)):
        raise ValueError('Cohort must contain all16 source groups')
    forecasts = np.load(Path(cfg['forecasts'])/f'{model}_native_forecast.npy', mmap_mode='r')
    y = normalize_delta(np.stack([forecasts[g['indices']] for g in selected]),
                        np.stack([forecasts[g['base']] for g in selected]), [g['amplitude'] for g in selected])
    return y, selected


def prepare(out, old, test):
    """Freeze candidates and selection/TEST roles before new frequency-analysis outcomes."""
    if not read(old/'audit.json')['passed'] or not read(test/'audit.json')['passed']:
        raise ValueError('Both saved response cohorts must already be audited')
    fit = Path(read(old/'protocol.json')['prior'])
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'experiments/feature_search/response_phase_frequency_audit.py',
             root/'experiments/feature_search/response_phase_frequency_plants.py', root/'tests/test_response_phase_frequency.py',
             root/'experiments/feature_search/coherence_study.py', root/'experiments/feature_search/real_background_poc.py']
    for run in (fit, test):
        files += [run/n for n in ['sources.json','execution.json','protocol.json','bolt_native_forecast.npy','timesfm_native_forecast.npy',
                                  'bolt_worker.json','timesfm_worker.json']]
        if not all(read(run/'execution.json')[m]['valid'] for m in ('bolt','timesfm')): raise ValueError('Invalid capture')
        for model in ('bolt','timesfm'):
            worker = read(run/f'{model}_worker.json');name=f'{model}_native_forecast.npy'
            if any(worker['gates'].values()) or not worker['batch_replay_exact'] or sha(run/name)!=worker['hashes'][name]:
                raise ValueError('Capture identity/hash gate failed')
    files += [old/'groups.json', old/'audit.json', test/'groups.json', test/'audit.json']
    fit_ids = {r['source'] for r in read(fit/'sources.json')};test_ids = {r['source'] for r in read(test/'sources.json')}
    if fit_ids & test_ids: raise ValueError('FIT/TEST sources overlap')
    protocol = dict(question='Does an old-selected output-frequency shift transfer beyond a same-input-frequency phase map?',
        scope='SAVED ONLY. Native policy,16OLD FIT sources and16fresh-transfer TEST sources. No native inference, no model-weight training.',
        candidate_output_periods=list(PERIODS), input_periods=list(INPUT_PERIODS), variants=list(VARIANTS),
        cohorts=dict(FIT=dict(groups=str((old/'groups.json').resolve()), forecasts=str(fit.resolve())),
                     TEST=dict(groups=str((test/'groups.json').resolve()), forecasts=str(test.resolve()))),
        source_roles=dict(FIT=sorted(fit_ids), TEST=sorted(test_ids)),
        selection='Separately model/input-period/variant, minimize mean leave-one-SOURCE-out OLD phase-pooled waveform NMSE across all9 candidates; exact ties choose first candidate order. Refit selected candidate on all16OLD. TEST never selects periods, variants, interpolators or hyperparameters.',
        baselines='Same-input-frequency shared2x2H; old-CV-selected output-frequency2x2H; old mean32x2 odd response waveform (64 coefficients, higher capacity descriptive). Every variant retains all9 FIT-train, FIT-CV and TEST source loss curves. Tone+DC+lineartrend jointly fits4x2 maps and uses the same nuisance capacity for every candidate and same-input baseline. Mean32x2 already spans nuisance.',
        target='All4 physical incremental forecasts minus background, divided by history-only stimulus amplitude. Equal-source error pooled over4phases, normalized by total ideal INPUT-frequency signal energy; even response remains scored and cannot be fit by these odd linear maps.',
        primary='19.2input, tone-only, TEST same-inputH minus OLD-CV-selected outputH NMSE, separately Bolt/TimesFM. Descriptive paired16384 balanced-type source bootstrap draws seed4954400; nominal95% and family2 Bonferroni97.5% percentile intervals. No automatic threshold/population claim.24h and nuisance variant descriptive.',
        identification='Report full candidate curves, selected/runner-up OLD-CV gap, source-bootstrap minimizer counts from fixed OLD-CV losses (stability only, not nested refitting or posterior), design singular values/condition, tone-subspace cosines before/after nuisance residualization, fittedH and waveform rank. No TEST minimizer is selected or promoted.',
        limitation='Only32 future samples (1to2 cycles), finite discrete candidate grid, four phases, one weak amplitude, conditional mixture types. Single-frequency projection may summarize mixed/nonstationary response. Reused TEST has prior phase-analysis exposure; exploratory saved-data transfer, not new sealed replication. Predicting model response is not improving forecasts; no identified patch causality.',
        stage_order='Prepare protocol -> tests/plants -> FIT-only selection and coefficient artifacts -> immutable FIT freeze -> TEST scoring -> independent replay -> knowledge. No native slot.',
        bindings={str(f.resolve()):sha(f) for f in files})
    out.mkdir(parents=True, exist_ok=False);write(out/'protocol.json',protocol)
    write(out/'knowledge.json',dict(status='prepared_before_frequency_metrics',question=protocol['question'],design=protocol,
        exact_findings=None,controls=protocol['baselines'],limitations=protocol['limitation'],
        evidence_class='Prospective saved-only hypothesis test with OLD-only selection and previously examined TEST cohort',
        failed_instrumentation_runs=[],next_decision='Run FIT-only selection after tests; freeze fits before TEST scoring.',native_capture=False))


def fit(out):
    """OLD responses only; publish immutable fitted coefficients before loading TEST outcomes."""
    p = validate(out)
    if (out/'fit_freeze.json').exists(): raise ValueError('Refuse overwrite fitted selection')
    arrays = {}; rows = []
    for model in ('bolt','timesfm'):
        for period in INPUT_PERIODS:
            y, groups = load_cohort(p, 'FIT', model, period)
            draws = bootstrap_sources([g['background_type'] for g in groups])
            for variant in VARIANTS:
                curve = fit_curve(y, period, variant); key = f'{model}_{period}_{variant}'
                arrays[f'{key}_FIT_delta'] = y
                for field, value in curve.items():
                    if isinstance(value,np.ndarray): arrays[f'{key}_{field}'] = value
                means = curve['cv_losses'].mean(axis=1);order = np.argsort(means,kind='stable')
                winners = np.argmin(curve['cv_losses'][:,draws].mean(axis=2),axis=0)
                h = curve['coefficients'][curve['selected_index']][:2]
                rows.append(dict(model=model,input_period=period,variant=variant,
                    selected_index=curve['selected_index'],selected_period=curve['selected_period'],
                    same_index=PERIODS.index(period),fit_train_curve=curve['train_losses'].mean(axis=1).tolist(),
                    fit_cv_curve=means.tolist(),cv_runner_up_gap=float(means[order[1]]-means[order[0]]),
                    cv_fixed_loss_bootstrap_selection_counts={str(f):int(np.sum(winners==i)) for i,f in enumerate(PERIODS)},
                    geometry=[geometry(f,variant,period) for f in PERIODS],
                    selected_H_singular_values=np.linalg.svd(h,compute_uv=False).tolist(),selected_H_rank=int(np.linalg.matrix_rank(h)),
                    mean_waveform_singular_values=np.linalg.svd(curve['mean_waveform'],compute_uv=False).tolist()))
    np.savez_compressed(out/'fit_arrays.npz',**arrays);write(out/'fit_selection.json',rows)
    freeze=dict(protocol_sha256=sha(out/'protocol.json'),files={f:sha(out/f) for f in ['fit_arrays.npz','fit_selection.json']},
                test_analysis_performed=False,selection_basis='OLD16 only; no TEST outcomes loaded by fit()')
    write(out/'fit_freeze.json',freeze)
    k=read(out/'knowledge.json');k.update(status='fit_frozen_before_test',exact_findings=dict(fit_selection=rows),
        next_decision='Evaluate all frozen candidates on TEST; never reselect from TEST.');write(out/'knowledge.json',k)


def evaluate(out):
    """Evaluate every frozen candidate while keeping OLD-selected period as the comparison."""
    p=validate(out);freeze=read(out/'fit_freeze.json')
    if freeze['protocol_sha256']!=sha(out/'protocol.json'): raise ValueError('Changed protocol')
    for file,digest in freeze['files'].items():
        if sha(out/file)!=digest: raise ValueError('Frozen FIT changed')
    if (out/'results.json').exists(): raise ValueError('Refuse overwrite TEST scoring')
    fitted=np.load(out/'fit_arrays.npz');selections=read(out/'fit_selection.json');arrays={};results=[]
    for row in selections:
        model=row['model'];period=row['input_period'];variant=row['variant'];key=f'{model}_{period}_{variant}'
        y,groups=load_cohort(p,'TEST',model,period);draws=bootstrap_sources([g['background_type'] for g in groups])
        waves=fitted[f'{key}_waveforms'];losses=np.stack([loss_per_source(y,w,period) for w in waves])
        mean_loss=loss_per_source(y,fitted[f'{key}_mean_waveform'],period)
        same=losses[row['same_index']];selected=losses[row['selected_index']]
        arrays.update({f'{key}_TEST_delta':y,f'{key}_test_losses':losses,f'{key}_mean_waveform_test_losses':mean_loss})
        results.append(dict(row,test_curve=losses.mean(axis=1).tolist(),
            same_input_test_nmse=float(same.mean()),selected_output_test_nmse=float(selected.mean()),
            mean_waveform_test_nmse=float(mean_loss.mean()),
            same_minus_selected=paired_summary(same-selected,draws),selected_minus_mean_waveform=paired_summary(selected-mean_loss,draws),
            by_type=[dict(background_type=k,sources=4,mean_same_minus_selected=float((same-selected)[[g['background_type']==k for g in groups]].mean())) for k in range(4)]))
    np.savez_compressed(out/'test_arrays.npz',**arrays);write(out/'results.json',results)
    k=read(out/'knowledge.json');k.update(status='computed_pending_independent_audit',exact_findings=results,
        next_decision='Independent projection/CV/source-split/selection/loss/bootstrap replay; assess ambiguity and heldout gain, then close saved-only study.')
    write(out/'knowledge.json',k)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','fit','evaluate'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--old',type=Path,default=Path('runs/response_phase4954200'))
    parser.add_argument('--test',type=Path,default=Path('runs/response_phase_transfer4954300'));args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.old,args.test)
    else:
        try:(fit if args.command=='fit' else evaluate)(args.out)
        except Exception as error:
            if (args.out/'knowledge.json').exists():
                k=read(args.out/'knowledge.json');k['status']='failed_instrumentation';k['failed_instrumentation_runs'].append(dict(stage=args.command,error=repr(error)));write(args.out/'knowledge.json',k)
            raise
