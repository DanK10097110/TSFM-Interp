"""Reuse-only four-phase response matrices, frozen before scoring saved forecasts.

The source/background is held fixed while the injected phase changes. These are
finite-context conditional response maps, not transfer functions of an LTI model.
Only amplitude.1 center rows from focused4954100 enter this experiment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha

CENTERS=(128/24,128/19.2)
PHASE_COEFFICIENTS=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
J=np.array([[0.,-1.],[1.,0.]])


def basis(frequency):
    """Physical sine/cosine forecast basis, indexed from the forecast origin."""
    t=np.arange(32)
    return np.column_stack([np.sin(2*np.pi*frequency*t/128),np.cos(2*np.pi*frequency*t/128)])


def split_matrix(h):
    """Frobenius-orthogonal complex-linear and symmetric-traceless decomposition."""
    h=np.asarray(h,float)
    if h.shape!=(2,2) or not np.isfinite(h).all():raise ValueError('Need finite2x2 matrix')
    q=float(np.trace(h)/2);r=float((h[1,0]-h[0,1])/2)
    linear=q*np.eye(2)+r*J;anisotropic=h-linear
    singular=np.linalg.svd(h,compute_uv=False);energy=float(np.sum(h*h))
    gain=float(np.hypot(q,r));beta=float(np.hypot(anisotropic[0,0],anisotropic[0,1]))
    return dict(q=q,r=r,complex_gain=gain,phase_radians=float(np.arctan2(r,q)) if gain>1e-12 else None,
        complex_linear=linear.tolist(),symmetric_traceless=anisotropic.tolist(),
        conjugate_gain=beta,anisotropy_frobenius_fraction=float(np.sum(anisotropic**2)/energy) if energy>0 else None,
        singular_values=singular.tolist(),singular_ratio=float(singular[1]/singular[0]) if singular[0]>0 else None,
        numerical_rank=int(np.sum(singular>1e-10*max(1.,singular[0]))),
        relative_rank_at_10pct=int(np.sum(singular>=.1*singular[0])) if singular[0]>1e-8 else 0,
        participation_rank=float(singular.sum()**2/np.sum(singular**2)) if np.sum(singular**2)>0 else 0.,
        determinant=float(np.linalg.det(h)))


def response_matrix(forecasts,background,amplitude,frequency):
    """Pair opposite phases to separate odd linear-response axes from even response.

Primary H projects onto the uncentered finite-horizon sine/cosine basis; its
residual is orthogonal and its source fit minimizes in-band waveform error.
A separate joint-DC fit exposes finite-window constant-offset leakage.
"""
    y=np.asarray(forecasts,float);base=np.asarray(background,float)
    if y.shape!=(4,32) or base.shape!=(32,) or amplitude<=0:raise ValueError('Bad phase geometry')
    delta=y-base
    odd=np.column_stack([(y[0]-y[2])/2,(y[1]-y[3])/2])
    even=np.column_stack([y[0]+y[2]-2*base,y[1]+y[3]-2*base])
    waveform=odd/amplitude;b=basis(frequency)
    h=np.linalg.lstsq(b,waveform,rcond=None)[0]
    residual=waveform-b@h
    dc_design=np.column_stack([b,np.ones(32)])
    dc_fit=np.linalg.lstsq(dc_design,waveform,rcond=None)[0]
    ideal=amplitude*PHASE_COEFFICIENTS@b.T
    response_energy=float(np.sum(waveform**2));ideal_energy=float(np.sum(b**2))
    metrics=dict(H=h.tolist(),**split_matrix(h),H_dc_adjusted=dc_fit[:2].tolist(),
        dc_coefficients=dc_fit[2].tolist(),dc_matrix_shift_frobenius=float(np.linalg.norm(h-dc_fit[:2])),
        off_frequency_odd_energy_fraction=float(np.sum(residual**2)/response_energy) if response_energy>0 else None,
        off_frequency_nmse_contribution=float(np.sum(residual**2)/ideal_energy),
        even_nmse_contribution=float(np.sum(even**2)/(4*amplitude**2*ideal_energy)),
        even_raw_rms=float(np.sqrt(np.mean(even**2))),
        odd_response_energy_over_ideal=response_energy/ideal_energy,
        even_response_energy_fraction=float(np.sum(even**2)/2/np.sum(delta**2)) if np.sum(delta**2)>0 else None)
    return metrics,dict(delta=delta,odd=odd,even=even,waveform=waveform,H=h,
                        residual=residual,ideal=ideal,H_dc_adjusted=dc_fit[:2],dc=dc_fit[2])


def loso_mean(matrices):
    """Hold out the entire source, including all its phases; equal-weight the remaining sources."""
    matrices=np.asarray(matrices,float)
    if len(matrices)<2:raise ValueError('Need at least two independent sources')
    return np.stack([matrices[np.arange(len(matrices))!=i].mean(0) for i in range(len(matrices))])


def prediction(h,amplitude,frequency):
    """Predict incremental response to the known stimulus, not a natural-data forecast."""
    return amplitude*PHASE_COEFFICIENTS@np.asarray(h,float).T@basis(frequency).T


def waveform_loss(pred,actual,ideal):
    """Pool phase energies before division; do not average unequal per-phase NMSE ratios."""
    return float(np.sum((np.asarray(pred,float)-actual)**2)/np.sum(ideal**2))


def bootstrap_indices(source_types,nboot=4096):
    """Balanced-type source bootstrap; the same draws are reused for every endpoint."""
    source_types=np.asarray(source_types);rng=np.random.default_rng(4954200)
    return np.concatenate([rng.choice(np.flatnonzero(source_types==kind),size=(nboot,np.sum(source_types==kind)),replace=True)
                           for kind in sorted(set(source_types))],axis=1)


def interval(values,draws):
    """Exploratory percentile interval of a source mean; no independent-phase resampling."""
    values=np.asarray(values,float);means=values[draws].mean(1)
    return dict(mean=float(values.mean()),median=float(np.median(values)),min=float(values.min()),max=float(values.max()),
                percentile95=[float(x) for x in np.quantile(means,[.025,.975])])


def prepare(out,prior):
    """Bind rows, arrays, formulas, baselines and source bootstrap before computing any matrix."""
    from experiments.feature_search import response_phase_audit
    sources=read(prior/'sources.json');signals=read(prior/'signals.json');execution=read(prior/'execution.json')
    if len(sources)!=16 or len(signals)!=128:raise ValueError('Expected focused16-source crossed-phase cohort')
    for model in ('bolt','timesfm'):
        if not execution[model]['valid']:raise ValueError('Invalid native run')
        worker=read(prior/f'{model}_worker.json')
        if any(worker['gates'].values()) or not worker['batch_replay_exact']:raise ValueError('Native gates failed')
        for policy in ('native','locked'):
            file=f'{model}_{policy}_forecast.npy'
            if sha(prior/file)!=worker['hashes'][file]:raise ValueError('Forecast binding changed')
    groups=[]
    for source_id,source in enumerate(sources):
        for frequency in CENTERS:
            rows=sorted([r for r in signals if r['source_id']==source_id and r['frequency']==frequency],key=lambda r:r['phase_id'])
            if len(rows)!=4 or [r['phase_id'] for r in rows]!=[0,1,2,3] or any(r['amplitude']!=.1 for r in rows):raise ValueError('Missing/falsely included phase cell')
            if len(set(r['base'] for r in rows))!=1 or len(set(r['input_scale'] for r in rows))!=1:raise ValueError('Phase pairs must share background and scale')
            groups.append(dict(source=source['source'],source_id=source_id,background_type=source['background_type'],frequency=frequency,
                period=128/frequency,base=rows[0]['base'],indices=[r['zero'] for r in rows],amplitude=.1*rows[0]['input_scale'],input_scale=rows[0]['input_scale']))
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),Path(response_phase_audit.__file__),root/'tests/test_response_phase_study.py',
           root/'experiments/feature_search/response_phase_plants.py',root/'experiments/feature_search/coherence_study.py']
    names=['protocol.json','sources.json','signals.json','inputs.npy','execution.json']
    names += [f'{m}_{policy}_forecast.npy' for m in ('bolt','timesfm') for policy in ('native','locked')]
    names += [f'{m}_worker.json' for m in ('bolt','timesfm')]
    out.mkdir(parents=True,exist_ok=False);write(out/'groups.json',groups)
    protocol=dict(question='Can conditional phase-response matrices explain smooth but phase-sensitive known-tone responses, and does a source-global mean predict held-out sources better than identity?',
        prior=str(prior.resolve()),source_count=16,groups=32,amplitude_peak_background_sd=.1,centers=list(CENTERS),horizon=32,
        scope='REUSE ONLY: center signal rows and base forecasts. No hidden states, no extra frequencies/amplitudes, no native inference.',
        coordinate_convention='Input/output coordinates [sin(omega*t),cos(omega*t)], phase at forecast origin. Input coefficients A[cos(phi),sin(phi)]. H columns are opposite-phase central differences projected onto forecast basis and divided by A.',
        primary_projection='Uncentered orthogonal least-squares projection onto exact continuous sine/cosine basis,32 future steps. Also fit a DC nuisance jointly and report H_dc_adjusted, DC coefficients and matrix shift; finite-window leakage is not ignored.',
        decomposition='J=[[0,-1],[1,0]],q=trace(H)/2,r=(H21-H12)/2; H=(qI+rJ)+S, S symmetric traceless. Frobenius energy split only, not a physical waveform-energy attribution. S includes phase conjugation; nonzero S need not imply an eccentric ellipse.',
        rank='Both singular values, axis ratio, participation rank, numerical rank at1e-10*max(1,smax), relative rank at10% of smax with absolute dead floor1e-8. Threshold flags do not prove absent information.',
        residuals='Odd waveform M=[(f0-fpi)/2,(fpi2-f3pi2)/2]/A. Off-frequency residual R=M-BH is orthogonal to B. E=[f0+fpi-2fbase,fpi2+f3pi2-2fbase]. Save raw E, odd M, projection and residual.',
        input_gate='Verify actual FP32 center-minus-background histories against nominal four-phase waveforms; RMS error/peak A must be<=1e-5. Record antipodal input closure; no phase or source is replaced.',
        energy_closure='For every matrix predictor K, phase-energy-pooled NMSE = ||B(H-K)||F^2/||B||F^2 + ||R||F^2/||B||F^2 + ||E||F^2/(4*A^2*||B||F^2). Exact finite-horizon algebra, no stationarity assumption.',
        baselines=['fixed H=I','source-global LOSO arithmetic mean H from other15 sources, separately model/policy/frequency','per-source H, fitted on same4 phases: optimistic descriptive in-band ceiling, NOT held-out prediction'],
        score='Sum squared waveform errors over all4 phases divided by total ideal signal energy. Equal-source aggregate. Different from earlier unweighted mean of per-phase NMSE because finite-window phase energies differ.',
        resampling='4096 source bootstrap draws, seed4954200, stratified4 background types with4 sources each. Reuse the same source draws for all models/policies/frequencies/endpoints; phases are never resampling units.',
        inference='Descriptive reused16-source exploratory comparison; percentile95 intervals on source means and paired gains, no corrected confirmation or p-values. All source rows and gain signs retained.',
        primary_comparison='Identity-minus-LOSO waveform NMSE by model/policy/frequency. Per-source H is an explanatory ceiling and cannot establish new-source predictive performance.',
        limitations='Finite horizon and a fixed background break phase/time-shift symmetry. No LTI, global smoothness, cross-model coordinate equivalence, training-causality or absent-information claim. Only4 phases at one amplitude: even response can expose nonlinearity, but full phase orbit is unobserved.',
        source_hashes={str(f.resolve()):sha(f) for f in files},prior_hashes={name:sha(prior/name) for name in names},
        hashes={'groups.json':sha(out/'groups.json')})
    write(out/'protocol.json',protocol)
    write(out/'knowledge.json',dict(status='prepared_before_metrics',question=protocol['question'],design=protocol,exact_findings=None,
        controls=protocol['baselines'],limitations=protocol['limitations'],evidence_class='Prospective reuse-only descriptive conditional response analysis',
        failed_instrumentation_runs=[],next_decision='Validate fixtures/plants, then compute frozen saved-array metrics. No native process.'))


def validate(out):
    """Check frozen source, selected rows, and all original forecast bindings."""
    p=read(out/'protocol.json');prior=Path(p['prior'])
    for name,digest in p['source_hashes'].items():
        if sha(name)!=digest:raise ValueError(f'Frozen source changed: {name}')
    for name,digest in p['prior_hashes'].items():
        if sha(prior/name)!=digest:raise ValueError(f'Prior artifact changed: {name}')
    for name,digest in p['hashes'].items():
        if sha(out/name)!=digest:raise ValueError(f'Prepared rows changed: {name}')
    return p


def run(out):
    """Fit all source matrices, score frozen baselines, and persist knowledge immediately."""
    started=time.monotonic();p=validate(out);prior=Path(p['prior']);groups=read(out/'groups.json')
    if (out/'results.json').exists():raise ValueError('Refuse overwrite scored experiment')
    source_types=[r['background_type'] for r in read(prior/'sources.json')];draws=bootstrap_indices(source_types)
    inputs=np.load(prior/'inputs.npy',mmap_mode='r')
    np.save(out/'bootstrap_source_indices.npy',draws)
    metrics=[];summary=[];arrays={}
    for model in ('bolt','timesfm'):
        for policy in ('native','locked'):
            forecast=np.load(prior/f'{model}_{policy}_forecast.npy',mmap_mode='r')
            for frequency in CENTERS:
                selected=[g for g in groups if g['frequency']==frequency];records=[];raws=[]
                for group in selected:
                    record,raw=response_matrix(forecast[group['indices']],forecast[group['base']],group['amplitude'],frequency)
                    observed=inputs[group['indices']].astype(float)-inputs[group['base']].astype(float)
                    t=np.arange(-128,0);past_basis=np.column_stack([np.sin(2*np.pi*frequency*t/128),np.cos(2*np.pi*frequency*t/128)])
                    nominal=group['amplitude']*PHASE_COEFFICIENTS@past_basis.T
                    input_error=float(np.sqrt(np.mean((observed-nominal)**2))/group['amplitude'])
                    if input_error>1e-5:raise ValueError('Input waveform binding failed')
                    record['input_nominal_rms_over_peak']=input_error
                    record['input_antipodal_rms_over_peak']=float(np.sqrt(np.mean(np.stack([observed[0]+observed[2],observed[1]+observed[3]])**2))/group['amplitude'])
                    records.append(dict(group,model=model,policy=policy,**record));raws.append(raw)
                hs=np.asarray([r['H'] for r in raws]);held=loso_mean(hs)
                for i,(record,raw) in enumerate(zip(records,raws)):
                    predictions={name:prediction(h,record['amplitude'],frequency) for name,h in
                                 [('identity',np.eye(2)),('loso',held[i]),('source_fit',hs[i])]}
                    losses={name:waveform_loss(pred,raw['delta'],raw['ideal']) for name,pred in predictions.items()}
                    record.update(losses=losses,loso_H=held[i].tolist(),
                                  identity_minus_loso=losses['identity']-losses['loso'],identity_minus_source_fit=losses['identity']-losses['source_fit'])
                    metrics.append(record)
                    key=f'{model}_{policy}_{i}_{list(CENTERS).index(frequency)}'
                    arrays.update({f'{key}_{name}':value for name,value in raw.items()})
                    arrays.update({f'{key}_prediction_{name}':value for name,value in predictions.items()})
                    arrays[f'{key}_loso_H']=held[i]
                item=dict(model=model,policy=policy,frequency=frequency,period=128/frequency,source_count=16,
                    baselines={name:interval([r['losses'][name] for r in records],draws) for name in ('identity','loso','source_fit')},
                    paired_identity_minus_loso=interval([r['identity_minus_loso'] for r in records],draws),
                    loso_win_sources=sum(r['identity_minus_loso']>0 for r in records),
                    paired_identity_minus_source_fit=interval([r['identity_minus_source_fit'] for r in records],draws))
                for field in ('q','r','complex_gain','conjugate_gain','anisotropy_frobenius_fraction','singular_ratio',
                              'participation_rank','off_frequency_odd_energy_fraction','off_frequency_nmse_contribution','even_nmse_contribution','dc_matrix_shift_frobenius'):
                    valid=[r[field] for r in records if r[field] is not None]
                    item[field]=interval(valid,draws) if len(valid)==16 else dict(values=valid,undefined_sources=16-len(valid))
                item['relative_rank_counts']={str(rank):sum(r['relative_rank_at_10pct']==rank for r in records) for rank in (0,1,2)}
                item['negative_determinant_sources']=sum(r['determinant']<0 for r in records)
                summary.append(item)
    np.savez_compressed(out/'raw_arithmetic.npz',**arrays)
    write(out/'matrix_rows.json',metrics);write(out/'results.json',dict(summary=summary,seconds=time.monotonic()-started))
    write(out/'knowledge.json',dict(status='computed_pending_independent_audit',question=p['question'],design=p,exact_findings=summary,
        controls=dict(baselines=p['baselines'],dc_sensitivity=True,raw_arithmetic_saved=True,source_grouped_resampling=True),
        limitations=p['limitations'],evidence_class='Descriptive conditional behavioral response geometry; LOSO response prediction is distinct from same-source explanatory fit',
        failed_instrumentation_runs=[],next_decision='Run independent projection, decomposition, energy-closure and LOSO/bootstrap audit; then distill conditional findings.',native_capture_performed=False))
    print(dict(run=str(out.resolve()),matrices=len(metrics),sources=16,seconds=time.monotonic()-started,native_capture=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prior',type=Path,default=Path('runs/response_surface_focused4954100'))
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.prior)
    else:
        try:run(args.out)
        except Exception as error:
            k=read(args.out/'knowledge.json');k['status']='failed_instrumentation';k['failed_instrumentation_runs'].append(dict(stage='metrics',error=repr(error)));k['next_decision']='Preserve failure; repair only new response_phase files in a fresh protocol/run.';write(args.out/'knowledge.json',k);raise
