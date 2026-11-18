"""OLD-only output-frequency fits for Bolt direct32-prefix versus64, weak versus strong."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_phase_horizon import OUTPUT_PERIODS,INPUT_PERIODS,AMPLITUDES,validate

PHASE=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
VARIANTS=('tone_only','tone_dc_trend')


def design(period,horizon,variant):
    """Absolute future timestamps; nuisance trend scales to the declared window."""
    t=np.arange(horizon,dtype=float);tone=np.column_stack([np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period)])
    return tone if variant=='tone_only' else np.column_stack([tone,np.ones(horizon),np.linspace(-1,1,horizon)])


def odd(y):
    """Mean odd phase axes; even response stays in the full phase-pooled loss."""
    return np.stack([(y[:,0]-y[:,2])/2,(y[:,1]-y[:,3])/2],axis=-1)


def loss(y,wave,input_period):
    """All candidates share the INPUT waveform-energy denominator at each horizon."""
    pred=np.einsum('pc,...tc->...pt',PHASE,wave);ideal=PHASE@design(input_period,y.shape[-1],'tone_only').T
    return np.sum((y-pred)**2,axis=(1,2))/np.sum(ideal**2)


def curve(y,input_period,variant):
    """Shared maps and entire-source CV; no TEST data argument or selection path."""
    horizon=y.shape[-1];axes=odd(y);coefficients=[];waves=[];cvs=[];trains=[];cvloss=[]
    for period in OUTPUT_PERIODS:
        d=design(period,horizon,variant);coef=np.linalg.lstsq(d,axes.mean(0),rcond=None)[0];wave=d@coef
        cv=np.stack([d@np.linalg.lstsq(d,axes[np.arange(len(y))!=held].mean(0),rcond=None)[0] for held in range(len(y))])
        coefficients.append(coef);waves.append(wave);cvs.append(cv);trains.append(loss(y,wave,input_period));cvloss.append(loss(y,cv,input_period))
    cvloss=np.stack(cvloss);index=int(np.argmin(cvloss.mean(1)))
    meanwave=axes.mean(0);cvmean=np.stack([axes[np.arange(len(y))!=held].mean(0) for held in range(len(y))])
    return dict(selected=index,selected_period=OUTPUT_PERIODS[index],coefficients=np.stack(coefficients),waveforms=np.stack(waves),cv_waveforms=np.stack(cvs),
                train_losses=np.stack(trains),cv_losses=cvloss,mean_waveform=meanwave,mean_train_losses=loss(y,meanwave,input_period),mean_cv_losses=loss(y,cvmean,input_period))


def draws(types):
    """Reuse one source bootstrap across every amplitude/horizon endpoint."""
    types=np.asarray(types);rng=np.random.default_rng(4954600)
    return np.concatenate([rng.choice(np.flatnonzero(types==k),(16384,np.sum(types==k))) for k in sorted(set(types))],axis=1)


def summary(values,indices):
    """Paired source effect and descriptive nominal/family4 percentile intervals."""
    x=np.asarray(values);boot=x[indices].mean(1)
    return dict(mean=float(x.mean()),source_differences=x.tolist(),positive_sources=int(np.sum(x>0)),percentile95=np.quantile(boot,[.025,.975]).tolist(),family4_percentile=np.quantile(boot,[.00625,.99375]).tolist())


def cohort(out,role,amplitude,period,horizon):
    """Read only the declared cohort rows of the same captured direct64 array."""
    groups=sorted([g for g in read(out/'groups.json') if g['cohort']==role and g['amplitude']==amplitude and g['input_period']==period],key=lambda g:g['cohort_source_id'])
    if len(groups)!=16:raise ValueError('Need16 independent cohort sources')
    f=np.load(out/'bolt_native_forecast64.npy',mmap_mode='r')
    y=np.stack([(f[g['indices'],:horizon].astype(float)-f[g['base'],:horizon].astype(float))/g['physical_amplitude'] for g in groups])
    return y,groups


def validate_capture(out):
    """No analysis after capability/resource/identity failure or changed capture bytes."""
    p=validate(out);execution=read(out/'execution.json')['bolt'];worker=read(out/'bolt_worker.json')
    if not execution['valid'] or not all(worker['gates'].values()):raise ValueError('Native direct-head gates failed')
    for file,digest in worker['hashes'].items():
        if sha(out/file)!=digest:raise ValueError('Changed native capture')
    for file,digest in p['prior_hashes'].items():
        if sha(file)!=digest:raise ValueError('Changed historical reference source')
    full=np.load(out/'bolt_native_forecast64.npy',mmap_mode='r');prefix=np.load(out/'bolt_native_forecast32.npy',mmap_mode='r')
    if not np.array_equal(full[:,:32],prefix):raise ValueError('Saved prefix mismatch')
    return p


def fit(out):
    """Fit and freeze all OLD maps before evaluating TEST curves."""
    p=validate_capture(out)
    if (out/'fit_freeze.json').exists():raise ValueError('Refuse overwrite old fits')
    arrays={};rows=[]
    for horizon in (32,64):
        for amplitude in AMPLITUDES:
            for period in INPUT_PERIODS:
                y,groups=cohort(out,'FIT',amplitude,period,horizon);indices=draws([g['background_type'] for g in groups])
                for variant in VARIANTS:
                    key=f'{horizon}_{amplitude}_{period}_{variant}';fitted=curve(y,period,variant);arrays[f'{key}_FIT_delta']=y
                    for name,value in fitted.items():
                        if isinstance(value,np.ndarray):arrays[f'{key}_{name}']=value
                    cv=fitted['cv_losses'];means=cv.mean(1);order=np.argsort(means,kind='stable');winners=np.argmin(cv[:,indices].mean(2),axis=0)
                    selected=fitted['selected'];d=design(fitted['selected_period'],horizon,variant);h=fitted['coefficients'][selected][:2]
                    rows.append(dict(horizon=horizon,amplitude=amplitude,input_period=period,variant=variant,selected_index=selected,selected_period=fitted['selected_period'],
                        same_index=OUTPUT_PERIODS.index(period),fit_train_curve=fitted['train_losses'].mean(1).tolist(),fit_cv_curve=means.tolist(),runner_up_gap=float(means[order[1]]-means[order[0]]),
                        fixed_cv_bootstrap_minimizers={str(v):int(np.sum(winners==i)) for i,v in enumerate(OUTPUT_PERIODS)},
                        selected_H_singular_values=np.linalg.svd(h,compute_uv=False).tolist(),selected_H_rank=int(np.linalg.matrix_rank(h)),design_condition=float(np.linalg.cond(d)),
                        mean_waveform_singular_values=np.linalg.svd(fitted['mean_waveform'],compute_uv=False).tolist()))
    np.savez_compressed(out/'fit_arrays.npz',**arrays);write(out/'fit_selection.json',rows)
    write(out/'fit_freeze.json',dict(protocol_sha256=sha(out/'protocol.json'),test_evaluated=False,files={n:sha(out/n) for n in ['fit_arrays.npz','fit_selection.json']}))
    k=read(out/'knowledge.json');k.update(status='old_fits_frozen_before_test',exact_findings=dict(fit_selection=rows),next_decision='Evaluate fixed OLD selections on TEST only; no period/variant selection from TEST.');write(out/'knowledge.json',k)


def evaluate(out):
    """TEST curves, fixed selected-map gains and paired amplitude/horizon interactions."""
    validate_capture(out);freeze=read(out/'fit_freeze.json')
    if freeze['protocol_sha256']!=sha(out/'protocol.json'):raise ValueError('Changed protocol')
    for file,digest in freeze['files'].items():
        if sha(out/file)!=digest:raise ValueError('Changed OLD fit freeze')
    if (out/'results.json').exists():raise ValueError('Refuse overwrite TEST results')
    fitted=np.load(out/'fit_arrays.npz');arrays={};rows=[]
    for r in read(out/'fit_selection.json'):
        horizon=r['horizon'];amplitude=r['amplitude'];period=r['input_period'];variant=r['variant'];key=f'{horizon}_{amplitude}_{period}_{variant}'
        y,groups=cohort(out,'TEST',amplitude,period,horizon);indices=draws([g['background_type'] for g in groups])
        losses=np.stack([loss(y,w,period) for w in fitted[f'{key}_waveforms']]);meanloss=loss(y,fitted[f'{key}_mean_waveform'],period)
        selected=losses[r['selected_index']];same=losses[r['same_index']]
        ideal=PHASE@design(period,horizon,'tone_only').T;even=np.stack([y[:,0]+y[:,2],y[:,1]+y[:,3]],axis=1)
        evenloss=np.sum(even**2,axis=(1,2))/(2*np.sum(ideal**2))
        arrays.update({f'{key}_TEST_delta':y,f'{key}_test_losses':losses,f'{key}_mean_test_losses':meanloss,f'{key}_even_floor':evenloss})
        rows.append(dict(r,test_curve=losses.mean(1).tolist(),same_test_nmse=float(same.mean()),selected_test_nmse=float(selected.mean()),mean_waveform_test_nmse=float(meanloss.mean()),
            mean_even_floor=float(evenloss.mean()),same_minus_selected=summary(same-selected,indices),selected_minus_waveform=summary(selected-meanloss,indices),
            by_type=[dict(type=k,gain=float((same-selected)[[g['background_type']==k for g in groups]].mean())) for k in range(4)]))
    interactions=[]
    def gains(h,a):
        """Return a fixed primary cell's paired source gain vector."""
        return np.asarray(next(r['same_minus_selected']['source_differences'] for r in rows if r['horizon']==h and r['amplitude']==a and r['input_period']==19.2 and r['variant']=='tone_only'))
    for h in (32,64):interactions.append(dict(contrast='weak_minus_strong_gain',horizon=h,**summary(gains(h,.1)-gains(h,2.),indices)))
    for a in AMPLITUDES:interactions.append(dict(contrast='64_minus32_gain',amplitude=a,**summary(gains(64,a)-gains(32,a),indices)))
    np.savez_compressed(out/'test_arrays.npz',**arrays);write(out/'results.json',dict(rows=rows,interactions=interactions))
    k=read(out/'knowledge.json');k.update(status='computed_pending_audit',exact_findings=dict(rows=rows,interactions=interactions),next_decision='Independent saved-array replay before scientific interpretation; no automatic expansion.');write(out/'knowledge.json',k)
