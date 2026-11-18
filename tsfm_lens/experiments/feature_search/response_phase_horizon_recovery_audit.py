"""Independent recovery replay, retaining original selection arithmetic and explicitly auditing the amended historical comparator."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_phase_horizon_recovery import validate_recovery

def audit(out,results=False):
    """Use raw forecasts and normal equations, independent of analysis fitting helpers."""
    p=validate_recovery(out);checks=0;maximum=0.
    def close(a,b):
        """Reject arithmetic mismatches while tracking the maximum numerical residual."""
        nonlocal checks,maximum
        maximum=max(maximum,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11);checks+=1
    for file,digest in p['source_hashes'].items():assert sha(file)==digest,file
    for file,digest in p['hashes'].items():assert sha(out/file)==digest,file
    for file,digest in p['prior_hashes'].items():assert sha(file)==digest,file
    assert p['models']==['bolt'] and p['horizon']==64 and p['seconds_per_model']==300
    inputs=np.load(out/'inputs.npy');groups=read(out/'groups.json');sources=read(out/'sources.json')
    mask=np.load(out/'historical_reference_mask.npy');reference=np.load(out/'historical_prefix32.npy')
    assert inputs.shape==(544,128) and len(groups)==128 and mask.sum()==288
    assert {s['source'] for s in sources if s['cohort']=='FIT'}.isdisjoint({s['source'] for s in sources if s['cohort']=='TEST'})
    for g in groups:
        t=np.arange(-128,0);expected=np.stack([g['physical_amplitude']*np.sin(2*np.pi*t/g['input_period']+k*np.pi/2) for k in range(4)])
        measured=inputs[g['indices']].astype(float)-inputs[g['base']].astype(float)
        assert np.sqrt(np.mean((measured-expected)**2))/g['physical_amplitude']<1e-5;checks+=1
    if results:
        worker=read(out/'bolt_worker.json');execution=read(out/'execution.json')['bolt']
        assert execution['valid'] and all(worker['gates'].values())
        for file,digest in worker['hashes'].items():assert sha(out/file)==digest
        full=np.load(out/'bolt_native_forecast64.npy');prefix=np.load(out/'bolt_native_forecast32.npy')
        assert np.array_equal(full[:,:32],prefix);checks+=1
        assert set(worker['gates'])==set(read(out/'amendment.json')['mandatory_gates'])
        assert worker['amendment_sha256']==sha(out/'amendment.json')
        assert np.load(out/'filled_mask.npy').all()
        np.testing.assert_array_equal(np.load(out/'historical_delta32.npy')[mask],prefix[mask].astype(float)-reference[mask].astype(float))
        checks+=1
        freeze=read(out/'fit_freeze.json');assert freeze['protocol_sha256']==sha(out/'protocol.json') and freeze['test_evaluated'] is False
        for file,digest in freeze['files'].items():assert sha(out/file)==digest
        fitted=np.load(out/'fit_arrays.npz');tested=np.load(out/'test_arrays.npz');rows=read(out/'results.json')['rows']
        phase=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        def design(period,h,variant):
            """Independent horizon basis from absolute future timestamps."""
            t=np.arange(h);d=np.stack([np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period)],axis=1)
            return d if variant=='tone_only' else np.column_stack([d,np.ones(h),(t-(h-1)/2)/((h-1)/2)])
        def loss(y,w,den):
            """Explicit four-phase numerator, source by source."""
            if w.ndim==2:w=np.repeat(w[None],16,axis=0)
            return np.asarray([sum(np.sum((phase[k]@w[i].T-y[i,k])**2) for k in range(4))/den for i in range(16)])
        for row in rows:
            h=row['horizon'];a=row['amplitude'];period=row['input_period'];variant=row['variant'];key=f'{h}_{a}_{period}_{variant}';ys={}
            for role in ('FIT','TEST'):
                gs=sorted([g for g in groups if g['cohort']==role and g['amplitude']==a and g['input_period']==period],key=lambda g:g['cohort_source_id'])
                assert len(gs)==16
                y=np.asarray([(full[g['indices'],:h].astype(float)-full[g['base'],:h].astype(float))/g['physical_amplitude'] for g in gs]);ys[role]=y
                close(y,(fitted if role=='FIT' else tested)[f'{key}_{role}_delta'])
            old=ys['FIT'];test=ys['TEST'];odd=np.stack([(old[:,0]-old[:,2])/2,(old[:,1]-old[:,3])/2],axis=-1)
            den=np.sum((phase@design(period,h,'tone_only').T)**2);cv_losses=[];test_losses=[]
            for i,output_period in enumerate(p['output_period_candidates']):
                d=design(output_period,h,variant);coef=np.linalg.solve(d.T@d,d.T@odd.mean(0));wave=d@coef
                close(coef,fitted[f'{key}_coefficients'][i]);close(wave,fitted[f'{key}_waveforms'][i])
                cv=np.stack([d@np.linalg.solve(d.T@d,d.T@(sum(odd[j] for j in range(16) if j!=held)/15)) for held in range(16)])
                close(cv,fitted[f'{key}_cv_waveforms'][i]);cv_losses.append(loss(old,cv,den));test_losses.append(loss(test,wave,den))
                close(loss(old,wave,den),fitted[f'{key}_train_losses'][i]);close(cv_losses[-1],fitted[f'{key}_cv_losses'][i]);close(test_losses[-1],tested[f'{key}_test_losses'][i])
            cv_losses=np.stack(cv_losses);test_losses=np.stack(test_losses)
            assert int(np.argmin(cv_losses.mean(1)))==row['selected_index']
            assert p['output_period_candidates'][row['selected_index']]==row['selected_period']
            close(cv_losses.mean(1),row['fit_cv_curve']);close(test_losses.mean(1),row['test_curve'])
            close(odd.mean(0),fitted[f'{key}_mean_waveform']);meanloss=loss(test,odd.mean(0),den);close(meanloss,tested[f'{key}_mean_test_losses'])
            even=np.stack([test[:,0]+test[:,2],test[:,1]+test[:,3]],axis=1)
            close(np.sum(even**2,axis=(1,2))/(2*den),tested[f'{key}_even_floor'])
            types=np.asarray([g['background_type'] for g in gs]);rng=np.random.default_rng(4954600)
            draws=np.column_stack([rng.choice(np.flatnonzero(types==k),size=(16384,4)) for k in range(4)])
            same=test_losses[row['same_index']];selected=test_losses[row['selected_index']]
            for field,difference in [('same_minus_selected',same-selected),('selected_minus_waveform',selected-meanloss)]:
                close(difference,row[field]['source_differences']);close(difference.mean(),row[field]['mean'])
                close(np.quantile(difference[draws].mean(1),[.00625,.99375]),row[field]['family4_percentile'])
        primary={(r['horizon'],r['amplitude']):np.asarray(r['same_minus_selected']['source_differences']) for r in rows if r['input_period']==19.2 and r['variant']=='tone_only'}
        for item in read(out/'results.json')['interactions']:
            difference=primary[(item['horizon'],.1)]-primary[(item['horizon'],2.)] if item['contrast']=='weak_minus_strong_gain' else primary[(64,item['amplitude'])]-primary[(32,item['amplitude'])]
            close(difference,item['source_differences']);close(difference.mean(),item['mean'])
    record=dict(passed=True,checks=checks,maximum_absolute_discrepancy=maximum,results_checked=results,native_loaded=False,
                nodes=544,old_fit_sources=16,test_sources=16,source_and_artifact_hashes_verified=True)
    write(out/('audit.json' if results else 'preflight_audit.json'),record);return record


def audit_bounds(out):
    """Check measured response perturbations and fixed-predictor Cauchy bounds independently."""
    p=validate_recovery(out);groups=read(out/'groups.json');prefix=np.load(out/'bolt_native_forecast32.npy')
    reference=np.load(out/'historical_prefix32.npy');fitted=np.load(out/'fit_arrays.npz');mask=np.load(out/'historical_reference_mask.npy')
    diagnostics=read(out/'historical_diagnostic.json');records=[];checks=0
    phase=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    for group in groups:
        if group['amplitude']!=.1:continue
        gs=sorted([g for g in groups if g['cohort']==group['cohort'] and g['amplitude']==.1 and g['input_period']==group['input_period']],key=lambda g:g['cohort_source_id'])
        current=(prefix[group['indices']].astype(float)-prefix[group['base']].astype(float))/group['physical_amplitude']
        old=(reference[group['indices']].astype(float)-reference[group['base']].astype(float))/group['physical_amplitude']
        delta=current-old;t=np.arange(32);b=np.stack([np.sin(2*np.pi*t/group['input_period']),np.cos(2*np.pi*t/group['input_period'])],axis=1)
        energy=np.sum((phase@b.T)**2);u=float(np.sqrt(np.sum(delta**2)/energy))
        saved=next(r for r in diagnostics['response_bounds'] if r['cohort']==group['cohort'] and r['source_id']==group['source_id'] and r['input_period']==group['input_period'])
        np.testing.assert_allclose(u,saved['u'],atol=1e-13,rtol=1e-10)
        assert saved['response_delta_l2']<=saved['triangle_l2_bound']+1e-14;checks+=2
        for variant in ('tone_only','tone_dc_trend'):
            key=f"32_0.1_{group['input_period']}_{variant}"
            for i,wave in enumerate(fitted[f'{key}_waveforms']):
                prediction=phase@wave.T;old_error=old-prediction
                old_nmse=float(np.sum(old_error**2)/energy);new_nmse=float(np.sum((current-prediction)**2)/energy)
                bound=float(2*np.sqrt(old_nmse)*u+u*u);actual=abs(new_nmse-old_nmse)
                assert actual<=bound+1e-11;checks+=1
                records.append(dict(cohort=group['cohort'],source=group['source'],input_period=group['input_period'],variant=variant,output_period=p['output_period_candidates'][i],actual_nmse_change=actual,bound=bound))
    for record in read(out/'query_records.json'):
        assert sha(out/record['file'])==record['raw_sha256'] and record['calls']==[1,1]
        raw=np.load(out/record['file']);assert np.array_equal(raw['current64'][:,:32],raw['current32']);checks+=1
    result=dict(passed=True,checks=checks,max_actual_nmse_change=max(r['actual_nmse_change'] for r in records),max_bound=max(r['bound'] for r in records),max_measured_u=max(r['u'] for r in diagnostics['response_bounds']),historical_reference_rows=int(mask.sum()),mixed_geometry_is_diagnostic_not_tolerance=True)
    write(out/'roundoff_bound_rows.json',records);write(out/'roundoff_audit.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    print(audit(args.out,not args.preflight))
    if not args.preflight:print(audit_bounds(args.out))
