"""Independent raw-forecast replay of output-frequency selection and transfer."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha


def audit(out):
    """Replay from saved physical forecasts using normal equations and explicit phase sums."""
    p=read(out/'protocol.json');freeze=read(out/'fit_freeze.json')
    for file,digest in p['bindings'].items():assert sha(file)==digest,file
    assert freeze['protocol_sha256']==sha(out/'protocol.json')
    for file,digest in freeze['files'].items():assert sha(out/file)==digest,file
    assert freeze['test_analysis_performed'] is False
    assert set(p['source_roles']['FIT']).isdisjoint(p['source_roles']['TEST'])
    fitted=np.load(out/'fit_arrays.npz');tested=np.load(out/'test_arrays.npz');selections=read(out/'fit_selection.json')
    results=read(out/'results.json');periods=p['candidate_output_periods']
    assert periods==[16.,18.,19.2,20.,22.,24.,26.,28.,32.]
    checks=0;maximum=0.
    phase=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    def close(a,b):
        """Track discrepancies and reject arithmetic changes, including source weights."""
        nonlocal checks,maximum
        maximum=max(maximum,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11);checks+=1
    def matrix(period,variant):
        """Construct absolute future-time design independently of the study helper."""
        time=np.arange(32,dtype=float);columns=[np.sin(time*2*np.pi/period),np.cos(time*2*np.pi/period)]
        if variant=='tone_dc_trend':columns += [np.ones(32),np.linspace(-1,1,32)]
        return np.asarray(columns).T
    def losses(y,w,denominator):
        """Independent explicit-phase error sum, always divided by input signal energy."""
        if w.ndim==2:w=np.repeat(w[None],16,axis=0)
        return np.asarray([sum(np.dot(error,error) for error in [phase[k]@w[sid].T-y[sid,k] for k in range(4)])/denominator for sid in range(16)])
    for selection,result in zip(selections,results):
        assert (selection['model'],selection['input_period'],selection['variant'])==(result['model'],result['input_period'],result['variant'])
        model=selection['model'];period=selection['input_period'];variant=selection['variant'];key=f'{model}_{period}_{variant}'
        ys={};groupings={}
        for cohort in ['FIT','TEST']:
            cfg=p['cohorts'][cohort];groups=read(cfg['groups'])
            selected=sorted([g for g in groups if abs(g['frequency']-128/period)<1e-12 and g.get('offset',0.)==0.],key=lambda r:r['source_id'])
            assert len(selected)==16 and {g['source'] for g in selected}==set(p['source_roles'][cohort])
            forecasts=np.load(Path(cfg['forecasts'])/f'{model}_native_forecast.npy')
            y=np.asarray([(forecasts[g['indices']].astype(float)-forecasts[g['base']].astype(float))/g['amplitude'] for g in selected])
            close(y,(fitted if cohort=='FIT' else tested)[f'{key}_{cohort}_delta'])
            ys[cohort]=y;groupings[cohort]=selected
        y=ys['FIT'];test_y=ys['TEST'];odd=np.transpose(np.asarray([(y[:,0]-y[:,2])/2,(y[:,1]-y[:,3])/2]),(1,2,0))
        denominator=float(sum(np.dot(phase[k]@matrix(period,'tone_only').T,phase[k]@matrix(period,'tone_only').T) for k in range(4)))
        cv_curves=[];test_curves=[]
        for index,candidate in enumerate(periods):
            d=matrix(candidate,variant);coef=np.linalg.solve(d.T@d,d.T@odd.mean(axis=0));wave=d@coef
            close(coef,fitted[f'{key}_coefficients'][index]);close(wave,fitted[f'{key}_waveforms'][index])
            train_loss=losses(y,wave,denominator);close(train_loss,fitted[f'{key}_train_losses'][index])
            cv=[]
            for held in range(16):
                old_mean=sum(odd[s] for s in range(16) if s!=held)/15
                cv.append(d@np.linalg.solve(d.T@d,d.T@old_mean))
            cv=np.stack(cv);close(cv,fitted[f'{key}_cv_waveforms'][index])
            cv_loss=losses(y,cv,denominator);close(cv_loss,fitted[f'{key}_cv_losses'][index]);cv_curves.append(cv_loss)
            test_loss=losses(test_y,wave,denominator);close(test_loss,tested[f'{key}_test_losses'][index]);test_curves.append(test_loss)
            geometry=selection['geometry'][index]
            assert geometry['design_rank']==np.linalg.matrix_rank(d)
            close(np.linalg.svd(d,compute_uv=False),geometry['design_singular_values'])
            close(np.linalg.cond(d),geometry['condition_number'])
        cv_curves=np.stack(cv_curves);test_curves=np.stack(test_curves)
        best=int(np.argmin(cv_curves.mean(axis=1)));assert best==selection['selected_index']==result['selected_index']
        assert periods[best]==selection['selected_period']==result['selected_period']
        close(cv_curves.mean(axis=1),result['fit_cv_curve']);close(test_curves.mean(axis=1),result['test_curve'])
        meanwave=odd.mean(axis=0);close(meanwave,fitted[f'{key}_mean_waveform'])
        close(losses(y,meanwave,denominator),fitted[f'{key}_mean_waveform_train_losses'])
        cvmean=np.stack([sum(odd[s] for s in range(16) if s!=held)/15 for held in range(16)])
        close(cvmean,fitted[f'{key}_cv_mean_waveforms']);close(losses(y,cvmean,denominator),fitted[f'{key}_mean_waveform_cv_losses'])
        testmean=losses(test_y,meanwave,denominator);close(testmean,tested[f'{key}_mean_waveform_test_losses'])
        types=np.asarray([g['background_type'] for g in groupings['TEST']]);rng=np.random.default_rng(4954400)
        draws=np.column_stack([rng.choice(np.where(types==k)[0],size=(16384,4)) for k in range(4)])
        same=test_curves[periods.index(period)];chosen=test_curves[best]
        for field,difference in [('same_minus_selected',same-chosen),('selected_minus_mean_waveform',chosen-testmean)]:
            close(difference,result[field]['source_differences']);close(difference.mean(),result[field]['mean'])
            assert result[field]['wins']==int(np.sum(difference>0))
            boot=difference[draws].mean(axis=1)
            close(np.quantile(boot,[.025,.975]),result[field]['percentile95'])
            close(np.quantile(boot,[.0125,.9875]),result[field]['primary_family2_percentile'])
        for row in result['by_type']:close((same-chosen)[types==row['background_type']].mean(),row['mean_same_minus_selected'])
        winners=np.argmin(cv_curves[:,draws].mean(axis=2),axis=0)
        assert {str(f):int(np.sum(winners==i)) for i,f in enumerate(periods)}==selection['cv_fixed_loss_bootstrap_selection_counts']
        close(chosen.mean(),result['selected_output_test_nmse']);close(same.mean(),result['same_input_test_nmse']);close(testmean.mean(),result['mean_waveform_test_nmse'])
    audit_result=dict(passed=True,checks=checks,maximum_absolute_discrepancy=maximum,
        source_hashes_verified=True,old_only_selection_replayed=True,source_cv_replayed=True,
        all_nine_test_curves_replayed=True,independent_sources_per_cohort=16,native_capture=False)
    write(out/'audit.json',audit_result)
    return audit_result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    print(audit(parser.parse_args().out))
