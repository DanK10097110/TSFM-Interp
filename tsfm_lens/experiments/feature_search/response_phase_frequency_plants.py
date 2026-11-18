"""In-memory output-frequency faults; all clean fixtures must pass before mutation."""
import importlib.util
import json
from pathlib import Path
import pytest


def run():
    """Exercise frequency/phase/nuisance, CV leakage, weighting, capacity and freeze guards."""
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('frequency_tests',root/'tests/test_response_phase_frequency.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    path=root/'experiments/feature_search/response_phase_frequency.py';source=path.read_text(encoding='utf-8')
    cases=[
        ('hardcode_input_period','fit_curve','selected = int(np.argmin(cv_losses.mean(axis=1)))','selected = PERIODS.index(input_period)','test_known_frequency_shift_selects_output24_not_input19point2'),
        ('phase_lag_mistaken_for_shift','fit_curve','selected = int(np.argmin(cv_losses.mean(axis=1)))','selected = PERIODS.index(24.)','test_pure_phase_rotation_keeps_input_frequency_and_rank_one_visible'),
        ('nuisance_omitted','design',"return np.column_stack([tone, np.ones(32), (t-15.5)/15.5])","return tone",'test_joint_dc_trend_prevents_false_frequency_shift'),
        ('mean_phase_ratios','loss_per_source','np.sum((y-prediction)**2, axis=(1, 2))/np.sum(ideal**2)','np.mean(np.sum((y-prediction)**2,axis=2)/np.sum(ideal**2,axis=1),axis=1)','test_input_energy_denominator_is_shared_across_output_candidates'),
        ('receiver_in_cv','cv_predictions','fit_map(y[np.arange(len(y)) != i], output_period, variant)','fit_map(y, output_period, variant)','test_source_loso_excludes_all_receiver_phases'),
        ('waveform_baseline_tone_projection','fit_curve','mean_wave = odd_waveforms(y).mean(axis=0)',"mean_wave = fit_map(y,input_period,'tone_only')[1]",'test_mean_waveform_capacity_is_explicit_and_preserves_non_tone_shape'),
        ('physical_amplitudes_weight_sources','normalize_delta','/a[:, None, None]','/np.mean(a)','test_amplitude_normalization_and_source_bootstrap'),
        ('phases_as_resampling_units','bootstrap_sources','size=(count, np.sum(types == kind))','size=(count, 4*np.sum(types == kind))','test_amplitude_normalization_and_source_bootstrap'),
        ('changed_fit_accepted','evaluate',"if sha(out/file)!=digest: raise ValueError('Frozen FIT changed')","if False: raise ValueError('Frozen FIT changed')",'test_test_evaluation_rejects_changed_fit_before_loading_test'),
    ]
    for name in sorted({r[-1] for r in cases}):getattr(tests,name)()
    results=[]
    for name,symbol,old,new,test in cases:
        assert source.count(old)==1,(name,source.count(old))
        space=dict(__name__='frequency_mutant',__file__=str(path));exec(compile(source.replace(old,new),str(path),'exec'),space)
        original=getattr(tests.study,symbol);setattr(tests.study,symbol,space[symbol])
        space['validate']=lambda *a,**kw:tests.study.validate(*a,**kw)
        space['load_cohort']=lambda *a,**kw:tests.study.load_cohort(*a,**kw)
        try:
            try:getattr(tests,test)()
            except (AssertionError,ValueError,IndexError,pytest.fail.Exception):results.append(dict(plant=name,test=test,detected=True))
            else:raise AssertionError(f'Plant survived {name}')
        finally:setattr(tests.study,symbol,original)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
