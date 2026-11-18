"""Fresh restoration status cannot be inferred from conditional discovery alone."""
import copy
import numpy as np
import pytest
from experiments.feature_search import stage49_report as report


def fixture():
    cap=int(1.5*2**30)
    old=dict(resource_limit_exceeded=True,peak_resident_bytes=cap+1)
    corrected=dict(effective_earliest_cut=15,numerical_checks=dict(passed=True),peak_resident_bytes=100,
        execution_equivalence=dict(all_arrays_bitwise_equal=True,array_count=29,all_scientific_fields_exact=True))
    values=dict(native_loss=np.ones(64),component_loss=np.full(64,.8),edited_loss=np.full(64,.6),random_mean_loss=np.ones(64))
    values.update(native_gain=values['native_loss']-values['edited_loss'],random_advantage=values['random_mean_loss']-values['edited_loss'])
    claims={key:dict(positive=64,negative=0,ties=0,n_seeds=64,mean=.4,p_holm=2.**-62) for key in ('native_gain','random_advantage')}
    fresh=dict(status='complete',qualified=True,candidate_cut=15,late_control_cut=20,family_size=2,claims=claims,
        per_seed=dict(seed_ids=list(range(3700000,3700064)),**{k:v.tolist() for k,v in values.items()}),
        gates=dict(reference_gap=.2,recovery_unclipped=2.,mean_suffix_energy=.04,passed=True),
        mean_losses=dict(native=1.,component_reference=.8,anchored=.6,random_mean=1.),descriptive_controls={},
        controller_peak_resident_bytes=10,maximum_simultaneous_task_RSS=110,peak_resident_bytes=100,total_wall_seconds=1.)
    validity=dict(all_four_valid=True,models_exited_before_analysis=True,validity=[dict(valid=True,shard_index=i,
        seed_ids=list(range(3700000+16*i,3700016+16*i)),numerical_checks=dict(passed=True),primary15_random_geometry_pass=True,
        model_released_before_return=True,peak_resident_bytes=100,wall_seconds=1.) for i in range(4)])
    return old,corrected,fresh,validity


def test_fresh64_confirmation_keeps_resource_history_and_unclipped_reference():
    summary=report.summarize_restoration(*fixture())
    assert summary['qualified'] and summary['n_seeds']==64 and summary['cut']==15
    assert summary['gates']['recovery_unclipped']==pytest.approx(2.)
    assert summary['original_resource_failure_retained']
    text=report.restoration_html(dict(followups=dict(restoration=summary)))
    assert 'not a deployable forecasting fix' in text
    assert 'native-to-component-reference gap' in text and 'not the fraction of native error removed' in text
    assert 'hidden states were not persisted' in text
    assert 'failed its memory cap' in text


def test_resource_partial_cohort_or_conditional_discovery_cannot_fake_pass():
    old,corrected,fresh,validity=fixture()
    invalid=copy.deepcopy(validity);invalid['validity'][2]['peak_resident_bytes']=int(1.5*2**30)+1
    with pytest.raises(ValueError,match='resource'):report.summarize_restoration(old,corrected,fresh,invalid)
    invalid=copy.deepcopy(fresh);invalid['per_seed']['seed_ids']=invalid['per_seed']['seed_ids'][:16]
    with pytest.raises(ValueError,match='Fresh64'):report.summarize_restoration(old,corrected,invalid,validity)
    invalid=copy.deepcopy(corrected);invalid['execution_equivalence']['all_scientific_fields_exact']=False
    with pytest.raises(ValueError,match='discovery'):report.summarize_restoration(old,invalid,fresh,validity)
    invalid=copy.deepcopy(fresh);invalid['gates']['mean_suffix_energy']=.051
    with pytest.raises(ValueError,match='qualification'):report.summarize_restoration(old,corrected,invalid,validity)
