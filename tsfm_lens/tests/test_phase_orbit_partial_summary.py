"""Known-answer disjoint energy summary with large gain/DC/high-order decoys."""
import copy
import pytest
from experiments.feature_search import phase_orbit_partial_summary as summary


def budget():
    modes=[dict(k=1,l=0,category='expected_axis_shape',predicted_energy=9.,physical_time_character_residual_energy=.2,error_energy=2.),
        dict(k=2,l=0,category='higher_axis',predicted_energy=.3,physical_time_character_residual_energy=.1,error_energy=.3),
        dict(k=-1,l=1,category='joint',predicted_energy=.4,physical_time_character_residual_energy=.01,error_energy=.4),
        dict(k=0,l=0,category='phase_DC',predicted_energy=11.,physical_time_character_residual_energy=0.,error_energy=4.),
        dict(k=4,l=0,category='higher_axis',predicted_energy=20.,physical_time_character_residual_energy=19.,error_energy=20.)]
    return dict(primary_D=.9,modes=modes,scaled_MSE=26.7,expected_axis_gain_rotation_error=1.8,
                error_categories=dict(expected_axis_shape=2.,higher_axis=20.3,joint=.4,phase_DC=4.))


def test_disjoint_terms_ignore_large_gain_dc_and_outside_order_decoys():
    terms,contributions=summary.disjoint_terms(budget())
    assert terms==pytest.approx(dict(expected_axis_wrong_time=.2,higher_axis=.3,joint=.4))
    assert sum(terms.values())==pytest.approx(.9)
    assert '0,0' not in contributions and '4,0' not in contributions
    assert contributions['1,0']==pytest.approx(.2)
    bad=budget();bad['primary_D']=.8
    with pytest.raises(ValueError,match='close'):summary.disjoint_terms(bad)


def test_all_low_order_modes_retained_and_aggregate_has_exact_answer():
    first=budget();second=copy.deepcopy(first)
    for mode in second['modes']:
        for key in ('predicted_energy','physical_time_character_residual_energy','error_energy'):mode[key]*=3
    second['primary_D']*=3;second['scaled_MSE']*=3;second['expected_axis_gain_rotation_error']*=3
    second['error_categories']={k:v*3 for k,v in second['error_categories'].items()}
    result=summary.aggregate([first,second])
    assert result['D']==pytest.approx(1.8)
    assert result['D_terms']['expected_axis_wrong_time']==pytest.approx(.4)
    assert len(result['low_order_modes'])==4  # Includes the DC scientific decoy.
    axis=next(r for r in result['low_order_modes'] if (r['k'],r['l'])==(1,0))
    assert axis['predicted_energy']==pytest.approx(18.)
    assert axis['D_contribution']==pytest.approx(.4)
