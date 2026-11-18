"""Known-answer shared inverse allocation,placeholder order and near-zero safeguards."""
import importlib.util
import sys
from pathlib import Path
import numpy as np


def _module():
    """Load the prospective accurate-comparator audit without model construction."""
    root=Path(__file__).parents[1]/'experiments/feature_search';sys.path.insert(0,str(root));s=importlib.util.spec_from_file_location('chronos2_head_audit',root/'chronos2_head_audit.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_common_secant_uses_shared_inverse_and_continuous_limit():
    """Opposite branch changes are a cancellation decoy for independently inverted branches."""
    m=_module();plus=np.array([1.,1+1e-12,2.]);minus=np.array([1.,1-1e-12,-.5]);j=m.common_secant(plus,minus)
    np.testing.assert_allclose(j*(plus-minus),np.sinh(plus)-np.sinh(minus),atol=1e-12)
    assert abs(j[0]-np.cosh(1))<1e-14
    assert abs(j[1]-np.cosh(1))<1e-12
    skip_change=np.array([.4,.4,.4]);nl_change=(plus-minus)-skip_change
    np.testing.assert_allclose(j*skip_change+j*nl_change,np.sinh(plus)-np.sinh(minus),atol=1e-12)


def test_point_layout_preserves_both_forecast_patches():
    """The second placeholder and neighboring quantile are explicit indexing decoys."""
    m=_module();flat=np.arange(672).reshape(1,2,336)
    expected=np.concatenate([np.arange(16)+160,np.arange(16)+496])
    np.testing.assert_equal(m.median_point(flat)[0],expected)


def test_absolute_energy_denominator_cancellation_and_projection_floor():
    """Near-zero native output must not yield unstable total-relative projection shares."""
    m=_module();skip=np.tile([2.,0.],(4,1));nl=-skip;strength=np.array([.001,1,10,100]);total=skip+nl
    report=m.absolute_attribution(total,{'skip':skip,'nonlinear':nl},strength)
    assert abs(report['branches']['skip']['energy_ratio']['value']-16/strength.sum())<1e-12
    assert report['total_energy_ratio']['value']==0
    assert report['branches']['skip']['signed_projection_share']['value'] is None
    assert report['descriptive_label']=='cancellation'
    weak=m.absolute_attribution(np.tile([.01,.01],(4,1)),{'skip':np.tile([.01,0],(4,1)),'nonlinear':np.tile([0,.01],(4,1))},np.ones(4))
    assert weak['descriptive_label']=='weak branch mixing'


def test_near_zero_replay_has_frozen_absolute_alternative():
    """A tiny native total is a decoy for rejecting an otherwise negligible replay mismatch."""
    m=_module();actual=np.zeros((4,2));replayed=np.full((4,2),1e-7);strength=np.ones(4)
    gate=m.replay_gate(1e-7,actual,replayed,strength)
    assert gate['relative_RMS'] is None and gate['passed']
    assert not m.replay_gate(1e-7,actual,np.full((4,2),1e-3),strength)['passed']
    assert not m.replay_gate(1e-4,actual,replayed,strength)['passed']


def test_head_replay_keeps_two_positions():
    """A loud second placeholder cannot be dropped as if this were a one-query Bolt head."""
    m=_module();h=np.zeros((1,2,768));h[0,:,0]=[1,3]
    hidden=np.zeros((1,768));hidden[0,0]=1;residual=np.zeros((336,768));residual[:,0]=1
    weights={'hidden_layer.weight':hidden,'hidden_layer.bias':np.zeros(1),'output_layer.weight':np.ones((336,1)),'output_layer.bias':np.zeros(336),'residual_layer.weight':residual,'residual_layer.bias':np.zeros(336)}
    skip,nl=m.replay_branches(h,weights)
    expected=np.array([1]*16+[3]*16)[None,:]
    np.testing.assert_equal(skip,expected);np.testing.assert_equal(nl,expected)
