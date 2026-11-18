"""Known-answer independent arithmetic and saved downprojection controls."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.feature_search import ffn_selectivity_causal_replication_audit as audit
from experiments.feature_search.ffn_selectivity_downprojection_diagnostic import projected_energy


def test_independent_event_reconstruction_detects_collateral_and_response_sign():
    base=np.ones((2,4));clean=base+1;patched=clean.copy();patched[:,1]=base[:,1]
    data=dict(clean_FFN=np.array([[base,clean]]),patched_FFN=np.array([patched]),
        clean_forecasts=np.array([[[0.,0.],[1.,0.]]]),patched_forecasts=np.array([[.5,0.]]))
    event=dict(source_index=0,condition=0,index=0)
    result=audit.reconstruct(data,event,[1])
    assert result['signed_response_removal']==.5
    assert result['remaining_energy_ratio']==.25 and result['orthogonal_change_ratio']==0
    assert result['actual_selected_edit_L2']==pytest.approx(np.sqrt(2))
    data['patched_FFN'][0,0,0]=0
    with pytest.raises(ValueError,match='support'):
        audit.reconstruct(data,event,[1])


def test_independent_source_CI_rejects_token_counts_and_tampering():
    result=audit.source_statistics(np.array([1e8,1.,3.]),['excluded','a','b'],np.array([False,True,True]))
    assert result==dict(n_sources=2,source_ids=['a','b'],excluded_sources=['excluded'],mean=2.,ci95=[1.,3.])
    audit.check_statistics(result,result)
    wrong=deepcopy(result);wrong['ci95']=[0.,4.]
    with pytest.raises(ValueError,match='statistic'):
        audit.check_statistics(result,wrong)
    wrong=deepcopy(result);wrong['n_sources']=32
    with pytest.raises(ValueError,match='identities/counts'):
        audit.check_statistics(result,wrong)


def test_equal_input_activity_does_not_imply_equal_projected_energy():
    weight=np.array([[100.,0.,2.,0.],[0.,.01,0.,3.]])
    first=np.array([[1.,0.,0.,0.]]);second=np.array([[0.,1.,0.,0.]])
    assert np.sum(first**2)==np.sum(second**2)==1
    assert projected_energy(weight,first,[0])==10000
    assert projected_energy(weight,second,[1])==.0001
    both=np.array([[1.,2.,0.,0.],[-2.,1.,0.,0.]])
    assert projected_energy(weight,both,[0,1])==pytest.approx(float(np.sum((both@weight.T)**2)))
    with pytest.raises(ValueError,match='support'):
        projected_energy(weight,both,[0])
