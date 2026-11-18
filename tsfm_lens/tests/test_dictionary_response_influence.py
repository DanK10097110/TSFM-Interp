"""Algebra and planted wrong-aggregation checks for saved-only influence analysis."""
import numpy as np
import pytest
from experiments.feature_search import dictionary_response_influence as diag


def test_error_projection_energy_and_signed_gain():
    a=np.array([1.,0.]);decoded=np.array([3.,4.])
    r=diag.decompose(a,decoded)
    assert r['error_parallel_energy']==2 and r['error_orthogonal_energy']==8
    assert r['error_energy']==10 and r['nmse']==20
    assert r['error_parallel_signed_gain']==2 and r['decoded_parallel_gain']==3
    assert r['parallel_orthogonal_inner_product']==0
    assert r['decoded_energy']==r['native_energy']+2*r['native_error_inner_product']+r['error_energy']
    wrong_decoded_projection=np.array([3.,0.])
    assert np.mean(wrong_decoded_projection**2)!=r['error_parallel_energy']
    with pytest.raises(ValueError):diag.decompose(np.zeros(2),decoded)


def test_scale_invariance_and_orthogonal_error():
    a=np.array([1.,2.,3.]);b=np.array([-2.,4.,1.])
    r=diag.decompose(a,b);scaled=diag.decompose(a*7,b*7)
    assert scaled['nmse']==pytest.approx(r['nmse'])
    assert scaled['error_energy']==pytest.approx(49*r['error_energy'])
    assert r['error_energy']==pytest.approx(r['error_parallel_energy']+r['error_orthogonal_energy'])


def test_influence_uses_paired_deterioration_and_same_sources():
    rows=[dict(source='a',lambda0_nmse=100,lambda1_nmse=50),dict(source='b',lambda0_nmse=1,lambda1_nmse=5),dict(source='c',lambda0_nmse=1,lambda1_nmse=3),dict(source='d',lambda0_nmse=2,lambda1_nmse=1)]
    values=diag.influence(rows)
    assert values[1]['removed_sources']==['b']
    assert values[2]['removed_sources']==['b','c']
    assert values[0]['n_sources']==4 and values[2]['lambda0_mean']==51 and values[2]['lambda1_mean']==25.5
    assert values[2]['removed_positive_deterioration_share']==1
    assert max(rows,key=lambda r:r['lambda1_nmse'])['source']!='b'


def test_association_ties_and_constant_guard():
    r=diag.association([1,1,2,3],[3,3,2,1])
    assert r['spearman']==pytest.approx(-1)
    assert diag.association([1,1,1],[1,2,3])['pearson'] is None
