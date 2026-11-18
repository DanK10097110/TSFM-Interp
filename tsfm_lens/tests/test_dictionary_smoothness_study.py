"""Linear, hard-boundary, resolution-floor and source-weighting fixtures."""
import numpy as np
import pytest
from experiments.feature_search import dictionary_smoothness_math as math
from experiments.feature_search import dictionary_smoothness_audit as audit


def pca(k=1):
    return dict(center=np.zeros(2,dtype=np.float32),scale=np.array(1.),k=np.array(k),basis=np.eye(2,dtype=np.float32))


def boundary(epsilon=1e-3):
    minus=np.array([[[1+epsilon,1-epsilon]]]);plus=np.array([[[1-epsilon,1+epsilon]]])
    mid=(minus+plus)/2
    return np.stack([minus,plus,mid,mid])


def test_fixed_first_PC_is_linear_but_adaptive_signed_topk_has_boundary_chord():
    h=boundary()
    fixed,support=math.evaluate(h,pca(),'pca_fixed',np.float64)
    linear,_,_=math.measure(h,fixed,support,np.ones(1),np.array([.01]))
    assert linear['isolated_dictionary_chord_RMS'][0]==0
    assert linear['chord_support_turnover'][0]==0
    adaptive,support=math.evaluate(h,pca(),'pca_adaptive',np.float64)
    nonlinear,_,_=math.measure(h,adaptive,support,np.ones(1),np.array([.01]))
    assert nonlinear['raw_chord_RMS'][0]==0
    assert nonlinear['isolated_dictionary_chord_RMS'][0]>0.99
    assert nonlinear['any_chord_support_switch'][0]==1
    assert nonlinear['endpoint_support_turnover'][0]==1
    smooth=np.array([[[[2.,.1]]],[[[2.2,.2]]],[[[2.1,.15]]],[[[2.1,.15]]]])
    g,s=math.evaluate(smooth,pca(),'pca_adaptive',np.float64)
    result,_,_=math.measure(smooth,g,s,np.ones(1),np.array([.01]))
    assert result['isolated_dictionary_chord_RMS'][0]<1e-14


def test_actual_midpoint_is_not_replaced_by_offmanifold_activation_midpoint():
    h=boundary();h[2,0,0,0]+=.1
    g,s=math.evaluate(h,None,'identity',np.float64)
    result,actual,isolated=math.measure(h,g,s,np.ones(1),np.array([.01]))
    assert result['raw_chord_RMS'][0]>0 and result['decoded_actual_chord_RMS'][0]>0
    assert result['isolated_dictionary_chord_RMS'][0]==0
    np.testing.assert_allclose(actual,isolated+2*(g[3]-g[2]),atol=1e-15)


def test_refloor_controls_denominator_artifact_and_keeps_all_values():
    for factor,expected in ((1,1000.),(10,100.),(100,10.)):
        assert math.floor_ratio(np.array([1e-3]),np.array([0.]),np.array([1e-6]),factor)[0]==pytest.approx(expected)
    with pytest.raises(ValueError,match='Positive'):
        math.floor_ratio([1.],[0.],np.array([0.]),1)


def test_positive_tie_zero_shortfall_and_frozen_learned_replay():
    indices=math.top_indices(np.array([[2.,-2.,2.],[0.,0.,0.]]),1)
    assert indices[0,0]==0
    artifact=dict(center=np.zeros(2,dtype=np.float32),scale=np.array(2.),k=np.array(1),
        b_dec=np.array([.2,.3],np.float32),W_enc=np.eye(2,dtype=np.float32),b_enc=np.zeros(2,np.float32),W_dec=np.eye(2,dtype=np.float32))
    h=boundary()
    g64,s64=math.evaluate(h,artifact,'topk',np.float64)
    g32,s32=math.evaluate(h,artifact,'topk',np.float32)
    result=audit.replay(h,artifact,'topk',g64,g32,s64,s32)
    assert result['valid'] and result['precision']['FP64']['repeat_exact']
    dead,active=math.evaluate(-np.ones_like(h),artifact,'nested',np.float64)
    assert not active.any()
    np.testing.assert_allclose(dead,np.broadcast_to(artifact['b_dec']*2,dead.shape))


def test_source_aggregation_does_not_weight_repeated_paths_as_sources():
    paths=[];values=[]
    for axis in ('amplitude','frequency'):
        for radius in (.002,.005,.01):
            for source,count,value in (('a',100,2.),('b',1,0.)):
                for _ in range(count):
                    paths.append(dict(source=source,axis=axis,radius=radius));values.append(value)
    summaries=math.aggregate(dict(score=np.asarray(values)),paths)
    assert len(summaries)==6
    assert all(r['n_sources']==2 and r['metrics']['score']['mean']==1 for r in summaries)


def test_FP32_FP64_precision_is_measured_not_native_precision_claim():
    h=boundary(1e-9)
    g64,s64=math.evaluate(h,pca(),'pca_adaptive',np.float64)
    g32,s32=math.evaluate(h,pca(),'pca_adaptive',np.float32)
    assert not np.array_equal(s64,s32)
    r64,_,_=math.measure(h,g64,s64,np.ones(1),np.array([.002]))
    r32,_,_=math.measure(h.astype(np.float32),g32,s32,np.ones(1),np.array([.002]))
    assert r64['isolated_dictionary_chord_RMS'][0]>.9
    assert r32['isolated_dictionary_chord_RMS'][0]==0
