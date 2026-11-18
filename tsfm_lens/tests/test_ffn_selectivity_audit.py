"""Known-answer checks and tampering decoys for the independent FFN audit."""
from copy import deepcopy
import numpy as np
import pytest

from experiments.feature_search import ffn_selectivity_analysis as primary
from experiments.feature_search import ffn_selectivity_audit as audit


def fixture_record(x,fit):
    """Record whose one-feature amplitude code classifies without type purity."""
    rule=primary.fit_readout(x,fit,1)
    result=primary.evaluate(x,rule)
    return dict(budget=1,FIT_rule=rule,per_source={key:value.tolist() for key,value in result.items()})


def test_audit_detects_score_and_feature_selection_tampering():
    x=np.tile(np.arange(1,6.)[None,:,None],(6,1,2))
    x[:,:,1]=1
    fit=np.array([True]*3+[False]*3)
    record=fixture_record(x,fit)
    assert record['per_source']['accuracy']==[1.]*6
    np.testing.assert_allclose(record['per_source']['purity'],1/3)
    assert audit.verify_readout(x,fit,record)<1e-14
    wrong=deepcopy(record)
    wrong['per_source']['purity']=[1.]*6
    with pytest.raises(ValueError,match='metric mismatch'):
        audit.verify_readout(x,fit,wrong)
    wrong=deepcopy(record)
    wrong['FIT_rule']['indices']=[1]
    with pytest.raises(ValueError,match='selection mismatch'):
        audit.verify_readout(x,fit,wrong)


def test_audit_pca_ignores_TEST_and_balances_duplicate_variants():
    rng=np.random.default_rng(51)
    x=rng.normal(size=(6,10,2,5))
    fit=np.array([True]*3+[False]*3)
    conditions=[dict(kind=k) for k in audit.TYPES for _ in range(2)]
    result=audit.project(x,'PCA',fit,conditions)
    poisoned=x.copy()
    poisoned[~fit]*=1e9
    other=audit.project(poisoned,'PCA',fit,conditions)
    np.testing.assert_array_equal(result[fit],other[fit])
    duplicate=audit.project(np.concatenate([x,x[:,:2]],axis=1),'PCA',fit,conditions+conditions[:2])
    np.testing.assert_allclose(abs(result),abs(duplicate[:,:10]),atol=1e-12)


def test_audit_pooling_distinguishes_canceling_tokens_and_rotation_geometry():
    conditions=[dict(kind=k) for k in audit.TYPES for _ in range(2)]
    x=np.tile(np.array([1.,-1.])[None,None,:,None],(4,10,1,8))
    np.testing.assert_array_equal(audit.features(x,conditions,'mean'),np.zeros((4,5,8)))
    np.testing.assert_array_equal(audit.features(x,conditions,'rms'),np.ones((4,5,8)))
    transformed=audit.project(x,'rotation4961101',np.array([1,1,0,0],bool),conditions)
    np.testing.assert_allclose(np.sum(transformed**2,axis=-1),np.sum(x*x,axis=-1),atol=1e-12)


def test_audit_bootstrap_rejects_source_and_CI_tampering():
    rows=[dict(source=str(i),kind='synthetic',split='FIT' if i<2 else 'TEST') for i in range(4)]
    actual=audit.summary(np.array([1e8,-1e8,1.,3.]),rows,'synthetic')
    assert actual==dict(n_sources=2,source_ids=['2','3'],mean=2.,ci95=[1.,3.])
    audit.check_summary(actual,actual)
    wrong=deepcopy(actual)
    wrong['source_ids']=['0','1']
    with pytest.raises(ValueError,match='source identity'):
        audit.check_summary(actual,wrong)
    wrong=deepcopy(actual)
    wrong['ci95']=[0.,4.]
    with pytest.raises(ValueError,match='bootstrap mismatch'):
        audit.check_summary(actual,wrong)
