"""Known-answer paired contrasts: model covariance, identity and baseline sign."""
import numpy as np
import pytest

from experiments.feature_search.natural_validation_replication_audit import paired_difference, paired_interval, hybrid_vs_seasonal


def test_source_alignment_preserves_shared_variation_and_exact_difference():
    ids=[f's{i}' for i in range(64)]
    common=100*np.arange(64.)
    order=np.random.default_rng(3).permutation(64)
    delta=paired_difference(ids,common+5,[ids[i] for i in order],(common+3)[order])
    np.testing.assert_array_equal(delta,np.full(64,2.))
    result=paired_interval(delta,family=3,n_boot=999)
    assert result['paired_mean_difference']==2 and result['paired_ci95']==[2.,2.]
    assert result['exploratory_adjusted_interval']==[2.,2.]


def test_mismatched_or_duplicate_sources_are_rejected():
    with pytest.raises(ValueError,match='unique source'):
        paired_difference(['a','b'],[1,2],['a','c'],[1,2])
    with pytest.raises(ValueError,match='unique source'):
        paired_difference(['a','a'],[1,2],['a','a'],[1,2])


def test_seasonal_comparison_sign_and_exploratory_interval_family():
    result=hybrid_vs_seasonal(np.full(64,3.),np.ones(64),n_boot=999)
    assert result['paired_mean_difference']==-2 and result['paired_ci95']==[-2.,-2.]
    assert result['negative_sources']==64 and result['exploratory_bonferroni_family']==6
    x=np.arange(64.)**2
    wide=paired_interval(x,family=6,n_boot=1999)
    assert wide['exploratory_adjusted_interval'][0] < wide['paired_ci95'][0]
    assert wide['exploratory_adjusted_interval'][1] > wide['paired_ci95'][1]
