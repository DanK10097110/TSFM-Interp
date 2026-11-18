"""Fresh native-head replication controls for isolation,energy units and exact majority tests."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest


def _module():
    """Load the separate native-only replication,without constructing a model."""
    root=Path(__file__).parents[1]/'experiments/feature_search';sys.path.insert(0,str(root));s=importlib.util.spec_from_file_location('native_head_replication',root/'native_head_replication.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_fresh_seed_isolation_and_frozen_conditions():
    """The earlier1659000cohort is a deliberate forbidden seed decoy."""
    m=_module();x,y,metadata=m.make_fresh_corpus();repeat,_,_=m.make_fresh_corpus()
    assert x.shape==(640,128) and y.shape==(640,32)
    assert {r['generator_seed'] for r in metadata}==set(range(2225000,2225128))
    assert not ({r['generator_seed'] for r in metadata}&set(range(1659000,1659128)))
    assert np.array_equal(x,repeat)
    assert all(c[1] in [None,'psi'] for c in m.CONDITIONS)


def test_ratio_uses_shared_aggregate_analytic_strength():
    """A tiny seed denominator is a decoy for averaging unstable individual ratios."""
    m=_module();a=np.array([1,2,3,4.]);b=np.full(4,.5);strength=np.array([.001,1,10,100])
    effect=m.paired_numerator_effect(a,b,strength)
    assert abs(effect['value']-np.sum(a-b)/np.sum(strength))<1e-12
    assert effect['value']<.1


def test_exact_sign_test_is_two_sided_conditional_majority_not_mean():
    """A huge favorable outlier makes the mean positive while most paired seeds disagree."""
    m=_module();negative_majority=m.exact_sign_counts(np.array([100.,-1,-1,-1,-1,-1,-1]))
    assert negative_majority['positive']==1 and negative_majority['negative']==6
    assert not negative_majority['majority_positive']
    positive=m.exact_sign_counts(np.ones(4));assert positive['p']==.125
    ties=m.exact_sign_counts(np.array([3.,0,-2,0,4]))
    assert ties['positive']==2 and ties['negative']==1 and ties['ties']==2 and ties['n_non_ties']==3
    assert ties['q']==2/3 and ties['q_cp95'][0]<ties['q']<ties['q_cp95'][1]


def test_complete_global_holm_family_and_practical_gate():
    """Two individually significant tests are a decoy for their complete family correction."""
    m=_module();effects={key:dict(value=value,lo=.01,sign_test=dict(p=p,majority_positive=True)) for key,value,p in [('timesfm_minus_chronos2',.12,.03),('bolt_minus_chronos2',.03,.04)]}
    m.finalize_comparisons(effects,dict(timesfm=True,bolt=True,chronos2=True))
    assert effects['timesfm_minus_chronos2']['p_holm']==.06
    assert effects['bolt_minus_chronos2']['p_holm']==.06
    assert not any(e['passed'] for e in effects.values())
    with pytest.raises(ValueError,match='complete'):m.finalize_comparisons({'timesfm_minus_chronos2':effects['timesfm_minus_chronos2']},dict(timesfm=True,chronos2=True))
