"""Exact sign sensitivity concerns directional prevalence, not the mean."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_replication_sensitivity.py'
spec=importlib.util.spec_from_file_location('mixture_replication_sensitivity',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_exact_one_sided_signs_fixed_holm_and_opposite_mean_estimand():
    """A negative heavy-tailed mean can coexist with a majority of positive seeds."""
    names=list(audit.load_protocol()['claims'])
    contrasts={name:np.arange(1,9,dtype=float) for name in names}
    contrasts[names[-1]]=np.array([1.]*7+[-100.])
    result=audit.sign_sensitivity(contrasts)
    assert result[names[0]]['p_sign']==1/256
    assert result[names[0]]['p_sign_holm4']==4/256
    assert result[names[0]]['distribution']['maximum']==8
    heavy=result[names[-1]]
    assert heavy['fraction_positive_all']==7/8
    assert heavy['distribution']['mean']<0
    assert heavy['p_sign']==9/256


def test_ties_missing_seeds_and_incomplete_family_are_explicit():
    """Ties do not become negative signs or disappear from the all-seed fraction."""
    names=list(audit.load_protocol()['claims'])
    contrasts={name:np.array([1.,1.,-1.,-1.,0.,0.,0.,0.,np.nan]) for name in names}
    result=audit.sign_sensitivity(contrasts)
    for value in result.values():
        assert value['ties']==4
        assert value['excluded_seeds']==1
        assert value['fraction_positive_all']==.25
        assert value['fraction_positive_nonzero']==.5
        assert value['p_sign']==.6875
    with pytest.raises(ValueError,match='complete frozen'):
        audit.sign_sensitivity({names[0]:np.ones(8)})
