import numpy as np

from research.activation_transfer.atoms.third.analyze import score
from research.activation_transfer.atoms.scores import resampling_counts


def test_atom_specificity_uses_source_outcome_and_rewards_correct_amplitude():
    rng=np.random.default_rng(188)
    source=rng.normal(size=(45,3,8)); m=np.ones((45,3),bool)
    preds={'dense':source*2, 'coactive':source*0, **{'wrong'+str(i):-source for i in range(4)}}
    r=score(m,preds,source,resampling_counts(45,128))
    assert np.isclose(r['methods']['dense']['skill']['value'],0.)
    assert np.isclose(r['dense_minus_wrong']['cosine']['value'],2.)
    assert np.isclose(r['methods']['coactive']['cosine']['value'],0.)


def test_silent_source_has_no_scored_atom():
    source=np.zeros((16,2,8)); m=np.zeros((16,2),bool)
    r=score(m,{},source,resampling_counts(16,12))
    assert r['measurable']==[False,False] and r['dense_minus_wrong'] is None
