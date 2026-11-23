import numpy as np

from research.activation_transfer.atoms.scores import aggregate, resampling_counts
from research.activation_transfer.atoms.vectors import approximate, bridge, choose_atoms, coactive


def test_clean_bridge_cannot_identify_an_unobserved_axis():
    rng=np.random.default_rng(55)
    x=np.c_[rng.normal(size=(128,4)),np.zeros(128)]
    q,_=np.linalg.qr(rng.normal(size=(5,5)))
    directions=np.eye(5)[[0,4]]
    moved,_,projected=bridge(x,x@q+4,directions,201)
    np.testing.assert_allclose(moved,projected@q,atol=1e-10)
    np.testing.assert_allclose(moved[1],np.zeros(5),atol=1e-10)
    assert np.linalg.norm(projected[0])>.95


def test_dictionary_projection_preserves_signed_single_atom():
    alphabet=np.eye(20)
    result=approximate(-alphabet[7]*2.5,alphabet,np.arange(20))
    for k,(value,ids) in result.items():
        assert len(ids)==k
        np.testing.assert_allclose(value,-alphabet[7]*2.5)


def test_native_feature_permutation_and_screen_are_outcome_free():
    rng=np.random.default_rng(11)
    z=rng.normal(size=(200,32))
    z[z<1.0]=0
    ids=choose_atoms(z,16,5)
    perm=rng.permutation(z.shape[1])
    matched,correlation=coactive(z[:,ids],z[:,perm],np.arange(z.shape[1]))
    np.testing.assert_array_equal(perm[matched],ids)
    np.testing.assert_allclose(correlation,1)


def test_skill_and_cosine_agree_only_at_correct_dose():
    rng=np.random.default_rng(4)
    r=rng.normal(size=(24,5,8))
    active=np.ones((24,5),bool)
    active[:3,2]=False
    vals,full,boot=aggregate(active,3*r,r,resampling_counts(24,64))
    np.testing.assert_allclose(vals["cosine"],1)
    np.testing.assert_allclose(full["skill"],-3)
    np.testing.assert_allclose(boot["gain"],3)


def test_empty_natural_removal_has_zero_waveform_even_with_batch_replay_noise():
    from research.activation_transfer.atoms.intervene import exact_removal_effects
    effects=np.full((2,2,8),1e-7)
    codes=np.array([[0,2],[0,1]])
    out=exact_removal_effects(effects,codes)
    np.testing.assert_array_equal(out[:,0],0)
    np.testing.assert_array_equal(out[:,1],effects[:,1])
