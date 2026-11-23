import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import project


def test_active_span_recovers_signed_vector_split_across_two_nonorthogonal_atoms():
    d=np.array([[1.,0.,0.],[1.,1.,0.]])
    v=np.array([.2,-1.,3.])
    best,single,span=project(v,d)
    assert best==1
    np.testing.assert_allclose(single,[-.4,-.4,0.])
    np.testing.assert_allclose(span,[.2,-1.,0.],atol=1e-12)
    np.testing.assert_allclose(d@(v-span),0.,atol=1e-12)
