import numpy as np
import torch

from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import cap_norm,encode_moves,project
from tsfm_lens.sae.models import TopKSAE


def test_crossing_keeps_true_endpoint_boolean_counts_and_distinct_decoder_writes():
    sae=TopKSAE(4,4,2)
    with torch.inference_mode():
        sae.W_enc.copy_(torch.eye(4));sae.W_dec.copy_(torch.eye(4))
        sae.b_enc.zero_();sae.b_dec.zero_()
        state=torch.tensor([[4.,1.01,1.,0.]])
        change=torch.tensor([[0.,-.02,.02,0.]])
        moves,entered,left,count=encode_moves(sae,state,change)
    np.testing.assert_array_equal([entered.item(),left.item(),count.item()],[1,1,2])
    np.testing.assert_allclose(moves["hard"],[[0.,-1.01,1.02,0.]],atol=2e-7)
    np.testing.assert_allclose(moves["locked"],[[0.,-.02,0.,0.]],atol=2e-7)
    # At clean k+1 floor=1; after the switch the floor=.99.
    np.testing.assert_allclose(moves["shoulder"],[[0.,-1.01*.01/3.,1.02*.03/3.01,0.]],atol=2e-7)


def test_projection_of_known_vector_into_dependent_decoder_span_and_physical_cap():
    d=np.array([[1.,1.,0.],[2.,2.,0.]])
    x=np.array([[3.,1.,2.],[0.,0.,.04]])
    p=project(x,d)
    np.testing.assert_allclose(p,[[2.,2.,0.],[0.,0.,0.]],atol=2e-14)
    got,raw=cap_norm(x,.2)
    np.testing.assert_allclose(raw,[np.sqrt(14.),.04])
    np.testing.assert_allclose(got[0],.2*x[0]/np.sqrt(14.))
    np.testing.assert_allclose(got[1],x[1])


def test_equal_gate_length_uses_the_actual_dense_length_and_preserves_orientation_or_zero():
    from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.matched import match_length
    v=np.array([[.3,.4],[.0,-2.],[0.,0.]])
    out,raw=match_length(v,np.array([.1,8.,.2]),.7)
    np.testing.assert_allclose(raw,[.5,2.,0.])
    np.testing.assert_allclose(out,[[.06,.08],[0.,-.7],[0.,0.]],atol=1e-14)
