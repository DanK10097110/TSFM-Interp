import torch
from tsfm_lens.sae.models import TopKSAE

from research.activation_transfer.shape_transfer.dictionary_prior.refit.encoding import endpoint


def test_passing_a_topone_boundary_changes_the_decoder_displacement_discontinuously():
    sae=TopKSAE(2,2,1)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2))
        sae.W_dec.copy_(torch.eye(2))
        x=torch.tensor([[1.,.99]])
        delta=torch.tensor([[-.02,.02]])
        hard,locked,enter=endpoint(sae,x,delta,1.)
        torch.testing.assert_close(hard,torch.tensor([[-1.,1.01]]))
        torch.testing.assert_close(locked,torch.tensor([[-.02,0.]]))
        assert enter.item()==1
        small,fixed,enter=endpoint(sae,x,delta,.125)
        assert enter.item()==0
        torch.testing.assert_close(small,fixed)
