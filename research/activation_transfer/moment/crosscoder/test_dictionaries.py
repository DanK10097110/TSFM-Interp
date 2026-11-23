import torch

from research.activation_transfer.moment.crosscoder.dictionaries import encode_views
from tsfm_lens.sae.crosscoder import CrosscoderSAE


def test_source_only_omits_the_entire_missing_encoder_contribution():
    model = CrosscoderSAE([2,2],3,1,source_scale=[2.,5.])
    with torch.no_grad():
        model.W_enc[0].copy_(torch.tensor([[1.,0.,0.],[0.,0.,0.]]))
        model.W_enc[1].copy_(torch.tensor([[0.,1.,0.],[0.,0.,1.]]))
        model.b_enc.zero_()
        model.b_dec[0].zero_()
        model.b_dec[1].copy_(torch.tensor([-6.,0.]))
    means = [torch.tensor([3.,4.]),torch.tensor([8.,4.])]
    small = torch.tensor([[5.,4.]])
    # A large input equal to its train mean still contributes the learned large decoder bias.
    torch.testing.assert_close(encode_views(model,means,small),torch.tensor([[1.,0.,0.]]))
    torch.testing.assert_close(encode_views(model,means,small,means[1][None]),torch.tensor([[0.,6.,0.]]))
