import numpy as np
import torch
from tsfm_lens.sae.crosscoder import CrosscoderSAE
from research.activation_transfer.moment.crosscoder.multilayer.train import minibatch,plan,code


def test_vertical_derangement_preserves_pairing_within_depth_and_physical_position():
    grid=(100*np.arange(768)[:,None]+np.arange(60)).astype('f4')[:,:,None]
    rows=np.array([0,14,62,767]);pos=np.array([0,37,59,8])
    perm,mask=plan('shuffled_depth',22112401)
    assert np.all(perm!=np.arange(768))
    np.testing.assert_array_equal(np.sort(perm),np.arange(768))
    assert mask.sum()==500
    taps=minibatch([grid+i for i in (0,1,2,3)],rows,pos,perm)
    np.testing.assert_equal(taps[0].numpy(),grid[rows,pos])
    np.testing.assert_equal(taps[1].numpy(),grid[rows,pos]+1)
    np.testing.assert_equal(taps[2].numpy(),grid[perm[rows],pos]+2)
    np.testing.assert_equal(taps[3].numpy(),grid[perm[rows],pos]+3)


def test_source_pass_suppresses_other_three_inputs_but_full_pass_observes_them():
    m=CrosscoderSAE([2]*4,2,1,source_scale=[1.]*4)
    with torch.no_grad():
        for w in m.W_enc:w.copy_(torch.eye(2))
    means=[torch.zeros(2)]*4
    source=torch.tensor([[5.,0.]])
    secondary=torch.tensor([[0.,6.]])
    inputs=[source,secondary,secondary,secondary]
    np.testing.assert_equal(code(m,means,inputs,True).detach().numpy(),[[5.,0.]])
    np.testing.assert_equal(code(m,means,inputs,False).detach().numpy(),[[0.,18.]])
