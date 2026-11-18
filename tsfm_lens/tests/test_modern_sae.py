"""Signed polarity and batch-independent inference are the new SAE contracts."""

import numpy as np
import pytest
import torch

from tsfm_lens.sae.modern import AbsTopKSAE, BatchTopKSAE, train_modern_sae, load_steering_dictionary
from tsfm_lens.sae.train import SAETrainConfig


def test_signed_codes_preserve_negative_polarity_and_gradient():
    model=AbsTopKSAE(3,4,2)
    pre=torch.tensor([[-5.,4.,3.,-2.]],requires_grad=True)
    code=model.sparsify(pre)
    assert torch.equal(code,torch.tensor([[-5.,4.,0.,0.]]))
    code.sum().backward()
    assert torch.equal(pre.grad,torch.tensor([[1.,1.,0.,0.]]))


def test_batch_budget_adapts_activity_then_inference_is_batch_independent():
    model=BatchTopKSAE(2,4,1)
    pre=torch.tensor([[1.,2.,3.,4.],[.1,.2,.3,.4]])
    code=model.sparsify(pre)
    assert torch.equal((code>0).sum(1),torch.tensor([2,0]))
    model.eval()
    with pytest.raises(RuntimeError):
        model.sparsify(pre)
    model.threshold.fill_(.25)
    together=model.sparsify(pre)
    alone=torch.cat([model.sparsify(row[None]) for row in pre])
    assert torch.equal(together,alone)


@pytest.mark.parametrize('kind',['topk','abstopk','batchtopk'])
def test_training_is_seeded_finite_and_checkpoint_restores_threshold(kind,tmp_path):
    x=np.random.default_rng(4).normal(size=(24,6)).astype('float32')
    cfg=SAETrainConfig(dict_size=12,k=2,min_train_steps=4,epochs=1,batch_size=12,seed=7)
    first,history=train_modern_sae(x,cfg,kind=kind)
    second,_=train_modern_sae(x,cfg,kind=kind)
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
    assert np.isfinite(history).all()
    assert torch.isfinite(first(torch.from_numpy(x))[0]).all()
    path=tmp_path/'sae.pt'
    torch.save(dict(kind=kind,config=vars(cfg),state_dict=first.state_dict()),path)
    restored=load_steering_dictionary(path)
    assert torch.equal(restored.encode(torch.from_numpy(x)),first.encode(torch.from_numpy(x)))
