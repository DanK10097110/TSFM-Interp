import numpy as np
import torch
from torch import nn

from research.activation_transfer.shape_transfer.depth.geometry import extract


class Chain:
    def __init__(self):
        self.module=nn.Sequential(nn.Linear(2,2,bias=False),nn.Linear(2,2,bias=False))
        with torch.no_grad():
            self.module[0].weight.copy_(torch.eye(2))
            self.module[1].weight.copy_(torch.tensor([[3.,0.],[0.,-2.]]))

    def token_slice(self,length):
        return slice(0,length)

    def predict_differentiable(self,x,horizon):
        h=self.module(torch.from_numpy(x))
        return h[:,0,0,None].expand(-1,horizon)


def test_simultaneously_injected_zero_leaves_preserve_both_chain_derivatives():
    x=np.ones((2,2,2),dtype="f4")
    _,j,y=extract(Chain(),["0","1"],np.array([0]),x,True)
    np.testing.assert_array_equal(y,3*np.ones_like(y))
    np.testing.assert_array_equal(j["0"][:,:,0],3*np.ones_like(y))
    np.testing.assert_array_equal(j["1"][:,:,0],np.ones_like(y))
    np.testing.assert_array_equal(j["0"][:,:,1],np.zeros_like(y))
