import numpy as np
import torch
from torch import nn

from research.activation_transfer.shape_transfer.depth.paired_sites import forecast


def test_later_live_patch_preserves_earlier_intervention():
    class Adapter:
        def __init__(self):
            self.module=nn.Sequential(nn.Identity(),nn.Linear(1,1,bias=False))
            with torch.no_grad():
                self.module[1].weight.fill_(3.)
        def predict(self,x,horizon,levels):
            with torch.no_grad():
                values=self.module(torch.from_numpy(x))[:,1:]
            return {"quantiles":values.numpy()}
    x=np.array([[[11.],[2.]]],dtype="f4")
    values=forecast(Adapter(),["0","1"],lambda n:slice(n-1,n),x,[np.array([[5.]]),np.array([[-4.]])])
    np.testing.assert_array_equal(values,np.array([[17.]]))
