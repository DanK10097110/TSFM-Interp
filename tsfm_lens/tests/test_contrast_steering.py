import numpy as np
import pytest
import torch
from tsfm_lens.steering import ActivationContrast, SteeringModel, SteeringUnavailable


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.eye(2))

    def forward(self, x):
        state = self.hidden(x.reshape(-1, 2, 2))
        return state[:, -1] @ self.weight


def test_frozen_contrast_separates_bias_and_sensitivity(tmp_path):
    net = Probe().eval()
    model = SteeringModel(lambda x,h: net(torch.from_numpy(x)).detach().numpy(), module=net)
    pos = np.array([[0,1,3,1],[1,0,3,-1]])
    neg = np.array([[0,1,-1,1],[1,0,-1,-1]])
    contrast = ActivationContrast.fit(model, pos, neg, layer="hidden", prediction_length=2)
    x = np.array([[1,2,2,5],[3,4,0,7],[1,2,1,8]])
    np.testing.assert_allclose(contrast.score(x), [.25, -.25, 0])
    base = model.forecast(x, 2)
    for mode, delta in [("add", [4,4,4]), ("amplify", [1,-1,0])]:
        calls = model.calls
        actual = contrast.predict(x, mode=mode, strength=1)
        assert model.calls - calls == len(x)
        np.testing.assert_allclose(actual[:,:,0] - base[:,:,0], np.column_stack([delta,np.zeros(3)]))
        np.testing.assert_allclose(contrast.predict(x, mode=mode, strength=0), base)
    path = tmp_path / "contrast.npz"
    contrast.save(path)
    np.testing.assert_array_equal(ActivationContrast.load(path, model).predict(x), contrast.predict(x))
    with torch.no_grad():
        net.weight.add_(1)
    with pytest.raises(SteeringUnavailable, match="checkpoint"):
        ActivationContrast.load(path, model)
    assert not net.hidden._forward_pre_hooks


def test_rejects_zero_contrast_and_cleans_up_hook():
    net = Probe().eval()
    model = SteeringModel(lambda x,h: net(torch.from_numpy(x)).detach().numpy(), module=net)
    same = np.ones((2,4))
    with pytest.raises(SteeringUnavailable, match="no measurable"):
        ActivationContrast.fit(model, same, same, layer="hidden", prediction_length=2)
    assert not net.hidden._forward_pre_hooks
