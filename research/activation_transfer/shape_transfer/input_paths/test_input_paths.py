import numpy as np
import torch
from torch import nn

from research.activation_transfer.shape_transfer.input_paths.run import forward
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite


def test_last_token_swap_separates_other_tokens_and_bypass():
    layer = nn.Sequential(nn.Identity())
    def forecast(x, horizon):
        tensor = torch.as_tensor(x)
        hidden = layer(tensor[:, :, None])[:, :, 0]
        return (2*hidden[:, :1] + 3*hidden[:, 1:] + 5*tensor[:, :1]).repeat(1, horizon)
    model = SteeringModel(forecast, module=layer)
    site = InterventionSite("0", ["args", 0], 0, [2, 1], "last")
    x = np.array([[1., 4.], [-2., -1.]], dtype="f4")
    y = x + np.array([[2., -1.]], dtype="f4")
    baseline, h0 = forward(model, site, x)
    changed, h1 = forward(model, site, y)
    np.testing.assert_allclose(changed - baseline, 11)
    np.testing.assert_allclose(forward(model, site, y, h0, "all")[0] - baseline, 10)
    np.testing.assert_allclose(forward(model, site, x, h1, "all")[0] - baseline, 1)
    np.testing.assert_allclose(forward(model, site, x, h1, "last")[0] - baseline, -3)
    np.testing.assert_allclose(forward(model, site, y, h0, "last")[0] - baseline, 14)
