import torch
from timesfm.torch.normalization import RMSNorm

from research.activation_transfer.shape_transfer.depth.feedback.last_block import tangent


def test_norm_radial_null_and_orthogonal_survival_match_finite_differences():
    x = torch.tensor([[1., -2., 3.]],dtype=torch.float64)
    dx = torch.tensor([[3., .5, -2.]],dtype=torch.float64)
    norm = RMSNorm(3).double()
    with torch.no_grad():
        norm.scale.copy_(torch.tensor([.5, -2., 1.]))
    for mod in (norm,torch.nn.ReLU(),torch.nn.SiLU()):
        baseline = mod(x)
        for delta in (x, dx):
            eps = 1.e-5
            measured = (mod(x+eps*delta)-mod(x-eps*delta))/(2*eps)
            applied = tangent(mod,x,baseline,x+delta) - baseline
            torch.testing.assert_close(applied,measured,rtol=1.e-6,atol=1.e-9)
