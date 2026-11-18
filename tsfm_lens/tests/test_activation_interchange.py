"""Train-basis and actual patch-path assertions for natural donor interchange."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

path = Path(__file__).resolve().parents[1]/'experiments/feature_search/activation_interchange.py'
spec = importlib.util.spec_from_file_location('interchange', path)
interchange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interchange)


def test_response_projection_and_energy_matched_control():
    """A rank-one planted donor change is captured; equal-energy null stays equal."""
    rng = np.random.default_rng(8)
    receiver = rng.normal(size=(12, 3, 8)).astype(np.float32)
    donor = receiver.copy()
    donor[..., 2] += 3
    response, _, random = interchange.response_bases(receiver, donor, max_rank=4)
    learned, energy = interchange.projected_replacement(receiver, donor, response[:, :1])
    np.testing.assert_allclose(learned, donor, atol=1e-6)
    control, control_energy = interchange.projected_replacement(receiver, donor, random[0][:,:1], target_energy=energy)
    np.testing.assert_allclose(control_energy, energy, atol=1e-6)
    assert not np.allclose(control, donor)
    assert len(random) == 16


def test_actual_predict_patch_preserves_special_tokens_and_reaches_output():
    """The same predictor sees a context patch and keeps its trailing readout."""
    class Fake:
        device = torch.device('cpu')
        module = torch.nn.Sequential(torch.nn.Identity())
        def token_slice(self, length):
            return slice(0, 2)
        def predict(self, contexts, horizon, quantiles):
            hidden = torch.tensor(np.concatenate((contexts, np.ones((len(contexts),1,1))), axis=1), dtype=torch.float32)
            values = self.module(hidden)
            return {'point': values.sum(dim=1).numpy().repeat(horizon, axis=1)}
    adapter = Fake()
    contexts = np.zeros((3,2,1), dtype=np.float32)
    base = interchange.patched_predict(adapter, contexts)
    same = interchange.patched_predict(adapter, contexts, '0', contexts)
    donor = interchange.patched_predict(adapter, contexts, '0', np.ones_like(contexts))
    np.testing.assert_array_equal(base, same)
    np.testing.assert_array_equal(donor, np.full((3,32), 3))
