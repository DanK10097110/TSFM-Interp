import numpy as np
import torch

from research.activation_transfer.steering.analyze import agreement, measure
from research.activation_transfer.steering.study import cues, learn_directions, patch_effect, sites_and_states
from tsfm_lens.steering import SteeringModel


def test_known_rotation_recovers_the_true_map():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(192,12))
    q, _ = np.linalg.qr(rng.normal(size=(12,12)))
    y = x @ q + 2.4
    cue = rng.normal(size=(2,12))
    mapping, shams = learn_directions(x,y,cue,[rng.permutation(len(x)) for _ in range(8)],1e-9)
    np.testing.assert_allclose(mapping,cue@q,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(shams,axis=2),np.linalg.norm(mapping,axis=1)[:,None]*np.ones((2,8)))
    assert np.median([agreement(s[None],(cue@q)[0:1])["cosine"] for s in shams[0]]) < 0.5


def test_native_match_can_diverge_from_source():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(16,8))
    shams = rng.normal(size=(16,8,8))
    arms = np.concatenate([reference[:,None],reference[:,None],shams],axis=1)
    result,_ = measure(arms,-reference,rng.integers(16,size=(100,16)))
    assert result["target_skill"]["value"] == 1
    assert result["source_skill"]["value"] == -3
    assert result["target_native"]["gain"] > 0.999
    assert result["source_native"]["gain"] < -0.999
    assert np.max(np.abs(cues()[:,-1])) < 1e-8


def test_steering_batch_adds_a_fixed_vector_to_last_token_only():
    class Simple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Identity()

        def forward(self,x,h):
            hiddens = self.block(torch.as_tensor(x).reshape(-1,2,8))
            return (hiddens[:,0,:] + 2*hiddens[:,1,:])[:,:h]
    module = Simple().eval()
    model = SteeringModel(module,module=module,quantiles=[0.5])
    x = np.arange(48,dtype="f4").reshape(3,16)
    sites,states=sites_and_states(model,["block"],x,2)
    directions = np.eye(8)[:2]
    delta = patch_effect(model,sites["block"],x,np.linalg.norm(states["block"],axis=1),directions,4)
    np.testing.assert_allclose(delta,np.broadcast_to(2*directions,(3,2,8)),atol=1e-5)
