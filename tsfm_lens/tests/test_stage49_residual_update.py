"""Known-answer residual mediation, including a forecast-irrelevant decoy."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

path = Path(__file__).parents[1]/'experiments/feature_search/stage49_residual_update.py'
spec = importlib.util.spec_from_file_location('stage49', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_checkpoint_snapshot_parent_survives_weight_symlink(tmp_path):
    blob = tmp_path/'blobs'/'weight'; blob.parent.mkdir(); blob.write_bytes(b'weights')
    snapshot = tmp_path/'snapshot'; snapshot.mkdir(); (snapshot/'config.json').write_text('{}')
    weight = snapshot/'model.safetensors'; weight.symlink_to(blob)
    actual = m.checkpoint_path(weight)
    assert actual.parent == snapshot
    assert (actual.parent/'config.json').exists()
    assert actual.read_bytes() == b'weights'


def test_update_preserves_target_and_removes_spurious_cross_frequency():
    base = torch.zeros(1, 2, 3)
    live = base.clone(); live[..., 1] = .01
    # Channel0 is unwanted frequency, channel1 intended, channel2 loud decoy.
    y = live.clone(); y[..., 0] += 3*live[..., 1]; y[..., 2] += 100*live[..., 1]
    cache = object()
    fixed = m.replacement(live, (y, cache), base, base, 'update')
    assert torch.equal(fixed[0][..., 1], live[..., 1])
    assert torch.count_nonzero(fixed[0][..., 0]) == 0
    assert fixed[1] is cache
    ceiling = m.replacement(live, (y, cache), base, base, 'output')[0]
    assert torch.count_nonzero(ceiling[..., 1]) == 0
    # Decoy-only update is large but forecast reads only first two channels.
    decoy = live.clone(); decoy[..., 2] = 100
    repaired = m.replacement(live, (decoy, cache), base, base, 'update')[0]
    assert torch.equal(repaired[..., :2], decoy[..., :2])


def test_equal_energy_random_and_anchored_identity():
    x = torch.tensor([[[1e10, 1., -3.]]])
    y = torch.tensor([[[1., 1e-10, 7.]]])
    assert torch.equal(m.replacement(x, (y, None), x, y, 'update')[0], y)
    live = x + 1024
    native = y + 2
    update = m.replacement(live, (native, None), x, y, 'update')[0]
    random = m.replacement(live, (native, None), x, y, 'random', torch.ones_like(x))[0]
    torch.testing.assert_close(torch.linalg.vector_norm(random-native), torch.linalg.vector_norm(update-native))
    with pytest.raises(ValueError, match='SAME baseline'):
        m.replacement(live, (native, None), x[:, :, :1], y, 'update')


def test_inplace_preinput_snapshot_and_live_cache():
    class Inplace(torch.nn.Module):
        def forward(self, x):
            return x.add_(5), 'live cache'
    block = Inplace(); x = torch.ones(2, 4, 3)
    with m.block_hook(block) as state:
        block(x)
    assert torch.equal(state['inputs'][0], torch.ones_like(x))
    assert torch.equal(state['outputs'][0], torch.full_like(x, 6))
    with m.block_hook(block, (torch.ones_like(x), torch.full_like(x, 6)), 'update'):
        output = block(torch.ones_like(x))
    assert output[1] == 'live cache'
    assert torch.equal(output[0], torch.full_like(x, 6))


def test_fresh_cohort_and_fixed_phase_nuisance():
    x, future, metadata = m.corpus()
    assert x.shape == (80, 128) and future.shape == (80, 32)
    assert {r['generator_seed'] for r in metadata} == set(range(3100000, 3100016))
    expected_amplitudes = np.random.default_rng(3100000).uniform(.5,2,2)
    np.testing.assert_array_equal([metadata[0]['a'],metadata[0]['b']], expected_amplitudes)
    for start in range(0, 80, 5):
        rows = metadata[start:start+5]
        assert len({r['phi'] for r in rows}) == 1
        assert len({r['scale'] for r in rows}) == 1
        np.testing.assert_allclose(x[start:start+5].mean(axis=1), rows[0]['offset'], atol=2e-7)


def test_complete_family_and_target_flattening_cannot_advance():
    _, future, metadata = m.corpus()
    t = np.arange(128,160)
    native = future.copy()
    for row, record in enumerate(metadata):
        native[row] += record['b']*np.sin(record['psi'])*np.cos(2*np.pi*t/32)
    predictions = {'native':native}
    for site in m.SITES:
        predictions[f'update_{site}'] = future.copy()
        predictions[f'random_{site}'] = native.copy()
        predictions[f'output_{site}'] = np.repeat(future[::5],5,axis=0)
    result = m.analyze(predictions,future,metadata,{'all_pass':True})
    assert len(result['holm8']) == 8
    assert all(site['advance'] for site in result['sites'].values())
    predictions['update_9'] = predictions['output_9']
    damaged = m.analyze(predictions,future,metadata,{'all_pass':True})
    assert not damaged['sites']['9']['advance']
    assert damaged['deltas']['0.01']['update_9']['intended_error_worsening'] > .05
