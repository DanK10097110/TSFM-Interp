"""Planted-answer controls for the bounded dictionary pilot; no native loading."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.feature_search import dictionary_core as core
from experiments.feature_search import dictionary_native as native
from experiments.feature_search import dictionary_pilot as pilot


def test_split_source_leakage():
    rows = [dict(source='a', split='fit'), dict(source='b', split='test')]
    fit, test = core.split_indices(rows)
    assert fit.tolist() == [0] and test.tolist() == [1]
    with pytest.raises(ValueError, match='Source leakage'):
        core.split_indices(rows+[dict(source='a', split='test')])


def test_source_weighted_train_scaler():
    x = np.array([[0., 2.], [0., 2.], [6., 8.], [1e9, 1e9]])
    weights = core.source_weights(['a', 'a', 'b'])
    np.testing.assert_allclose(weights, [.25, .25, .5])
    center, scale = core.fit_scaler(x[:3], weights)
    np.testing.assert_allclose(center, [3., 5.])
    assert scale == 3.
    x[-1] *= -100
    again = core.fit_scaler(x[:3], weights)
    np.testing.assert_array_equal(center, again[0])
    assert scale == again[1]


def test_signed_and_actual_active_budget():
    x = np.array([[-9., 3., 2.], [0., -8., 4.], [0., 0., 0.]])
    z = core.signed_topk(x, np.array([1, 2, 0]))
    np.testing.assert_array_equal(z, [[-9., 0., 0.], [0., -8., 4.], [0., 0., 0.]])
    report = core.health(x, z, z, np.ones(3)/3)
    assert report['mean_l0'] == 1. and report['dead_fraction'] == 0.


def test_single_feature_native_rotation_selectivity():
    rng = np.random.default_rng(55)
    x = rng.normal(size=(1200, 16))
    weights = np.ones(600)/600
    basis = core.fit_basis(x[:600], weights, 'rotation', 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(16), atol=3e-7)
    np.testing.assert_allclose((x @ basis) @ basis.T, x, atol=1e-6)
    target = x[:, 0]
    pred_native, selected = core.sparse_probe(x[:600], target[:600], weights, x[600:], 1)
    rotated = x @ basis
    pred_rotated, _ = core.sparse_probe(rotated[:600], target[:600], weights, rotated[600:], 1)
    assert selected == [0]
    assert np.mean((pred_native-target[600:])**2) < .001
    assert np.mean((pred_rotated-target[600:])**2) > .2


def test_pca_uses_weighted_fit_covariance():
    x = np.array([[5., 0.], [-5., 0.], [0., 4.]])
    b = core.fit_basis(x, np.array([.49, .49, .02]), 'pca', 1)
    assert abs(b[0, 0]) > .999
    swapped = core.fit_basis(x, np.array([.01, .01, .98]), 'pca', 1)
    assert abs(swapped[1, 0]) > .999


def test_nested_shared_code_and_tiny_budget():
    model = core.NestedTopK(8, 16, 4, generator=torch.Generator().manual_seed(3))
    x = torch.arange(48, dtype=torch.float32).reshape(6, 8)/20
    z = model.encode(x)
    small = ((z[:, :8] @ model.W_dec[:8]+model.b_dec-x)**2).mean()
    full = ((model.decode(z)-x)**2).mean()
    torch.testing.assert_close(model.reconstruction_loss(x), (small+full)/2)
    assert abs(float((small-full).detach())) > .01
    model.reconstruction_loss(x).backward()
    assert torch.isfinite(model.W_dec.grad).all()
    a = x.numpy()
    weights = np.ones(len(a))/len(a)
    for method in ('topk', 'nested'):
        params, budget = core.train_sae(a, weights, method, 4, 4, 16, steps=2, batch=4)
        assert budget['sampled_tokens'] == 8
        assert budget['trainable_parameters'] == 280
        again, _ = core.train_sae(a, weights, method, 4, 4, 16, steps=2, batch=4)
        for key in params:
            np.testing.assert_array_equal(params[key], again[key])


def test_frozen_inference_batch_independence():
    artifact = dict(W_enc=np.eye(3), W_dec=np.eye(3), b_enc=np.zeros(3), b_dec=np.zeros(3), k=np.array(1))
    x = np.array([[1., 2., 3.], [4., 3., 2.]])
    single = core.encode(x[:1], artifact)
    np.testing.assert_array_equal(single, [[0., 0., 3.]])
    np.testing.assert_array_equal(single, core.encode(x, artifact)[:1])


def test_stability_one_to_one_and_dead_exclusion():
    a = np.array([[1., 0.], [1., 0.], [0., 1.]])
    b = np.eye(2)
    result = core.stability(a, b, np.array([True, True, False]), np.array([True, True]))
    assert result['matched'] == 2
    assert result['mean_abs_cosine'] == .5


def test_ffn_hook_is_projection_input_and_replays():
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.data[:] = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    x = torch.tensor([[1., 2., 3.]])
    states = []
    with native.expansion_hook(module, states):
        baseline = module(x).detach().numpy()
    assert states[0].shape == (1, 3)
    with native.expansion_hook(module, [], replacement=states[0]):
        np.testing.assert_array_equal(baseline, module(x).detach().numpy())
    with native.expansion_hook(module, [], replacement=np.zeros((1, 3))):
        np.testing.assert_array_equal(module(x).detach().numpy(), np.zeros((1, 2)))
    with pytest.raises(ValueError, match='geometry'):
        with native.expansion_hook(module, [], replacement=np.zeros((1, 2))):
            module(x)
    assert not module._forward_pre_hooks
    timesfm = SimpleNamespace(name='timesfm', model=SimpleNamespace(stacked_xf=[SimpleNamespace(ff1=module)]))
    assert native.ffn_projection(timesfm, '0')[0] is module
    ff = SimpleNamespace(wo=module)
    bolt = SimpleNamespace(name='bolt', model=SimpleNamespace(encoder=SimpleNamespace(block=[SimpleNamespace(layer=[SimpleNamespace(DenseReluDense=ff)])])))
    assert native.ffn_projection(bolt, '0')[0] is module


def test_exact_native_gate():
    native.exact_gate(np.array([1.]), np.array([1.]), 'self')
    with pytest.raises(ValueError, match='Exact'):
        native.exact_gate(np.array([1.]), np.array([1.+1e-12]), 'self')


def test_source_summary_not_row_count_and_receiver_selection():
    rows = [dict(source='a', domain='d', split='test'), dict(source='a', domain='d', split='test'),
            dict(source='b', domain='d', split='test'), dict(source='fit', domain='d', split='fit')]
    summary = core.source_summary([0., 0., 9.], rows[:3])['d']
    assert summary['n_sources'] == 2 and summary['mean'] == 4.5
    assert native.receiver_indices(rows) == [0, 2]


def test_slot_gate_precedes_launch():
    with pytest.raises(ValueError, match='No slot'):
        pilot.launch(SimpleNamespace(slot_granted=False))


def test_resource_monitor_exit_race():
    assert pilot.rss_bytes(['Name: worker', 'VmRSS:\t1234 kB']) == 1234*1024
    assert pilot.rss_bytes(['Name: worker', 'State:\tZ (zombie)']) == 0


def test_normalizer_metadata_and_normalized_targets():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(12, 128))+np.repeat([0., 3., -1., 7.], 32)
    for model, width in [('bolt', 2), ('timesfm', 8)]:
        meta, normalized = pilot.normalizer_metadata(x, model)
        assert meta.shape == (12, width)
        np.testing.assert_allclose(meta[:, width//2-1], x.mean(1))
        np.testing.assert_allclose(meta[:, -1], x.std(1))
        y, names, _, _, _ = pilot.concept_targets(x, model)
        changed, _, _, _, _ = pilot.concept_targets(3*x+9, model)
        np.testing.assert_allclose(y[:, :6], changed[:, :6], atol=1e-12)
        np.testing.assert_allclose(y[:, 12:18], changed[:, 12:18], atol=1e-12)
        np.testing.assert_allclose(y[:, 6:12]*3, changed[:, 6:12], atol=1e-12)
        assert 'model_normalized' in names[12]
        if model == 'timesfm':
            np.testing.assert_allclose(normalized[:, :32].mean(1), 0., atol=1e-12)
            assert np.mean(abs(normalized[:, -32:].mean(1))) > .3


def test_metadata_conditioned_feature_increment():
    rng = np.random.default_rng(44)
    metadata = rng.normal(size=(1000, 1))
    signal = rng.normal(size=1000)
    features = np.column_stack([metadata[:, 0], signal, rng.normal(size=1000)])
    target = 5*metadata[:, 0]+signal
    prediction, baseline, selected = core.conditional_probe(features[:500], target[:500], np.ones(500)/500,
        features[500:], 1, metadata[:500], metadata[500:])
    assert selected == [1]
    assert np.mean((prediction-target[500:])**2) < .01
    assert np.mean((baseline-target[500:])**2) > .5


def test_tiny_stage_test_states_cannot_change_fitted_dictionary():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    rng = np.random.default_rng(45)
    rows = [dict(source=f'{s}{i}', domain='synthetic', split=s) for s in ('fit', 'test') for i in range(4)]
    h = rng.normal(size=(8, 1, 3, 16)).astype(np.float32)
    history = rng.normal(size=(8, 128)).astype(np.float32)
    with TemporaryDirectory(prefix='dictionary-test-') as temporary:
        root = Path(temporary)
        pilot.write(root/'rows.json', rows)
        pilot.write(root/'bolt_worker.json', {'cuts': ['0']})
        pilot.write(root/'protocol.json', {})
        np.savez(root/'inputs.npz', history=history)
        results = []
        for repetition in range(2):
            arrays = {f'{i}_full': value for i, value in enumerate(h)}
            for i, value in enumerate(h):
                for direction in ('sinusoid', 'noise'):
                    arrays[f'{i}_{direction}_0.1_locked'] = np.stack([value+.1, value-.1], axis=1)
            np.savez(root/'bolt_states.npz', **arrays)
            dest = root/str(repetition)
            dest.mkdir()
            args = SimpleNamespace(out=root, model='bolt', cut='0', method='topk', seed=4951000, k=8)
            with patch.object(pilot, 'verify', return_value=({'steps': 2, 'batch': 4, 'limitations': []}, root)):
                pilot.fit_stage(args, dest)
            assert (dest/'knowledge.json').exists()
            results.append(dict(np.load(dest/'dictionary.npz')))
            h[4:] += 1000
        for key in results[0]:
            np.testing.assert_array_equal(results[0][key], results[1][key])
