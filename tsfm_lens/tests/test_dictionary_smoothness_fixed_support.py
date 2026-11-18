"""Known answers for midpoint choice, rectifier gates and smoothing/fidelity tradeoffs."""
import numpy as np
from experiments.feature_search import dictionary_smoothness_fixed_support_math as m
from experiments.feature_search import dictionary_smoothness_fixed_support_audit as audit


def artifact(learned=False):
    """Two-dimensional identity weights with a single selected coordinate."""
    a = dict(center=np.zeros(2), scale=np.array(1.), k=np.array(1))
    a.update(dict(W_enc=np.eye(2), W_dec=np.eye(2), b_dec=np.zeros(2), b_enc=np.zeros(2))
             if learned else dict(basis=np.eye(2)))
    return a


def evaluate(h, a, method, mode):
    """Small complete controlled inference fixture."""
    pre = m.coefficients(h, a, method)
    indices, gate = m.central_masks(pre, int(a['k']), method)
    return m.decode_control(pre, a, method, mode, indices, gate)


def test_actual_midpoint_support_not_activation_midpoint_or_first_k():
    h = np.array([[[[2., .4]]], [[[2., .6]]], [[[.2, 3.]]], [[[2., .5]]]])
    a = artifact()
    pre = m.coefficients(h, a, 'pca_adaptive')
    ids, gate = m.central_masks(pre, 1, 'pca_adaptive')
    assert ids.tolist() == [[[1]]]
    g = m.decode_control(pre, a, 'pca_adaptive', 'fixed_support', ids, gate)
    np.testing.assert_array_equal(g[..., 0], 0)
    np.testing.assert_allclose(g[0] + g[1] - 2 * g[3], 0, atol=1e-15)
    assert np.max(abs(g[0] + g[1] - 2 * g[2])) > 4


def test_preserved_relu_and_frozen_relu_are_distinct_with_zero_tie():
    h = np.array([[[[-1., 0.]]], [[[3., 0.]]], [[[1., 0.]]], [[[1., 0.]]]])
    a = artifact(True)
    preserved = evaluate(h, a, 'topk', 'fixed_support')
    affine = evaluate(h, a, 'topk', 'fixed_affine')
    assert preserved[0, 0, 0, 0] == 0
    assert affine[0, 0, 0, 0] == -1
    assert (preserved[0] + preserved[1] - 2 * preserved[3])[0, 0, 0] == 1
    np.testing.assert_array_equal(affine[0] + affine[1] - 2 * affine[3], 0)
    tied = np.zeros((4, 1, 1, 4))
    ids, gate = m.central_masks(tied, 2, 'topk')
    assert ids.tolist() == [[[0, 1]]] and not gate.any()


def test_smoothing_can_worsen_state_fidelity_while_improving_response():
    h = np.array([[[[1.1, .9]]], [[[.9, 1.1]]], [[[1., 1.]]], [[[1., 1.]]]])
    a = artifact()
    hard = evaluate(h, a, 'pca_adaptive', 'hard')
    fixed = evaluate(h, a, 'pca_adaptive', 'fixed_support')
    assert np.linalg.norm(hard[0] + hard[1] - 2 * hard[3]) > 1
    np.testing.assert_allclose(fixed[0] + fixed[1] - 2 * fixed[3], 0, atol=1e-15)
    baseline = m.fidelity(h, hard, hard, np.ones(1))
    result = m.fidelity(h, fixed, hard, np.ones(1))
    assert result['state_reconstruction_RMS'][0] > baseline['state_reconstruction_RMS'][0]
    assert result['response_reconstruction_RMS'][0] < baseline['response_reconstruction_RMS'][0]
    np.testing.assert_allclose(result['response_reconstruction_RMS'], np.sqrt(.005), atol=1e-15)
    assert result['central_output_change_from_hard_RMS'][0] == 0


def test_bias_scale_dense_replay_and_equal_source_aggregation():
    a = artifact(True)
    a.update(center=np.array([2., -1.]), scale=np.array(3.), b_dec=np.array([.2, -.4]), b_enc=np.array([.1, .3]))
    h = np.array([[[[1., 2.]]], [[[4., -2.]]], [[[2., .1]]], [[[2.5, 0.]]]])
    for mode in ('hard', 'fixed_support', 'fixed_affine'):
        g = evaluate(h, a, 'nested', mode)
        np.testing.assert_allclose(g, audit.reference(h, a, 'nested', mode, np.float64), atol=1e-14)
    paths = [dict(source='a')] * 100 + [dict(source='b')]
    summary = m.source_summary(np.array([2.] * 100 + [0.]), paths)
    assert summary['n_sources'] == 2 and summary['mean'] == 1.
