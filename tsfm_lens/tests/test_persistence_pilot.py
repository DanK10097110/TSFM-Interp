"""Known-answer and adversarial guards for the bounded persistence pilot."""
import numpy as np
import pytest

from experiments.feature_search import persistence_protocol as protocol
from experiments.feature_search import persistence_analysis as analysis
from experiments.feature_search import persistence_study as study


def test_generator_changes_only_when_observable():
    baseline = protocol.signal(4951000, 'persistent', 0)
    for regime in protocol.REGIMES[1:]:
        future_change = protocol.signal(4951000, regime, 0)
        cue = protocol.signal(4951000, regime, -32)
        assert np.array_equal(baseline[:128], future_change[:128])
        assert not np.array_equal(baseline[128:], future_change[128:])
        assert np.array_equal(baseline[:96], cue[:96])
        assert not np.array_equal(baseline[96:128], cue[96:128])
    decay = protocol.signal(4951000, 'amplitude_decay', 0)
    stopped = protocol.signal(4951000, 'abrupt_cessation', 0)
    np.testing.assert_allclose(decay[144]-stopped[144], .5*(baseline[144]-stopped[144]))


def test_sources_never_split_worlds():
    x, y, rows = protocol.corpus()
    assert x.shape == (64, 128) and y.shape == (64, 32)
    fit = {r['source'] for r in rows if r['split'] == 'fit'}
    test = {r['source'] for r in rows if r['split'] == 'test'}
    assert len(fit) == len(test) == 4 and not fit.intersection(test)
    for seed in protocol.SEEDS:
        same = [r for r in rows if r['seed'] == seed and r['arm'] == 'identical']
        assert len(same) == 4
        assert len({x[r['row']].tobytes() for r in same}) == 1
        assert len({y[r['row']].tobytes() for r in same}) == 4


def test_plan_retains_level_in_residual_and_zero_really_means_zero():
    t = np.arange(-128, 0)
    x = 3+8*np.sin(2*np.pi*t/19.2+.4)
    info, residual, tone, seasonal = protocol.plan(x)
    expected = 8*np.sin(2*np.pi*np.arange(32)/19.2+.4)
    assert info['accepted'] == 1
    np.testing.assert_allclose(residual, 3, atol=1e-5)
    np.testing.assert_allclose(tone, expected, atol=1e-5)
    lag = info['lag']
    np.testing.assert_array_equal(seasonal, x[-lag:][np.arange(32) % lag])
    info, residual, tone, seasonal = protocol.plan(np.full(128, 3.))
    assert info['accepted'] == 0
    assert np.array_equal(tone, np.zeros(32))
    assert np.array_equal(residual, np.full(128, 3.))


def test_identity_gate_rejects_tiny_perturbations():
    a = np.ones(3); b = a.copy(); b[0] = np.nextafter(1., 2.)
    protocol.exact(a, a.copy(), 'self')
    with pytest.raises(ValueError, match='Exact identity'):
        protocol.exact(a, b, 'planted tiny regression')


def test_finite_artifacts_reject_nan(tmp_path):
    with pytest.raises(ValueError):
        protocol.write(tmp_path/'bad.json', {'bad': float('nan')})
    assert not (tmp_path/'bad.json').exists()
    with pytest.raises(ValueError):
        protocol.finite([0., float('inf')])


def test_expected_loss_is_a_tradeoff_not_detection():
    targets = np.array([[2., 2.], [0., 0.], [0., 0.], [0., 0.]])
    result = analysis.risk(np.array([.5, .5]), targets, 1., .25)
    assert result == dict(expected=.75, irreducible=.75, excess=0.)
    assert analysis.risk(np.array([2., 2.]), targets, 1., 1.)['expected'] == 0.
    assert analysis.risk(np.array([2., 2.]), targets, 1., 0.)['expected'] == 4.


def test_cue_gain_uses_same_target_and_correct_sign():
    assert analysis.cue_gain(np.array([4., 4.]), np.zeros(2), np.zeros(2), 2.) == 8.
    assert analysis.cue_gain(np.ones(2), np.ones(2), np.array([9., 3.]), 1.) == 0.


def test_target_guard_rejects_direct_and_symlink_open(tmp_path):
    evaluation = tmp_path/'evaluation'; evaluation.mkdir()
    target = evaluation/'targets.npz'; target.write_bytes(b'future')
    alias = tmp_path/'alias'; alias.symlink_to(target)
    guard = study.target_guard(evaluation)
    for path in [target, alias, bytes(str(target), 'utf-8')]:
        with pytest.raises(PermissionError):
            guard('open', (path, 'r', 0))
    guard('open', (str(tmp_path/'inputs.npz'), 'r', 0))


def test_frozen_inputs_cannot_change(tmp_path):
    folder = tmp_path/'inference'; folder.mkdir()
    inp = folder/'inputs.npz'; inp.write_bytes(b'original')
    protocol.write(folder/'protocol.json', dict(sources={}, hashes={'inputs.npz': protocol.sha(inp)}))
    study.bound(tmp_path)
    inp.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        study.bound(tmp_path)


def test_uncertainty_resamples_sources():
    result = analysis.interval([0., 2.])
    assert result['n_sources'] == 2 and result['mean'] == 1.
    assert result['ci95'] == [0., 2.] and result['positive_sources'] == 1
