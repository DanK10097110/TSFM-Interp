"""Known-answer replication controls; TEST decoys and batch-slot guard included."""
import numpy as np
import pytest
from experiments.feature_search import persistence_replication_protocol as p
from experiments.feature_search import persistence_replication_analysis as a
from experiments.feature_search import persistence_replication_worker as w
from experiments.feature_search.persistence_protocol import signal as pilot_signal, write, sha


def test_generator_matches_pilot_and_hides_future_regime():
    for regime in p.REGIMES:
        np.testing.assert_array_equal(p.signal(4953000, 19.2, regime, 32), pilot_signal(4953000, regime, -32))
        np.testing.assert_array_equal(p.signal(4953000, 24., regime, 0)[:128], p.signal(4953000, 24., 'persistent', 0)[:128])
    assert not np.array_equal(p.signal(4953000, 19.2, 'persistent', 0), p.signal(4953000, 24., 'persistent', 0))


def test_source_split_and_cue_grid_are_frozen():
    assert p.SEEDS == tuple(range(4953000, 4953064))
    assert p.LEADS == (0, 8, 16, 32, 64) and p.PERIODS == (19.2, 24.)
    x, y, rows = p.corpus([4953031, 4953032])
    assert x.shape == (80, 128) and y.shape == (80, 32)
    assert {r['split'] for r in rows if r['seed'] == 4953031} == {'fit'}
    assert {r['split'] for r in rows if r['seed'] == 4953032} == {'test'}


def test_recent_refit_ignores_old_prefix_and_preserves_dc():
    t = np.arange(-128, 0)
    x = 3+8*np.sin(2*np.pi*t/19.2+.4)
    decoy = x.copy(); decoy[:-32] += .2*np.cos(2*np.pi*t[:-32]/19.2)
    first, second = p.make_plan(x), p.make_plan(decoy)
    assert first[0]['accepted'] == second[0]['accepted'] == 1
    np.testing.assert_array_equal(first[4], second[4])
    np.testing.assert_allclose(first[4], 8*np.sin(2*np.pi*np.arange(32)/19.2+.4), atol=1e-5)
    np.testing.assert_allclose(first[2], 3, atol=1e-4)
    assert p.make_plan(np.full(128, 3.))[0]['accepted'] == 0


def test_recent_error_detects_observed_change_at_shared_phase_origin():
    t = np.arange(-128, 0)
    detail = dict(frequency=128/24, coefficients=[8., 0.])
    x = 2+8*np.sin(2*np.pi*t/24)
    assert p.recent_error(x, detail) < 1e-12
    changed = x.copy(); changed[-32:] = 2.
    assert abs(p.recent_error(changed, detail)-1.) < 1e-12


def test_veto_is_observation_only_and_cannot_force_extraction():
    assert p.use_hybrid(.2, 1, .25)
    assert not p.use_hybrid(.8, 1, .25)
    assert not p.use_hybrid(0., 0, None)
    assert not p.use_hybrid(0., 1, -1.)
    assert p.use_hybrid(100., 1, None)


def test_fit_selector_ignores_adversarial_test_targets_and_breaks_ties_native():
    rows = [dict(source='fit-a', split='fit', query=0, scale=1.),
            dict(source='fit-b', split='fit', query=1, scale=1.),
            dict(source='test-decoy', split='test', query=2, scale=1.)]
    native = np.zeros((3, 32)); hybrid = np.ones((3, 32))
    plans = [dict(accepted=1, recent_error=e) for e in [.2, .8, .1]]
    targets = np.array([np.ones(32), np.zeros(32), np.full(32, 1e8)])
    before = a.choose(rows, targets, native, hybrid, plans)
    assert before['threshold'] == .25
    targets[2] = np.nan
    assert a.choose(rows, targets, native, hybrid, plans) == before
    tie = a.choose(rows[:2], np.zeros((2, 32)), native, native, plans)
    assert tie['threshold'] == -1.


def test_worker_cannot_bypass_slot_hold(tmp_path):
    with pytest.raises(PermissionError, match='HOLD'):
        w.require_grant(tmp_path, 'bolt')
    with pytest.raises(PermissionError, match='HOLD'):
        w.run(tmp_path, 'bolt', None)
    write(tmp_path/'protocol.json', {'frozen': True})
    write(tmp_path/'bolt_slot_grant.json', dict(coordinator_grant='explicit test fixture', protocol_sha256='stale'))
    with pytest.raises(PermissionError, match='stale'):
        w.require_grant(tmp_path, 'bolt')
    write(tmp_path/'bolt_slot_grant.json', dict(coordinator_grant='explicit test fixture', protocol_sha256=sha(tmp_path/'protocol.json')))
    assert w.require_grant(tmp_path, 'bolt')['coordinator_grant'] == 'explicit test fixture'


def test_holm_retains_complete_six_test_family():
    np.testing.assert_allclose(a.holm([.001, .01, .02, .2, .5, 1.]), [.006, .05, .08, .6, 1., 1.])
    assert a.holm([1.]*6) == [1.]*6


def test_statistics_count_sources_and_do_not_promote_ties():
    result = a.summarize([1., -1., 2.])
    assert result['n_sources'] == 3 and result['positive_sources'] == 2
    assert result['one_sided_sign_p'] == .5
    assert a.summarize([0., 0.])['one_sided_sign_p'] == 1.
