"""Availability-only amendment controls, including singleton reporting and poison."""
import numpy as np
import pytest

from experiments.feature_search import natural_validation_v2 as v2
from experiments.feature_search import natural_validation_analysis_v2 as analysis
from experiments.feature_search.coherence_study import DOMAINS
from test_natural_validation_study import sources, fixture_run, AffineNative


def selected(domain, count):
    """Synthetic known-answer fixtures retaining exact rank and split ordering."""
    return [(m, x) for m, x in sources(count) if m['domain'] == domain]


def test_domain_helper_is_scoped_without_mutating_shared_globals(monkeypatch):
    def helper(cache, offset, count):
        return dict(domains=DOMAINS, offset=offset, count=count)
    monkeypatch.setattr(v2.coherence_study, 'real_data', helper)
    result = v2.domain_data(None, DOMAINS[1], 3)
    assert result == dict(domains=(DOMAINS[1],), offset=64, count=3)
    assert helper.__globals__['DOMAINS'] == DOMAINS and len(DOMAINS) == 4


def test_prospective_availability_uses_all_remaining_never_old_ranks(monkeypatch):
    calls = []
    def loader(cache, domain, count):
        calls.append((domain, count))
        if domain == DOMAINS[0] and count == 8:
            raise ValueError(f'Only 69 eligible sources in {domain}')
        return selected(domain, count)
    monkeypatch.setattr(v2, 'domain_data', loader)
    data, availability = v2.available_sources(None)
    assert len(data) == 23
    assert availability[DOMAINS[0]]['ranks'] == [64, 65, 66, 67, 68]
    assert availability[DOMAINS[0]]['FIT'] == 2 and availability[DOMAINS[0]]['TEST'] == 3
    assert availability[v2.PEDESTRIAN]['ranks'] == [64, 65]
    assert availability[v2.PEDESTRIAN]['FIT'] == availability[v2.PEDESTRIAN]['TEST'] == 1
    assert calls == [(DOMAINS[0], 8), (DOMAINS[0], 5), (DOMAINS[1], 8), (v2.PEDESTRIAN, 2), (DOMAINS[3], 8)]


@pytest.mark.parametrize('error', ['Selected target missing; refusing outcome-based replacement',
                                  'Only 65 eligible sources in monash_electricity_hourly'])
def test_target_error_or_fewer_than_two_never_substitutes(monkeypatch, error):
    calls = []
    def loader(cache, domain, count):
        calls.append(count)
        raise ValueError(error)
    monkeypatch.setattr(v2, 'domain_data', loader)
    with pytest.raises(ValueError):
        v2.available_sources(None)
    assert calls == [8]


def test_singleton_has_no_ci_while_larger_sample_retains_descriptive_interval():
    one = analysis.summary([4.], [1.])
    assert one['paired_mean_gain'] == 3. and one['gain_ci95'] is None
    assert 'no CI or significance' in one['interval_status']
    many = analysis.summary([4., 6.], [1., 3.])
    assert many['gain_ci95'] == [3., 3.]


def test_amendment_precedes_data_and_preserves_failed_pilot(tmp_path, monkeypatch):
    failed = tmp_path/'natural_validation_pilot4951000'
    failed.mkdir()
    v2.write(failed/'knowledge.json', dict(status='failed', error='Only 66 eligible sources in monash_pedestrian_counts'))
    previous = (failed/'knowledge.json').read_bytes()
    checkpoint = tmp_path/'fourier_poc4929000'
    checkpoint.mkdir()
    v2.write(checkpoint/'protocol.json', dict(checkpoints={m: dict(path='', weights={}) for m in v2.base.MODELS}))
    out = tmp_path/'natural_validation_pilot4951001'
    def loader(cache, domain, count):
        amendment = v2.read(out/'amendment.json')
        assert amendment['outcome_blind'] and not amendment['future_losses_read_before_amendment']
        return selected(domain, count)
    monkeypatch.setattr(v2, 'domain_data', loader)
    v2.prepare(out, tmp_path)
    p = v2.read(out/'protocol.json')
    assert p['rows'] == 26 and p['version'] == 2
    assert p['hashes']['amendment.json'] == v2.sha(out/'amendment.json')
    assert (failed/'knowledge.json').read_bytes() == previous
    monkeypatch.setenv('NATURAL_VALIDATION_MONITORED', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for key in v2.base.THREAD_VARS:
        monkeypatch.setenv(key, '1')
    for model in v2.base.MODELS:
        v2.base.worker(out, model, native_factory=AffineNative)
    v2.write(out/'execution.json', {m: dict(valid=True) for m in v2.base.MODELS})
    result = analysis.analyze(out)
    ped = [r for r in result['aggregates'] if r['domain'] == v2.PEDESTRIAN]
    other = [r for r in result['aggregates'] if r['domain'] != v2.PEDESTRIAN]
    assert all(r['n'] == 1 and r['gain_ci95'] is None and r['source_limited'] for r in ped)
    assert all(r['n'] == 4 and r['gain_ci95'] is not None and not r['source_limited'] for r in other)
    assert all(r['evidence'] == 'source-limited descriptive' for r in ped)
    assert v2.read(out/'knowledge.json')['exact_findings']['aggregates'] == result['aggregates']
    assert (failed/'knowledge.json').read_bytes() == previous


def test_v2_future_poison_cannot_change_selectors_or_fit_rules(tmp_path, monkeypatch):
    a, b = tmp_path/'a', tmp_path/'b'
    fixture_run(a, monkeypatch)
    fixture_run(b, monkeypatch, poison_test=True)
    original = analysis.bound_load
    def guarded(out, protocol, name):
        if name == 'TEST_targets.npz':
            assert (out/'fit_rules.json').exists()
        return original(out, protocol, name)
    monkeypatch.setattr(analysis, 'bound_load', guarded)
    first, second = analysis.analyze(a), analysis.analyze(b)
    assert first['rules'] == second['rules']
    assert v2.read(a/'past_choices.json')['choices'] == v2.read(b/'past_choices.json')['choices']
    assert [r['losses'] for r in first['source_records'] if r['split'] == 'TEST'] != [r['losses'] for r in second['source_records'] if r['split'] == 'TEST']
    assert all(r['gain_ci95'] is None for r in first['aggregates'])


def test_v2_selector_matches_frozen_original():
    from experiments.feature_search.natural_validation_analysis import past_choices, fit_rule
    rng = np.random.default_rng(4951001)
    f = rng.normal(size=(12, 2, len(v2.base.METHODS), 32))
    y = rng.normal(size=(12, 2, 32))
    usable = np.ones((12, 2), bool)
    for expected, actual in zip(past_choices(f, y, usable), analysis.past_choices(f, y, usable)):
        np.testing.assert_array_equal(expected, actual)
    losses = rng.normal(size=(4, 7))**2
    assert fit_rule(losses) == analysis.fit_rule(losses)
    f[:] = 0
    for expected, actual in zip(past_choices(f, y, usable), analysis.past_choices(f, y, usable)):
        np.testing.assert_array_equal(expected, actual)
