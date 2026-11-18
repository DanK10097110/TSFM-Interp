"""Known-answer natural pilot controls, using cheap affine stand-ins only."""
from pathlib import Path
import builtins

import numpy as np
import pytest

from experiments.feature_search import natural_validation_study as study
from experiments.feature_search import natural_validation_analysis as analysis


def sources(count=2):
    """Four domains with disjoint IDs, a strong off-grid tone and a smaller decoy."""
    result = []
    t = np.arange(-256, 32)
    for domain in study.DOMAINS:
        for j in range(count):
            value = 3+2*np.sin(2*np.pi*5.35*t/128+.4)+.4*np.cos(2*np.pi*17.35*t/128+.7+j*.1)
            result.append((dict(source=f'{domain}:fresh{j}', domain=domain, rank=64+j,
                split='fit' if j < count//2 else 'test'), value))
    return result


class AffineNative:
    """A predictable offset exposes missing zero-anchor subtraction."""
    def __init__(self, name, checkpoint):
        self.name = name

    def predict(self, x):
        value = np.repeat((np.asarray(x)[:, -1]+.25)[:, None], 32, axis=1)
        return value.astype(np.float32), None


def fixture_run(out, monkeypatch, poison_test=False, guard_targets=False):
    """Prepare tiny artifact contracts without cache access or native construction."""
    data = sources()
    if poison_test:
        for meta, value in data:
            if meta['split'] == 'test':
                value[256:] = np.arange(32)*1e8
    bank, rows, plans, hind, future = study.build(data)
    out.mkdir()
    bank.save(out/'bank.npz')
    study.write(out/'rows.json', rows)
    study.write(out/'plans.json', plans)
    np.savez(out/'hindcast_targets.npz', targets=hind)
    for split in ('FIT', 'TEST'):
        ids = np.asarray([i for i, r in enumerate(rows) if r['split'] == split])
        np.savez(out/f'{split}_targets.npz', indices=ids, targets=future[ids])
    protocol = dict(sources={}, source_ids=[r['source'] for r in rows],
        checkpoints={m: dict(path='', weights={}) for m in study.MODELS},
        worker_cap=3*2**30, seconds_per_model=600, gates=dict(closure_fp32=2e-6),
        selector='strict mean two-hindcast loss', fit_rule='FIT only', interpretation='descriptive',
        hashes={name: study.sha(out/name) for name in ('bank.npz', 'rows.json', 'plans.json',
            'hindcast_targets.npz', 'FIT_targets.npz', 'TEST_targets.npz')})
    study.write(out/'protocol.json', protocol)
    monkeypatch.setenv('NATURAL_VALIDATION_MONITORED', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for key in study.THREAD_VARS:
        monkeypatch.setenv(key, '1')
    with monkeypatch.context() as scoped:
        if guard_targets:
            original_path, original_builtin = Path.open, builtins.open
            def path_guard(path, mode='r', *args, **kwargs):
                if 'targets' in path.name and 'r' in mode:
                    raise AssertionError('Worker read a target file')
                return original_path(path, mode, *args, **kwargs)
            def builtin_guard(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, Path)) and 'targets' in str(path) and 'r' in mode:
                    raise AssertionError('Worker read a target file')
                return original_builtin(path, mode, *args, **kwargs)
            scoped.setattr(Path, 'open', path_guard)
            scoped.setattr(builtins, 'open', builtin_guard)
        for model in study.MODELS:
            study.worker(out, model, native_factory=AffineNative)
    study.write(out/'execution.json', {m: dict(valid=True) for m in study.MODELS})
    return bank, rows, plans


def test_origins_normalize_only_own_history_and_exact_targets():
    data = sources()
    value = data[0][1].copy()
    value[224:256] *= 20
    data[0] = (data[0][0], value)
    bank, rows, plans, hind, future = study.build(data[:1])
    for j, origin in enumerate(study.ORIGINS):
        hist = value[origin-128:origin]
        root = bank.values[plans[0][j]['levels'][0][0]]
        np.testing.assert_allclose(root, (hist-hist.mean())/hist.std(), atol=1e-7)
        expected = (value[origin:origin+32]-hist.mean())/hist.std()
        actual = hind[0, j] if j < 2 else future[0]
        np.testing.assert_allclose(actual, expected)
        assert plans[0][j]['history'] == [origin-128, origin]
    assert rows[0]['split'] == 'FIT'


def test_helper_last_history_scale_is_removed_and_future_poison_is_isolated():
    meta, raw = sources()[0]
    def helper(value):
        return (value-value[128:256].mean())/value[128:256].std()
    base = study.build([(meta, helper(raw))])
    affine = study.build([(meta, helper(raw)*700+900)])
    for a, b in zip(base[0].values, affine[0].values):
        np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(base[3], affine[3], atol=1e-11)
    later = raw.copy()
    later[224:] = np.linspace(-1e4, 2e4, 64)
    poisoned = study.build([(meta, helper(later))])
    early_a, early_b = base[2][0][0], poisoned[2][0][0]
    np.testing.assert_allclose(base[0].values[early_a['levels'][0][0]],
        poisoned[0].values[early_b['levels'][0][0]], atol=1e-7)
    np.testing.assert_allclose(base[3][0, 0], poisoned[3][0, 0], atol=1e-10)
    assert abs(early_a['details'][0]['frequency']-early_b['details'][0]['frequency']) < 1e-6
    future_poison = raw.copy()
    future_poison[256:] = 1e12
    other = study.build([(meta, helper(future_poison))])
    for a, b in zip(base[0].values, other[0].values):
        np.testing.assert_array_equal(a, b)
    assert base[1:3] == other[1:3]
    np.testing.assert_array_equal(base[3], other[3])
    assert not np.array_equal(base[4], other[4])


def test_composition_keeps_dc_and_zero_anchor_and_tone_last_baseline():
    bank, _, plans, _, future = study.build(sources()[:1])
    f = AffineNative('mock', '').predict(np.stack(bank.values))[0]
    output, closure = study.assemble(bank.values, plans, f)
    assert closure < 2e-6
    for depth in study.DEPTHS:
        np.testing.assert_allclose(output[:, :, study.METHODS.index(f'output_free{depth}')], output[:, :, 0], atol=3e-7)
    assert plans[0][2]['extracted_depth'] >= 2
    k = study.METHODS.index('tone_last2')
    assert np.mean((output[0, 2, k]-future[0])**2) < .002
    assert np.mean((output[0, 2, 0]-future[0])**2) > .5
    root = bank.values[plans[0][2]['levels'][0][0]]
    np.testing.assert_array_equal(output[0, 2, study.METHODS.index('seasonal_naive24')], np.resize(root[-24:], 32))


def test_constant_hindcast_abstains_without_dropping_main_source():
    normalized, _, _, valid = study.normalize_history(np.full(128, .1))
    assert not valid
    np.testing.assert_array_equal(normalized, np.zeros(128))
    meta, value = sources()[0]
    value[:192] = 3
    bank, rows, plans, _, _ = study.build([(meta, value)])
    assert len(rows) == 1
    assert not plans[0][0]['selector_usable'] and plans[0][0]['abstained']
    f = AffineNative('mock', '').predict(np.stack(bank.values))[0]
    output, _ = study.assemble(bank.values, plans, f)
    for depth in study.DEPTHS:
        np.testing.assert_array_equal(output[0, 0, study.METHODS.index(f'hybrid{depth}')], output[0, 0, 0])


def test_selector_requires_two_losses_ties_native_and_usable_history():
    f = np.zeros((4, 2, len(study.METHODS), 32))
    f[:, :, 0] = 1
    f[0, 1, 1:4] = 2
    f[1, :, 1:4] = 1
    f[2:, :, 1:4] = .5
    choices, _ = analysis.past_choices(f, np.zeros((4, 2, 32)), [[True, True]]*3+[[False, True]])
    np.testing.assert_array_equal(choices, [[0, 0, 0], [0, 0, 0], [1, 2, 3], [0, 0, 0]])
    with pytest.raises(ValueError, match='exactly two'):
        analysis.past_choices(np.zeros((4, 3, len(study.METHODS), 32)), np.zeros((4, 3, 32)), [[True, True]]*4)
    assert analysis.fit_rule([[1, 1, 2], [1, 1, 0]]) == 0
    assert analysis.fit_rule([[4, 1, 2], [4, 1, 2]]) == 1


def test_worker_never_opens_targets_and_slot_is_required(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='slot not granted'):
        study.run(tmp_path)
    monkeypatch.delenv('NATURAL_VALIDATION_MONITORED', raising=False)
    with pytest.raises(RuntimeError, match='monitor'):
        study.worker(tmp_path, 'bolt', native_factory=AffineNative)
    fixture_run(tmp_path/'run', monkeypatch, guard_targets=True)
    assert (tmp_path/'run/bolt_worker.json').exists()


def test_end_to_end_future_poison_preserves_choices_and_fit_rules(tmp_path, monkeypatch):
    normal, poison = tmp_path/'normal', tmp_path/'poison'
    fixture_run(normal, monkeypatch)
    fixture_run(poison, monkeypatch, poison_test=True)
    original = analysis.bound_load
    def guarded(out, protocol, name):
        if name == 'FIT_targets.npz':
            assert (out/'past_choices.json').exists()
        if name == 'TEST_targets.npz':
            assert (out/'past_choices.json').exists() and (out/'fit_rules.json').exists()
        return original(out, protocol, name)
    monkeypatch.setattr(analysis, 'bound_load', guarded)
    first, second = analysis.analyze(normal), analysis.analyze(poison)
    assert first['rules'] == second['rules']
    assert study.read(normal/'past_choices.json')['choices'] == study.read(poison/'past_choices.json')['choices']
    assert [r['losses'] for r in first['source_records'] if r['split'] == 'FIT'] == [r['losses'] for r in second['source_records'] if r['split'] == 'FIT']
    assert [r['losses'] for r in first['source_records'] if r['split'] == 'TEST'] != [r['losses'] for r in second['source_records'] if r['split'] == 'TEST']
    assert all(first['gates'].values())
    assert len(first['aggregates']) == 2*4*2*(len(study.METHODS)+4)
    with pytest.raises(FileExistsError):
        analysis.analyze(normal)


def test_stability_matches_swapped_peaks_and_records_abstention():
    def plan(details, scale):
        return dict(details=details, scale=scale, abstained=not details, extracted_depth=len(details))
    a = dict(frequency=5., coefficients=[2., 0.])
    b = dict(frequency=17., coefficients=[1., 0.])
    shifted_a = dict(frequency=5.1, coefficients=[1., 0.])
    shifted_b = dict(frequency=17.2, coefficients=[.5, 0.])
    result = analysis.stability([plan([a, b], 1), plan([shifted_b, shifted_a], 2), plan([], 1)])
    matches = result['pairs'][0]['matched']
    assert [m['right_tone'] for m in matches] == [1, 0]
    np.testing.assert_allclose([m['frequency_difference'] for m in matches], [.1, .2])
    assert all(m['common_scale_amplitude_relative_change'] == 0 for m in matches)
    assert result['abstained'] == [False, False, True]


def test_prepare_checks_fresh_source_ids_and_freezes_32_sources(tmp_path, monkeypatch):
    data = sources(8)
    monkeypatch.setattr(study, 'real_data', lambda cache, offset, count: data if (offset, count) == (64, 8) else [])
    old = tmp_path/'fourier_poc4929000'
    old.mkdir()
    study.write(old/'protocol.json', dict(checkpoints={}))
    out = tmp_path/'natural_validation_pilot4951000'
    study.prepare(out, tmp_path/'unused-cache')
    p = study.read(out/'protocol.json')
    assert len(p['source_ids']) == 32 and p['ranks'] == list(range(64, 72))
    rows = study.read(out/'rows.json')
    assert sum(r['split'] == 'FIT' for r in rows) == 16
    assert p['worker_cap'] == 3*2**30 and p['seconds_per_model'] == 600
    assert not any('target' in name for name in p['worker_reads'])
    with pytest.raises(FileExistsError):
        study.prepare(out, tmp_path)
    with pytest.raises(ValueError, match='Source reuse'):
        study.prepare(tmp_path/'other', tmp_path)
    knowledge = study.read(tmp_path/'other/knowledge.json')
    assert knowledge['status'] == 'failed' and 'Source reuse' in knowledge['exact_findings']['error']
    assert knowledge['evidence_class'] == 'instrumentation'


def test_knowledge_retains_failed_attempt_and_exact_analysis(tmp_path, monkeypatch):
    out = tmp_path/'run'
    fixture_run(out, monkeypatch)
    study.record_knowledge(out, 'prepare', 'failed', dict(error='planted failure'), 'repair')
    result = analysis.analyze(out)
    knowledge = study.read(out/'knowledge.json')
    assert [e['status'] for e in knowledge['events']] == ['failed', 'complete']
    assert knowledge['events'][0]['exact_findings']['error'] == 'planted failure'
    assert knowledge['exact_findings']['aggregates'] == result['aggregates']
    assert knowledge['evidence_class'] == 'exploratory behavioral/descriptive'


@pytest.mark.parametrize('damage', ['forecast', 'input', 'invalid_execution'])
def test_hash_and_execution_gates_reject_planted_damage(tmp_path, monkeypatch, damage):
    out = tmp_path/'run'
    fixture_run(out, monkeypatch)
    if damage == 'forecast':
        with (out/'bolt_forecasts.npz').open('ab') as handle:
            handle.write(b'poison')
    elif damage == 'input':
        with (out/'plans.json').open('a') as handle:
            handle.write(' ')
    else:
        study.write(out/'execution.json', {m: dict(valid=False) for m in study.MODELS})
    with pytest.raises(ValueError):
        analysis.analyze(out)


def test_source_bootstrap_counts_ties_and_has_known_gain():
    result = analysis.summary(np.array([2., 3., 4., 5.]), np.array([1., 2., 3., 4.]))
    assert result['paired_mean_gain'] == 1 and result['gain_ci95'] == [1., 1.]
    result = analysis.summary(np.ones(4), np.array([0., 1., 2., 1.]))
    assert (result['wins'], result['ties'], result['losses']) == (1, 2, 1)


@pytest.mark.parametrize('over_budget', [False, True])
def test_monitor_serializes_models_and_rejects_resource_overrun(tmp_path, monkeypatch, over_budget):
    study.write(tmp_path/'protocol.json', dict(worker_cap=3*2**30, seconds_per_model=600))
    events = []
    class Process:
        """A completed subprocess stand-in; its attested HWM can exceed the cap."""
        pid = 99999999
        returncode = 0
        def __init__(self, command, env, **kwargs):
            model = command[-1]
            assert model in study.MODELS
            assert all(env[k] == '1' for k in study.THREAD_VARS)
            assert env['CUDA_VISIBLE_DEVICES'] == ''
            if events:
                assert study.read(tmp_path/'execution.json')[events[-1]]['valid']
            events.append(model)
            study.write(tmp_path/f'{model}_worker.json', dict(peak_rss=4*2**30 if over_budget else 100, seconds=1))
        def poll(self):
            return 0
    monkeypatch.setattr(study.subprocess, 'Popen', Process)
    if over_budget:
        with pytest.raises(RuntimeError, match='invalid'):
            study.run(tmp_path, slot_granted=True)
        assert events == ['bolt']
        assert not study.read(tmp_path/'execution.json')['bolt']['valid']
    else:
        study.run(tmp_path, slot_granted=True)
        assert events == list(study.MODELS)
        assert all(v['valid'] for v in study.read(tmp_path/'execution.json').values())
