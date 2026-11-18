"""Known-answer fresh-source, multiplicity, temporal and worker-isolation tests."""
import builtins
from pathlib import Path

import numpy as np
import pytest

from experiments.feature_search import natural_validation_replication as rep
from experiments.feature_search import natural_validation_replication_analysis as analysis
from experiments.feature_search.natural_validation_study import normalize_history


class Values:
    """Array spy forbids materializing target values beyond mainpast256."""
    def __init__(self, array, target=False, selected=False):
        self.array, self.target, self.selected = array, target, selected

    def __len__(self):
        return len(self.array)

    def slice(self, start, size):
        if self.target:
            assert start+size <= len(self.array)-32
        return Values(self.array[start:start+size], target=False, selected=True)

    def to_numpy(self, zero_copy_only=False):
        assert self.selected
        return self.array


class Scalar:
    """Minimal Arrow scalar interface."""
    def __init__(self, value, target=False):
        self.value = value
        if not isinstance(value, str):
            self.values = Values(value, target=target)

    def as_py(self):
        assert isinstance(self.value, str)
        return self.value


class Table:
    """Known source ordering with one ineligible constant past and poisoned futures."""
    def __init__(self, poison=False):
        stamps = np.arange(288).astype('timedelta64[h]')+np.datetime64('2020-01-01')
        self.columns = {key: [] for key in ('target', 'timestamp', 'id')}
        for i in range(137):
            x = np.sin(np.arange(288)/5.)
            if i == 0:
                x[:256] = 3.
            if poison:
                x[256:] = np.nan
            self.columns['target'].append(Scalar(x, target=True))
            self.columns['timestamp'].append(Scalar(stamps))
            self.columns['id'].append(Scalar(f'id{i}'))

    def __len__(self):
        return len(self.columns['id'])

    def __getitem__(self, name):
        return self.columns[name]


def test_availability_never_materializes_future_values_and_ignores_poison():
    normal = rep.audit_table(Table(), rep.DOMAINS[0])
    poisoned = rep.audit_table(Table(poison=True), rep.DOMAINS[0])
    assert normal == poisoned and normal['sufficient']
    order = [int(i) for i in np.random.default_rng(492306).permutation(137) if i != 0]
    assert normal['selected_source_ids'] == [f'{rep.DOMAINS[0]}:id{i}' for i in order[72:136]]
    assert normal['ranks'] == list(range(72, 136))
    with pytest.raises(ValueError, match='mainpast256'):
        rep.eligible_past(np.zeros(288), np.arange(288))


def test_short_availability_blocks_before_any_helper_future_access(monkeypatch):
    audit = {d: dict(sufficient=True, selected_source_ids=[]) for d in rep.DOMAINS}
    audit[rep.DOMAINS[-1]]['sufficient'] = False
    def forbidden(*args, **kwargs):
        raise AssertionError('Helper accessed before full availability audit passed')
    monkeypatch.setattr(rep.cached, 'real_data', forbidden)
    with pytest.raises(ValueError, match='Insufficient'):
        rep.source_data(None, audit)


def test_original_helper_scope_and_target_errors_do_not_replace_sources(monkeypatch):
    data = fixture_sources()
    calls = []
    def helper(cache, offset, count):
        domain = globals()['DOMAINS'][0]
        calls.append((domain,offset,count))
        return [(dict(m,split='fit' if j<32 else 'test'),x)
                for j,(m,x) in enumerate((v for v in data if v[0]['domain']==domain))]
    monkeypatch.setattr(rep.cached,'real_data',helper)
    audit = {d:dict(sufficient=True,selected_source_ids=[m['source'] for m,_ in data if m['domain']==d]) for d in rep.DOMAINS}
    selected = rep.source_data(None,audit)
    assert calls==[(d,72,64) for d in rep.DOMAINS]
    assert len(selected)==192 and all(m['split']=='test' for m,_ in selected)
    assert sum(m['helper_split']=='fit' for m,_ in selected)==96
    calls.clear()
    def failure(cache,offset,count):
        calls.append((offset,count))
        raise ValueError('Selected target missing; refusing outcome-based replacement')
    monkeypatch.setattr(rep.cached,'real_data',failure)
    with pytest.raises(ValueError,match='target missing'):
        rep.source_data(None,audit)
    assert calls==[(72,64)]


def fixture_sources(poison=False):
    """192 distinct identities on deterministic tones; no native models needed."""
    t = np.arange(-256, 32)
    value = 3+8*np.sin(2*np.pi*5.35*t/128+.4)+.4*np.cos(2*np.pi*17.35*t/128+.7)
    result = []
    for domain in rep.DOMAINS:
        for i in range(64):
            x = value.copy()
            if poison:
                x[256:] = np.arange(32)*1e8
            result.append((dict(source=f'{domain}:rep{i}', domain=domain, rank=72+i, split='test'), x))
    return result


def setup_run(root, monkeypatch, poison=False):
    """Prepare the exact192 contract with mocked cache inputs and frozen pilot rules."""
    root.mkdir(parents=True, exist_ok=True)
    pilot = root/'natural_validation_pilot4951001'
    pilot.mkdir()
    rules = {'bolt': {rep.DOMAINS[0]: 'hybrid1', rep.DOMAINS[1]: 'past_selector3', rep.DOMAINS[2]: 'native'},
             'timesfm': {rep.DOMAINS[0]: 'hybrid1', rep.DOMAINS[1]: 'hybrid3', rep.DOMAINS[2]: 'native'}}
    rep.write(pilot/'fit_rules.json', dict(rules=rules))
    rep.write(pilot/'protocol.json', dict(nodes=277, checkpoints={m: dict(path='', weights={}) for m in rep.MODELS}))
    rep.write(pilot/'execution.json', {m: dict(seconds=22.) for m in rep.MODELS})
    rep.write(pilot/'knowledge.json', dict(status='complete'))
    data = fixture_sources(poison)
    audit = {d: dict(sufficient=True, selected_source_ids=[m['source'] for m, _ in data if m['domain']==d]) for d in rep.DOMAINS}
    out = root/'natural_validation_replication4955000'
    def availability(cache):
        assert rep.read(out/'declaration.json')['hold_native']
        assert not (out/'main_targets.npz').exists()
        return audit
    monkeypatch.setattr(rep, 'availability', availability)
    monkeypatch.setattr(rep, 'source_data', lambda cache, audit: data)
    rep.prepare(out, root/'unused-cache')
    return out


class AffineNative:
    """Cheap last-value forecast with offset; never imports or loads native models."""
    def __init__(self, name, checkpoint):
        self.name = name

    def predict(self, x):
        return np.repeat((np.asarray(x)[:, -1]+.25)[:, None], 32, axis=1).astype(np.float32), None


def mock_workers(out, monkeypatch):
    """Forbid every target archive read during actual worker orchestration code."""
    monkeypatch.setenv('NATURAL_REPLICATION_MONITORED', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for key in rep.base.THREAD_VARS:
        monkeypatch.setenv(key, '1')
    with monkeypatch.context() as scoped:
        original, original_path = builtins.open, Path.open
        def guarded(path, mode='r', *args, **kwargs):
            if isinstance(path, (str, Path)) and 'targets' in str(path) and 'r' in mode:
                raise AssertionError('Worker accessed targets')
            return original(path, mode, *args, **kwargs)
        def path_guarded(path, mode='r', *args, **kwargs):
            if 'targets' in path.name and 'r' in mode:
                raise AssertionError('Worker accessed targets')
            return original_path(path, mode, *args, **kwargs)
        scoped.setattr(builtins, 'open', guarded)
        scoped.setattr(Path, 'open', path_guarded)
        for model in rep.MODELS:
            rep.worker(out, model, native_factory=AffineNative)
    rep.write(out/'execution.json', {m: dict(valid=True) for m in rep.MODELS})


def test_preparation_freezes192_test_sources_primary6_and_secondary(tmp_path, monkeypatch):
    out = setup_run(tmp_path, monkeypatch)
    p, rows = rep.read(out/'protocol.json'), rep.read(out/'rows.json')
    assert p['rows'] == 192 and len(set(p['source_ids'])) == 192
    assert len(p['primary_family']) == 6 and p['primary_depth'] == 1
    assert p['hold_native'] and p['fresh_slot_required']
    assert all(r['split']=='TEST' for r in rows)
    assert all([r['rank'] for r in rows if r['domain']==d] == list(range(72,136)) for d in rep.DOMAINS)
    assert 'monash_pedestrian_counts' not in p['domains']
    plans = rep.read(out/'plans.json')
    assert all(len(q['details']) <= 1 for i,r in enumerate(rows) if r['domain'] != 'm4_hourly' for q in plans[i])
    assert p['frozen_pilot_rules']['bolt']['m4_hourly'] == 'past_selector3'
    assert p['seconds_per_model'] == 900 and p['worker_cap'] == 3*2**30
    assert rep.read(out/'knowledge.json')['exact_findings']['native_launched'] is False
    assert not (out/'execution.json').exists()


def test_reused_source_fails_and_persists_negative_knowledge(tmp_path,monkeypatch):
    monkeypatch.setattr(rep.base,'prior_sources',lambda root:({f'{rep.DOMAINS[0]}:rep0'},{}))
    with pytest.raises(ValueError,match='fresh distinct'):
        setup_run(tmp_path,monkeypatch)
    out=tmp_path/'natural_validation_replication4955000'
    k=rep.read(out/'knowledge.json')
    assert k['status']=='failed' and 'fresh distinct' in k['exact_findings']['error']
    assert not (out/'main_targets.npz').exists()


def test_origin_normalization_and_main_future_poison_isolation():
    source = fixture_sources()[0]
    normal = rep.build([source])
    affine = rep.build([(source[0], source[1]*731+515)])
    poison = source[1].copy()
    poison[256:] = 1e12
    changed = rep.build([(source[0], poison)])
    assert normal[1:3] == changed[1:3]
    np.testing.assert_array_equal(normal[3], changed[3])
    assert not np.array_equal(normal[4], changed[4])
    for a,b,c in zip(normal[0].values, affine[0].values, changed[0].values):
        np.testing.assert_allclose(a,b,atol=2e-6)
        np.testing.assert_array_equal(a,c)
    later = source[1].copy()
    later[224:] *= 1000
    centered = (later-later[128:256].mean())/later[128:256].std()
    other = rep.build([(source[0], centered)])
    np.testing.assert_allclose(normal[0].values[normal[2][0][0]['root']], other[0].values[other[2][0][0]['root']],atol=1e-7)
    for j,o in enumerate(rep.ORIGINS):
        expected, _, _, _ = normalize_history(source[1][o-128:o])
        np.testing.assert_allclose(normal[0].values[normal[2][0][j]['root']],expected,atol=1e-7)


def test_fixed_depth1_controls_and_frozen_secondary_composition():
    data=[fixture_sources()[i] for i in (0,64,128)]
    bank,rows,plans,_,_=rep.build(data)
    forecasts=AffineNative('mock','').predict(np.stack(bank.values))[0]
    rules={'bolt':{rep.DOMAINS[0]:'hybrid1',rep.DOMAINS[1]:'past_selector3',rep.DOMAINS[2]:'native'}}
    result=rep.assemble(bank.values,plans,forecasts,rows,'bolt',rules)
    for i,source in enumerate(plans):
        for j,q in enumerate(source):
            tone=rep.continuation(q['details'],q['primary_depth'])
            np.testing.assert_array_equal(result[i,j,1],tone+forecasts[q['residual1']])
            np.testing.assert_array_equal(result[i,j,3],tone+bank.values[q['residual1']][-1])
            np.testing.assert_array_equal(result[i,j,2],np.resize(bank.values[q['root']][-24:],32))
    assert plans[1][2]['secondary_depth']>=2
    q=plans[1][2]
    np.testing.assert_array_equal(result[1,2,4],rep.continuation(q['details'],q['secondary_depth'])+forecasts[q['secondary_residual']])
    np.testing.assert_array_equal(result[2,:,4],result[2,:,0])


def test_primary_uncertainty_is_source_paired_adjusted6_and_ties_explicit():
    exact = analysis.paired_summary([2,3,4,5], [1,2,3,4], family=6, n_boot=1999)
    assert exact['n']==4 and exact['paired_mean_gain']==1 and exact['gain_interval']==[1.,1.]
    assert exact['non_tie_sign_p']==.0625 and exact['interval_marginal_coverage']==1-.05/6
    gain = np.array([-4,-2,0,0,1,4,8,11.])
    unadjusted = analysis.paired_summary(np.full(8,20.), 20-gain, family=1, n_boot=1999)
    adjusted = analysis.paired_summary(np.full(8,20.), 20-gain, family=6, n_boot=1999)
    assert adjusted['gain_interval'][0] < unadjusted['gain_interval'][0]
    assert adjusted['gain_interval'][1] > unadjusted['gain_interval'][1]
    ties = analysis.paired_summary(np.ones(4), [0,1,2,1], n_boot=99)
    assert ties['winrate']==.25 and ties['non_tie_winrate']==.5
    assert (ties['wins'],ties['ties'],ties['losses'])==(1,2,1)
    assert analysis.paired_summary([1],[1])['gain_interval'] is None
    with pytest.raises(ValueError,match='source-loss'):
        analysis.paired_summary(np.ones((4,32)), np.zeros((4,32)))


def test_holm_keeps_six_and_undefined_diagnostics_in_family():
    np.testing.assert_allclose(analysis.holm6([.001,.02,.03,.5,1,1]), [.006,.1,.12,1,1,1])
    with pytest.raises(ValueError,match='six'):
        analysis.holm6([.001,.02,.03])


def test_continuous_diagnostic_known_positive_negative_and_temporal_decoy():
    x = np.arange(12.)
    past = np.column_stack([-x,3*x])
    good = analysis.continuous_diagnostic(past, x, n_perm=199)
    bad = analysis.continuous_diagnostic(past, -x, n_perm=199)
    assert good['spearman'] == 1 and good['positive_association_p'] == .005
    assert bad['spearman'] == -1 and bad['positive_association_p'] == 1
    np.testing.assert_array_equal(good['past_mean_benefit'],x)
    assert good['opposite_sign_hindcasts']==11 and good['either_hindcast_tied']==1
    np.testing.assert_allclose(good['per_hindcast_pearson_with_main'], [-1,1])
    constant = analysis.continuous_diagnostic(np.ones((12,2)),x,n_perm=199)
    assert constant['spearman'] is None and constant['positive_association_p']==1


def test_selector_uses_both_hindcasts_ties_native_and_no_future_threshold():
    f = np.zeros((4,2,len(rep.METHODS),32))
    f[:,:,0] = 1
    f[0,1,1] = 2
    f[1,:,1] = 1
    f[2:,:,1] = .5
    chosen, _, _ = analysis.past_choices(f,np.zeros((4,2,32)),[[True,True]]*3+[[False,True]])
    np.testing.assert_array_equal(chosen,[False,False,True,False])
    with pytest.raises(ValueError, match='two past'):
        analysis.past_choices(np.zeros((4,3,len(rep.METHODS),32)),np.zeros((4,3,32)),np.ones((4,2)))


def test_worker_target_blindness_and_fresh_hold(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='HOLD native'):
        rep.run(tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()
    monkeypatch.delenv('NATURAL_REPLICATION_MONITORED',raising=False)
    with pytest.raises(RuntimeError,match='fresh slot'):
        rep.worker(tmp_path,'bolt',native_factory=AffineNative)
    out = setup_run(tmp_path,monkeypatch)
    mock_workers(out,monkeypatch)
    assert all(rep.read(out/f'{m}_worker.json')['repeat_exact'] for m in rep.MODELS)


def test_forecast_tampering_fails_before_main_target_read(tmp_path,monkeypatch):
    out=setup_run(tmp_path,monkeypatch)
    mock_workers(out,monkeypatch)
    with (out/'bolt_forecasts.npz').open('ab') as handle:
        handle.write(b'planted damage')
    original=analysis.bound_targets
    def guarded(out,p,name):
        assert name!='main_targets.npz'
        return original(out,p,name)
    monkeypatch.setattr(analysis,'bound_targets',guarded)
    with pytest.raises(ValueError,match='gate failed'):
        analysis.analyze(out)


def test_analysis_future_poison_cannot_change_frozen_choices_or_pilot_rules(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'BOOTSTRAPS',199)
    monkeypatch.setattr(analysis,'PERMUTATIONS',49)
    a = setup_run(tmp_path/'a',monkeypatch)
    b = setup_run(tmp_path/'b',monkeypatch,poison=True)
    mock_workers(a,monkeypatch)
    mock_workers(b,monkeypatch)
    original = analysis.bound_targets
    def guarded(out,p,name):
        if name=='main_targets.npz':
            assert (out/'past_choices.json').exists()
            assert len(rep.read(out/'past_choices.json')['choices'])==2
        return original(out,p,name)
    monkeypatch.setattr(analysis,'bound_targets',guarded)
    first,second = analysis.analyze(a),analysis.analyze(b)
    assert rep.read(a/'past_choices.json')['choices']==rep.read(b/'past_choices.json')['choices']
    assert rep.read(a/'past_choices.json')['frozen_pilot_rules']==rep.read(b/'past_choices.json')['frozen_pilot_rules']
    assert [r['paired_mean_gain'] for r in first['primary']] != [r['paired_mean_gain'] for r in second['primary']]
    assert len(first['primary'])==len(first['diagnostic'])==6
    assert all(r['n']==64 and r['interval_family']==6 for r in first['primary'])
    assert len(first['conditional'])==12 and all(first['gates'].values())
    assert rep.read(a/'knowledge.json')['status']=='complete'


def test_budget_has_conservative_allowance_and_rejects_over15min():
    execution = {m:dict(seconds=22.) for m in rep.MODELS}
    value = rep.budget(1344,277,execution)
    assert all(x['allowance_seconds']==2*x['estimated_seconds'] and x['within_900_seconds'] for x in value.values())
    over = rep.budget(20000,277,execution)
    assert not any(x['within_900_seconds'] for x in over.values())


@pytest.mark.parametrize('over',[False,True])
def test_monitor_serial_and_resource_rejection(tmp_path,monkeypatch,over):
    events=[]
    class Process:
        """Completed process stand-in; no native construction."""
        pid=99999999
        returncode=0
        def __init__(self,command,env,**kwargs):
            if events:
                assert rep.read(tmp_path/'execution.json')[events[-1]]['valid']
            model=command[-1]
            assert all(env[k]=='1' for k in rep.base.THREAD_VARS)
            events.append(model)
            rep.write(tmp_path/f'{model}_worker.json',dict(peak_rss=4*2**30 if over else 100,seconds=1))
        def poll(self):
            return 0
    monkeypatch.setattr(rep.subprocess,'Popen',Process)
    if over:
        with pytest.raises(RuntimeError,match='invalid'):
            rep.run(tmp_path,slot_granted=True)
        assert events==['bolt'] and rep.read(tmp_path/'knowledge.json')['status']=='failed'
    else:
        rep.run(tmp_path,slot_granted=True)
        assert events==list(rep.MODELS)
