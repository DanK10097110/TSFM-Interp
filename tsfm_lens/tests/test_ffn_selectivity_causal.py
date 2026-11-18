"""Known-answer causal preparation, activity matching and toy-only interventions."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments.feature_search import ffn_selectivity_causal as causal
from experiments.feature_search import ffn_selectivity_causal_analysis as analysis
from experiments.feature_search import ffn_selectivity_causal_run as runner
from test_ffn_selectivity_study import ToyNative


def test_FIT_activity_matching_ignores_TEST_poison_and_variant_repetition():
    rng=np.random.default_rng(12)
    raw=rng.uniform(.1,.3,size=(6,15,2,64))
    fit=np.array([True]*3+[False]*3)
    _,conditions=causal.stage1.perturbations(0)
    values=causal.activity(raw,fit,conditions)
    poisoned=raw.copy();poisoned[~fit]*=1e9
    np.testing.assert_array_equal(values,causal.activity(poisoned,fit,conditions))
    duplicate=np.concatenate([raw,raw[:,1:5]],axis=1)
    np.testing.assert_allclose(values,causal.activity(duplicate,fit,conditions+conditions[:4]),atol=1e-15)
    selected=[0,1,2,3]
    result=causal.match_units(values,selected)
    assert result==causal.match_units(causal.activity(poisoned,fit,conditions),selected)
    draws=np.asarray(result['draws'])
    assert draws.shape==(8,4) and all(len(set(row))==4 for row in draws)
    assert not set(draws.ravel())&set(selected)
    assert all(d['maximum_log_ratio']<=np.log(2) for d in result['details'])
    bad=np.zeros((10,2));bad[:4]=1
    with pytest.raises(ValueError,match='Insufficient'):
        causal.match_units(bad,selected)


def test_unit_delta_patch_preserves_unselected_coordinates_and_baseline():
    anchor=np.arange(24,dtype=float).reshape(1,3,8)
    clean=anchor+np.arange(1,25).reshape(1,3,8)
    result=causal.patch_units(clean,anchor,[1,6])
    np.testing.assert_array_equal(result[..., [1,6]],anchor[..., [1,6]])
    np.testing.assert_array_equal(result[..., [0,2,3,4,5,7]],clean[..., [0,2,3,4,5,7]])
    np.testing.assert_array_equal(causal.patch_units(anchor,anchor,[1,6]),anchor)
    assert not np.array_equal(result[..., [1,6]],np.zeros((1,3,2)))


def test_response_removal_distinguishes_direction_energy_offtarget_and_amplification():
    baseline=np.zeros(2);clean=np.array([1.,0.])
    removal=causal.response_metrics(baseline,clean,np.array([.5,0.]))
    assert removal['signed_response_removal']==.5 and removal['remaining_energy_ratio']==.25
    assert removal['orthogonal_change_ratio']==0
    orthogonal=causal.response_metrics(baseline,clean,np.array([1.,-1.]))
    assert orthogonal['signed_response_removal']==0 and orthogonal['orthogonal_change_ratio']==1
    amplify=causal.response_metrics(baseline,clean,np.array([1.5,0.]))
    assert amplify['signed_response_removal']==-.5 and amplify['remaining_energy_ratio']==2.25
    zero=causal.response_metrics(baseline,baseline,np.ones(2))
    assert not zero['eligible_response'] and zero['signed_response_removal'] is None
    for r in (removal,orthogonal,amplify):
        assert r['remaining_energy_ratio']==pytest.approx(1-2*r['signed_response_removal']+r['changed_energy_ratio'])


def test_source_bootstrap_exclusions_and_equal_type_aggregation():
    result=analysis.summarize([1.,None,3.],['a','small','b'])
    assert result==dict(n_sources=2,source_ids=['a','b'],excluded_sources=['small'],mean=2.,ci95=[1.,3.])
    records=[dict(kind='ramprecent64',value=1.),dict(kind='noise',value=3.),dict(kind='sin24',value=1e9)]
    assert analysis.aggregate(records,['ramprecent64','noise'],'value')==2
    records[1]['value']=None
    assert analysis.aggregate(records,['ramprecent64','noise'],'value') is None


def stage1_fixture(root):
    """Audited-shape synthetic fixture with ample known FIT activity matches."""
    source=root/'ffn_selectivity_pilot4961000';source.mkdir()
    rows=[];queries=[]
    for i in range(24):
        seed=4961000+i
        rows.append(dict(source=f'syn:{seed}',seed=seed,kind='synthetic' if i<16 else 'real_reused',
            split='FIT' if i<8 or 16<=i<20 else 'TEST'))
        x=causal.stage1.synthetic_background(seed,i%4);delta,conditions=causal.stage1.perturbations(seed)
        queries.append(np.concatenate([x[None],x[None]+delta]).astype(np.float32))
    np.savez(source/'inputs.npz',queries=np.asarray(queries))
    causal.write(source/'rows.json',rows);causal.write(source/'conditions.json',conditions)
    rng=np.random.default_rng(3)
    raw=rng.uniform(.1,.3,size=(24,15,2,2048)).astype(np.float32)
    np.savez(source/'bolt_captures.npz',first_ffn=raw,source_ids=[r['source'] for r in rows])
    causal.write(source/'bolt_worker.json',dict(valid=True,sha256=causal.sha(source/'bolt_captures.npz')))
    records=[dict(model='bolt',site='first',space='ffn',basis='native',view='rms',budget=k,
        FIT_rule=dict(indices=[1925,909,108,901][:k])) for k in (1,4)]
    causal.write(source/'analysis.json',dict(records=records))
    causal.write(source/'independent_audit.json',dict(valid=True,analysis_sha256=causal.sha(source/'analysis.json')))
    causal.write(source/'execution.json',dict(bolt=dict(seconds=37.56228056701366)))
    causal.write(source/'causal_plan.json',dict(budgets=[1,4]))
    causal.write(source/'protocol.json',dict(sources={},hashes={},checkpoints={'bolt':dict(path='',weights={})},
        capture={'bolt':dict(sites=[dict(tokens=2,ffn_width=2048)])}))
    return source


def test_preparation_freezes384_cells_balanced_fresh_sources_and_no_native(tmp_path):
    stage1_fixture(tmp_path)
    out=tmp_path/'ffn_selectivity_causal4963000'
    causal.prepare(out)
    p=causal.read(out/'protocol.json');rows=causal.read(out/'rows.json')
    assert p['cells']==384 and p['model']=='bolt' and p['hold_native']
    budget=causal.read(out/'budget.json')
    assert budget['fresh_cells']==152 and budget['reused_cells']==232
    assert len(rows)==len({r['source'] for r in rows})==16
    assert [r['seed'] for r in rows[8:]]==list(range(4963000,4963008))
    assert all(r['budgets']==[4] for r in rows[8:]) and all(r['budgets']==[1,4] for r in rows[:8])
    for group in (rows[:8],rows[8:]):
        assert sorted(r['control_draw'] for r in group)==list(range(8))
        assert [sum(r['condition_indices'][0]==i for r in group) for i in range(4)]==[2]*4
    assert not (out/'execution.json').exists()
    with pytest.raises(FileExistsError):
        causal.prepare(out)


def test_toy_worker_exact_delta_support_parameter_hash_and_saved_analysis(tmp_path,monkeypatch):
    rows=[];queries=[]
    for i,group in enumerate(('reused_TEST','fresh_confirmation')):
        x=causal.stage1.synthetic_background(55+i,0);delta,conditions=causal.stage1.perturbations(55+i)
        ix=causal.condition_indices(i)
        queries.append(np.concatenate([x[None],x[None]+delta[ix]]).astype(np.float32))
        rows.append(dict(index=i,source=f'toy:{i}',group=group,control_draw=i,budgets=[1,4] if i==0 else [4]))
    np.savez(tmp_path/'inputs.npz',queries=np.asarray(queries));causal.write(tmp_path/'rows.json',rows)
    causal.write(tmp_path/'matching.json',dict(selected=[0,1,2,3],draws=[[4,5,6,7]]*8))
    p=dict(sources={},hashes={},checkpoint=dict(path='',weights={}),cells=48,capture=dict(tokens=4,ffn_width=8),
        memory_cap=3*2**30,seconds_cap=900,budgets=dict(reused_TEST=[1,4],fresh_confirmation=[4]),
        target_types={'1':['ramprecent64'],'4':['ramprecent64','noise']})
    causal.write(tmp_path/'protocol.json',p)
    monkeypatch.setenv('FFN_CAUSAL_MONITORED','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for key in causal.stage1.THREAD_VARS:
        monkeypatch.setenv(key,'1')
    def toy_factory(name,checkpoint):
        assert name=='bolt'
        return ToyNative('timesfm','')
    causal.worker(tmp_path,native_factory=toy_factory)
    worker=causal.read(tmp_path/'bolt_worker.json')
    assert worker['cells']==ToyNative.instances[-1].calls==48
    assert worker['parameter_hash_before']==worker['parameter_hash_after']
    causal.write(tmp_path/'execution.json',dict(bolt=dict(valid=True,returncode=0,monitored_peak=worker['peak_rss'],seconds=worker['seconds'])))
    result=analysis.analyze(tmp_path)
    assert result['gates']['patch_support_independently_verified']
    assert result['primary']['group']=='fresh_confirmation' and result['primary']['budget']==4
    assert result['primary']['paired_selected_minus_matched']['ci95'] is None
    with pytest.raises(FileExistsError):
        causal.worker(tmp_path,native_factory=toy_factory)
    drift=tmp_path/'parameter_drift';drift.mkdir()
    for name in ('protocol.json','inputs.npz','rows.json','matching.json'):
        shutil.copyfile(tmp_path/name,drift/name)
    hashes=iter(['before','changed'])
    monkeypatch.setattr(causal,'parameter_hash',lambda model:next(hashes))
    with pytest.raises(ValueError,match='Parameters changed'):
        causal.worker(drift,native_factory=toy_factory)


def test_hold_and_Bolt_only_budget_gate_prevent_subprocess(tmp_path,monkeypatch):
    def forbid(*args,**kwargs):
        raise AssertionError('Native launch forbidden in tests')
    monkeypatch.setattr(runner.subprocess,'Popen',forbid)
    with pytest.raises(RuntimeError,match='HOLD'):
        runner.run(tmp_path)
    monkeypatch.delenv('FFN_CAUSAL_MONITORED',raising=False)
    with pytest.raises(RuntimeError,match='HOLD'):
        causal.worker(tmp_path,native_factory=forbid)
    for model,cells in (('timesfm',384),('bolt',385)):
        causal.write(tmp_path/'protocol.json',dict(model=model,cells=cells))
        with pytest.raises(ValueError,match='Bolt schedule'):
            runner.run(tmp_path,slot_granted=True)
