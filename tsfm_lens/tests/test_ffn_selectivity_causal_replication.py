"""Crossed controls, practical-band restraint, fresh sources and toy-only worker."""
import numpy as np
import pytest
from experiments.feature_search import ffn_selectivity_causal_replication as study
from test_ffn_selectivity_study import ToyNative


def test_crossed_control_mean_uses_sources_not_control_replicates():
    controls=np.tile(np.array([-.01,.01,.03,.05]),(4,1))
    selected=np.full(4,.02)
    result=study.crossed_contrast(selected,controls,['a','b','c','d'])
    assert result['paired_primary']['n_sources']==4
    assert result['paired_primary']['mean']==pytest.approx(0,abs=1e-16)
    assert result['mean_control']['mean']==pytest.approx(.02)
    assert result['selected_absolute']['mean']==.02
    assert [v['mean'] for v in result['individual_controls']]==[-.01,.01,.03,.05]
    np.testing.assert_allclose(result['control_spread']['per_source_sd'],np.std(controls,axis=1,ddof=1))
    assert result['practical_check']['paired_primary']['CI_contained']


def test_practical_bound_requires_entire_CI_and_never_changes_primary():
    result=study.crossed_contrast(np.array([-.04,.04]),np.zeros((2,4)),['a','b'])
    assert result['paired_primary']['mean']==0 and result['paired_primary']['ci95']==[-.04,.04]
    assert not result['practical_check']['paired_primary']['CI_contained']
    assert not result['practical_check']['selected_absolute']['CI_contained']
    small=study.crossed_contrast(np.array([-.01,.01]),np.zeros((2,4)),['a','b'])
    assert small['practical_check']['paired_primary']['CI_contained']
    assert small['paired_primary']['mean']==0 and small['paired_primary']['ci95']==[-.01,.01]
    with pytest.raises(ValueError,match='four crossed'):
        study.crossed_contrast(np.zeros(2),np.zeros((2,1)),['a','b'])


def fixture_prior(root):
    """Minimal completed/audited prior run with known first-four control sets."""
    prior=root/'ffn_selectivity_causal4963000';prior.mkdir()
    stage1=root/'ffn_selectivity_pilot4961000';stage1.mkdir()
    study.write(stage1/'rows.json',[dict(source='syn:4961000')])
    study.write(prior/'rows.json',[dict(source='syn:4963000')])
    study.write(prior/'analysis.json',dict(frozen=True))
    study.write(prior/'independent_statistics_audit.json',dict(valid=True,analysis_sha256=study.sha(prior/'analysis.json')))
    study.write(prior/'compact_record.json',dict(status='COMPLETE_AUDITED_SLOT2_RELEASED'))
    study.write(prior/'execution.json',dict(bolt=dict(seconds=32.17311086098198)))
    matching=dict(selected=[1925,909,108,901],draws=[[4*j+k for k in range(4)] for j in range(8)],details=[])
    study.write(prior/'matching.json',matching)
    study.write(prior/'protocol.json',dict(sources={},hashes={'matching.json':study.sha(prior/'matching.json')},
        target_types={'4':['ramprecent64','noise']},checkpoint=dict(path='',weights={}),capture=dict(tokens=9,ffn_width=2048)))
    return matching


def test_prepare_freezes32fresh1088cells_and_first4_controls_every_source(tmp_path):
    matching=fixture_prior(tmp_path)
    out=tmp_path/'ffn_selectivity_causal_replication4964000'
    study.prepare(out)
    p=study.read(out/'protocol.json');rows=study.read(out/'rows.json');controls=study.read(out/'controls.json')
    assert p['cells']==1088 and p['cell_cap']==1200 and p['hold_native']
    assert p['units']==[1925,909,108,901] and p['targets']==['ramprecent64','noise']
    assert len(rows)==len({r['source'] for r in rows})==32
    assert p['synthetic_seeds']==[r['seed'] for r in rows]==list(range(4964000,4964032))
    assert controls['controls']==matching['draws'][:4]
    assert all(r['control_ids']==[0,1,2,3] and r['budget']==4 for r in rows)
    assert [sum(r['condition_indices'][0]==v for r in rows) for v in range(4)]==[8]*4
    for family in range(4):
        assert [sum(r['family']==family and r['condition_indices'][0]==v for r in rows) for v in range(4)]==[2]*4
    assert not (out/'execution.json').exists()


def test_toy_crossed_worker_saves_every_arm_and_float64_remaining_energy(tmp_path,monkeypatch):
    rows=[];queries=[]
    for i in range(2):
        x=study.causal.stage1.synthetic_background(74+i,0);delta,_=study.causal.stage1.perturbations(74+i)
        queries.append(np.concatenate([x[None],x[None]+delta[study.causal.condition_indices(i)]]).astype(np.float32))
        rows.append(dict(index=i,source=f'toy{i}'))
    np.savez(tmp_path/'inputs.npz',queries=np.asarray(queries));study.write(tmp_path/'rows.json',rows)
    study.write(tmp_path/'controls.json',dict(selected=[0,1,2,3],controls=[[4,5,6,7]]*4))
    p=dict(sources={},hashes={},checkpoint=dict(path='',weights={}),cells=68,capture=dict(tokens=4,ffn_width=8),
        types=list(study.causal.TYPES),targets=['ramprecent64','noise'],off_targets=['sin24','sin19.2','steprecent32'],
        response_floor=1e-8,practical_rule='Entire CI in±.02 only, no primary rescue')
    study.write(tmp_path/'protocol.json',p)
    monkeypatch.setenv('FFN_CROSSED_MONITORED','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    for key in study.causal.stage1.THREAD_VARS:
        monkeypatch.setenv(key,'1')
    study.worker(tmp_path,native_factory=lambda name,checkpoint:ToyNative('timesfm',''))
    worker=study.read(tmp_path/'bolt_worker.json')
    assert worker['cells']==ToyNative.instances[-1].calls==68
    assert worker['parameter_hash_before']==worker['parameter_hash_after']
    events=study.read(tmp_path/'interventions.json')
    assert len(events)==50
    assert {(e['source_index'],e['condition'],e['arm']) for e in events}=={(i,c,arm) for i in range(2) for c in range(5) for arm in study.ARMS}
    study.write(tmp_path/'execution.json',dict(bolt=dict(valid=True,returncode=0,monitored_peak=worker['peak_rss'],seconds=worker['seconds'])))
    result=study.analyze(tmp_path)
    assert result['primary']['paired_primary']['n_sources']==2
    arrays=np.load(tmp_path/'bolt_causal.npz');differences=[]
    for e in result['events']:
        base=arrays['clean_forecasts'][e['source_index'],0];patched=arrays['patched_forecasts'][e['index']]
        expected=float(np.mean((patched.astype(float)-base)**2))
        assert e['remaining_output_energy']==expected
        differences.append(abs(expected-float(np.mean((patched-base)**2))))
    assert max(differences)>1e-12
    assert all('practical_check' not in r for r in result['summaries'] if r['metric']!='signed_response_removal')


def test_replication_hold_wrong_model_and_cell_limit_prevent_native(tmp_path,monkeypatch):
    def forbid(*args,**kwargs):
        raise AssertionError('No subprocess permitted')
    monkeypatch.setattr(study.subprocess,'Popen',forbid)
    with pytest.raises(RuntimeError,match='HOLD'):
        study.run(tmp_path)
    monkeypatch.delenv('FFN_CROSSED_MONITORED',raising=False)
    with pytest.raises(RuntimeError,match='HOLD'):
        study.worker(tmp_path,native_factory=forbid)
    for model,cells in (('timesfm',1088),('bolt',1201)):
        study.write(tmp_path/'protocol.json',dict(model=model,cells=cells))
        with pytest.raises(ValueError,match='Bolt-only'):
            study.run(tmp_path,slot_granted=True)
