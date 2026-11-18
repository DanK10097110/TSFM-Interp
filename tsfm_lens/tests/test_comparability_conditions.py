"""Known-answer prelaunch tests for4962000; no native model is imported or called."""
import json
import subprocess
import sys
import numpy as np
import pytest
from experiments.feature_search import comparability_conditions as r
from experiments.feature_search import comparability_core as c


def primary_rows():
    """Thirty-two independent sources, each with five dependent conditions."""
    return [dict(source=f's{i}',case=f'c{j}') for i in range(32) for j in range(5)]


def test_namespace_and_waveforms():
    """Check fresh namespace and independently reconstruct every first-source condition."""
    rows,x,y=r.generate()
    assert len(rows)==260 and len({a['source'] for a in rows})==52
    assert sorted({a['seed'] for a in rows})==list(range(4962000,4962052))
    assert {a['seed'] for a in rows if a['split']=='fit'}==set(range(4962000,4962020))
    assert {a['seed'] for a in rows if a['split']=='test'}==set(range(4962020,4962052))
    t=np.arange(-128,32);rng=np.random.default_rng(4962000);phi,psi=rng.uniform(-np.pi,np.pi,2);noise=rng.normal(0,.05,len(t))
    expected=[(5.35,17.35,8.),(5.35,17.35,2.),(7.35,23.35,8.),(7.35,23.35,2.),(7.35,23.35,.5)]
    for i,(a,b,amp) in enumerate(expected):
        value=.5*np.sin(2*np.pi*a*t/128+phi)+amp*np.sin(2*np.pi*b*t/128+psi)+noise
        np.testing.assert_array_equal(x[i],value[:128]);np.testing.assert_array_equal(y[i],value[128:])


def test_exact_local_budget_and_whole_sources():
    """Only the declared four FIT rows are omitted; no source crosses FIT/TEST."""
    rows,_,_=r.generate();fit,test=r.fit_ids(rows)
    assert len(fit)==96 and len(test)==160
    omitted=[a for a in rows if a['split']=='fit' and not a['fit_eligible']]
    assert [(a['seed'],a['case']) for a in omitted]==[(i,'shifted_balanced') for i in range(4962000,4962004)]
    assert all(a['split']=='fit' for a in [rows[i] for i in fit])
    broken=[dict(a) for a in rows];replace=broken[test[0]]['source']
    for row in broken:
        if row['source']==replace:row['source']=broken[fit[0]]['source']
    with pytest.raises(AssertionError):r.fit_ids(broken)


def test_primary_both_families_and_fixed_coverage():
    """One passing family cannot rescue a failing family; source n is32, not160."""
    rows=primary_rows();trained=np.full(160,.9);random={f:np.ones((3,160)) for f in r.FAMILIES}
    good=r.primary_rule(trained,random,rows)
    assert good['passed'] and good['coverage']==.975
    assert all(v['gain']['n_sources']==32 for v in good['families'].values())
    random['mlp_relu']=np.full((3,160),.8)
    bad=r.primary_rule(trained,random,rows)
    assert bad['families']['rff']['passed'] and not bad['families']['mlp_relu']['passed'] and not bad['passed']


def test_primary_five_percent_random_denominator():
    """Positive intervals below5% fail; denominator is random risk, not trained risk."""
    rows=primary_rows();random={f:np.ones((3,160)) for f in r.FAMILIES}
    assert not r.primary_rule(np.full(160,.951),random,rows)['passed']
    assert r.primary_rule(np.full(160,.95),random,rows)['passed']
    zeros={f:np.zeros((3,160)) for f in r.FAMILIES}
    assert not r.primary_rule(np.zeros(160),zeros,rows)['passed']


def test_primary_interval_and_initialization_average():
    """A large outlier cannot bypass uncertainty; never pick a favorable initialization."""
    rows=primary_rows();v=np.repeat([4.]+[.99]*31,5);random={f:np.repeat(v[None],3,axis=0) for f in r.FAMILIES}
    result=r.primary_rule(np.ones(160),random,rows)
    assert result['families']['rff']['relative_reduction']>.05
    assert result['families']['rff']['gain']['low']<0 and not result['passed']
    random={f:np.array([np.full(160,.5),np.full(160,2.),np.full(160,2.)]) for f in r.FAMILIES}
    result=r.primary_rule(np.full(160,1.4),random,rows)
    assert result['passed']
    assert result['families']['rff']['random_mean_loss']==1.5


def test_response_energy_is_not_correctness():
    """A strong but wrong response must have positive energy and negative correctness gain."""
    history=np.tile(np.arange(128,dtype=float),(4,1));future=np.zeros((4,32));native=np.ones((4,32));variance=np.var(history,axis=1)
    wrong=r.behavior_metrics(history,future,native,2*native)
    np.testing.assert_allclose(wrong['response_energy'],1/variance)
    np.testing.assert_allclose(wrong['correctness_gain'],-3/variance)
    right=r.behavior_metrics(history,future,native,future)
    np.testing.assert_allclose(right['correctness_gain'],1/variance)
    zero=r.behavior_metrics(history,future,native,native)
    np.testing.assert_array_equal(zero['response_energy'],0)


def test_old_or_missing_slot_grant_is_rejected(tmp_path):
    """Prior slot2 permission cannot authorize this new protocol."""
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(RuntimeError):r.require_grant(tmp_path)
    for slot,digest in [(2,'prior-study'),(3,r.sha(tmp_path/'protocol.json'))]:
        (tmp_path/'slot_grant.json').write_text(json.dumps(dict(granted=True,slot=slot,protocol_sha256=digest)))
        with pytest.raises(RuntimeError):r.require_grant(tmp_path)
    for slot in (1,2):
        (tmp_path/'slot_grant.json').write_text(json.dumps(dict(granted=True,slot=slot,protocol_sha256=r.sha(tmp_path/'protocol.json'))))
        r.require_grant(tmp_path)


def test_readiness_binds_tests_and_plants_to_protocol(tmp_path):
    """Passed artifacts from another protocol and failed controls cannot launch."""
    (tmp_path/'protocol.json').write_text('{}');digest=r.sha(tmp_path/'protocol.json')
    for name in ('validation.json','plant_audit.json'):(tmp_path/name).write_text(json.dumps(dict(passed=True,protocol_sha256=digest)))
    r.require_ready(tmp_path)
    for name in ('validation.json','plant_audit.json'):
        for record in [dict(passed=False,protocol_sha256=digest),dict(passed=True,protocol_sha256='old')]:
            (tmp_path/name).write_text(json.dumps(record))
            with pytest.raises(RuntimeError):r.require_ready(tmp_path)
        (tmp_path/name).write_text(json.dumps(dict(passed=True,protocol_sha256=digest)))


def test_local_selection_ignores_test_labels():
    """Known latent signal is recoverable while held-out labels cannot choose rank/alpha."""
    rng=np.random.default_rng(49);h=rng.normal(size=(100,6));x=rng.normal(size=(100,4));y=h[:,:1]
    rows=[dict(source=f's{i}',domain='synthetic') for i in range(100)];tr=np.arange(72);te=np.arange(72,100);banks=[]
    for t,v in c.folds(rows,tr):
        w=c.weights(rows,t);base=c.pca_kernel(x@x.T,t,w,4);extra={rank:c.pca_kernel(h@h.T,t,w,rank) for rank in (8,16)};banks.append((t,v,base,extra))
    choice=r.choose_added(banks,y,rows);assert len(choice['cv_losses'])==8
    changed=y.copy();changed[te]=1e9;assert r.choose_added(banks,changed,rows)==choice
    w=c.weights(rows,tr);k=(2*c.pca_kernel(x@x.T,tr,w,4)+c.pca_kernel(h@h.T,tr,w,choice['rank']))/3
    prediction=c.ridge(k,y,tr,te,w,choice['alpha'])
    assert np.mean((prediction-y[te])**2)<.01


def test_prepare_fixed_base_and_no_native(tmp_path):
    """Preparation freezes thresholds/budgets and imports no native framework."""
    out=tmp_path/'prepared';r.prepare(out);p=r.read(out/'protocol.json');r.verify(out)
    fixed=r.base_specs(p,'timesfm','bolt','fusion_response')
    assert fixed['input+A_forecast']['kernel']=='rbf0.5+pca16'
    assert fixed['input+A_forecast']['alpha']==.001
    assert p['budget']['ranks']==[8,16] and p['budget']['alphas']==[.001,.01,.1,1.]
    assert p['budget']['initialization_seeds']==[4953000,4953001,4953002]
    assert p['budget']['widths']==dict(bolt=4608,timesfm=5120)
    assert p['primary']['family_size']==2 and p['primary']['minimum_relative_reduction']==.05
    assert p['primary']['paired_source_CI_coverage']==.975
    assert p['natural_leg'].startswith('NONE') and not (out/'slot_grant.json').exists()
    code='import sys; import experiments.feature_search.comparability_conditions; assert "torch" not in sys.modules; assert "chronos" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_mock_pipeline_keeps_fit_and_axes_separate(tmp_path):
    """Exercise all analysis paths using invented captures, without any native inference."""
    out=tmp_path/'mock';r.prepare(out);digest=r.sha(out/'protocol.json')
    for name in ('validation.json','plant_audit.json'):(out/name).write_text(json.dumps(dict(passed=True,protocol_sha256=digest)))
    with np.load(out/'inputs.npz') as f:x=f['history'].astype(float)
    mean=x.mean(1);sd=x.std(1);z=(x-mean[:,None])/sd[:,None]
    for model,shape,factor in [('bolt',(9,512),.6),('timesfm',(4,1280),.3)]:
        hidden=np.tile(z[:,:16],(1,np.prod(shape)//16)).reshape((len(x),)+shape).astype(np.float32)
        native=mean[:,None]+sd[:,None]*factor*z[:,-32:]
        fusion=native+.05*sd[:,None]*z[:,-32:]
        dest=out/f'{model}_native.npz';np.savez_compressed(dest,hidden=hidden,native=native,fusion=fusion)
        (out/f'{model}_worker.json').write_text(json.dumps(dict(sha256=r.sha(dest))))
    r.fit_local(out)
    fitrows=r.read(out/'local_fit_rows.json')
    assert len(fitrows)==96 and len({v['source'] for v in fitrows})==20
    assert all(v['split']=='fit' and v['fit_eligible'] for v in fitrows)
    assert len(r.read(out/'local_probe_specs.json'))==27
    r.score_local(out);r.score_frozen(out)
    local=r.read(out/'local_results.json');frozen=r.read(out/'frozen_extrapolation.json')
    assert len(local)==len(frozen)==18
    assert {v['axis'] for v in local}=={'local_FIT'}
    assert {v['axis'] for v in frozen}=={'frozen_extrapolation_descriptive'}
    assert all(m['n_sources']==32 for v in local+frozen for m in v['losses'].values())
    assert r.read(out/'primary_result.json')['coverage']==.975
    assert not (out/'slot_grant.json').exists()
