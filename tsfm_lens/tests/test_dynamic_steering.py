"""Unseen model discovery, horizon protection, context policies and safe persistence."""

import numpy as np
import json
import pytest
import torch

from tsfm_lens.steering import DynamicSteerer,SearchConfig,SteeringModel,SteeringRequest,SteeringUnavailable,TASKS
from tsfm_lens.steering.model import InterventionSite
from tsfm_lens.steering.requests import requested_change,context_counterfactual,window_mask


class UnseenForecaster(torch.nn.Module):
    """Arbitrary names and a numpy/no-grad public path; no custom adapter."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(5)
            self.obscure=torch.nn.Linear(16,24)
            self.banana=torch.nn.Linear(24,24)
            # Plant a well-conditioned reachable head for the positive test.
            torch.nn.init.orthogonal_(self.banana.weight)
        self.eval()

    def forward(self,x):
        return self.banana(torch.tanh(self.obscure(x))).reshape(-1,8,3)


def registered():
    module=UnseenForecaster()
    def predict(x,h):
        with torch.no_grad():return module(torch.from_numpy(x))[:,:h].numpy()
    return module,SteeringModel(predict,module=module,quantiles=[.1,.5,.9],model_id='unseen-v1')


def data():
    return np.random.default_rng(31).normal(size=(8,16)).astype('float32')


def config(**kwargs):
    return SearchConfig(max_candidates=3,rank=24,calibration_samples=1,validation_samples=1,
                        radius=.8,refinement_steps=2,**kwargs)


def test_unseen_no_grad_model_search_protects_far_horizon_and_replays():
    module,model=registered();steerer=DynamicSteerer(model,config())
    request=SteeringRequest('level',strength=.03,forecast_window='short')
    report=steerer.fit(data()[:6],request,prediction_length=8)
    result=steerer.steer(data()[6:],request,prediction_length=8,auto_discover=False)
    assert all(d['achieved'] for d in result.diagnostics)
    assert all(d['protected_relative_error']<.1 for d in result.diagnostics)
    assert np.allclose(steerer.replay(data()[6:],result),result.forecast)
    assert report[request.key(8)]['selection']=='separate calibration/validation'
    assert all(len(m._forward_pre_hooks)==0 for m in module.modules())


def test_profiles_roundtrip_and_reject_changed_model_weights(tmp_path):
    module,model=registered();steerer=DynamicSteerer(model,config())
    request=SteeringRequest('trend',strength=.02)
    steerer.fit(data()[:6],request,prediction_length=8)
    path=tmp_path/'profile';steerer.save(path)
    restored=DynamicSteerer.load(path,model)
    first=steerer.steer(data()[6:],request,prediction_length=8)
    second=restored.steer(data()[6:],request,prediction_length=8)
    assert np.array_equal(first.forecast,second.forecast)
    with torch.no_grad():module.banana.weight[0,0]+=.01
    with pytest.raises(SteeringUnavailable,match='weights differ'):
        DynamicSteerer.load(path,model)


def test_input_only_callable_uses_same_api_and_explicit_backend():
    model=SteeringModel(lambda x,h:x[:,-h:],model_id='private-service-v1')
    controller=DynamicSteerer(model,config())
    request=SteeringRequest('level',strength=.05,forecast_window='long')
    controller.fit(data(),request,prediction_length=8)
    result=controller.steer(data()[0:1],request,prediction_length=8)
    assert result.diagnostics[0]['achieved']
    assert result.diagnostics[0]['site']['source']=='input'
    assert np.allclose(result.forecast[:,:4],result.baseline[:,:4],atol=1e-5)


def test_no_response_is_a_reported_failure_not_fake_success():
    model=SteeringModel(lambda x,h:np.zeros((len(x),h)),model_id='constant')
    controller=DynamicSteerer(model,config())
    result=controller.steer(data()[:1],SteeringRequest('level',.2),prediction_length=8)
    assert not result.diagnostics[0]['achieved']
    assert result.diagnostics[0]['relative_error']==1
    assert 'same-context' in result.diagnostics[0]['selection']


def test_direct_context_policy_removes_selected_values_without_claiming_internal_control():
    model=SteeringModel(lambda x,h:np.repeat(x.mean(1,keepdims=True),h,axis=1))
    controller=DynamicSteerer(model,config())
    request=SteeringRequest('context_ignore',1.,context_window=(4,8))
    x=data()[:2];altered=x.copy();altered[:,4:8]+=100
    a=controller.steer(x,request,prediction_length=8)
    b=controller.steer(altered,request,prediction_length=8)
    assert np.array_equal(a.forecast,b.forecast)
    assert a.diagnostics[0]['site']['source']=='context_policy'
    assert a.diagnostics[0]['radius'] is None
    assert np.array_equal(controller.replay(x,a),a.forecast)


def test_actual_probe_batch_size_is_checked():
    def predict(x,h):return x[:,-h:]+(1. if len(x)>2 else 0.)
    model=SteeringModel(predict);controller=DynamicSteerer(model,config())
    r=SteeringRequest('level',.03)
    reports=controller.fit(data(),r,prediction_length=8)
    chosen=reports[r.key(8)]['chosen']
    assert chosen['details'][0]['probe_batch']==1
    assert controller.steer(data()[:1],r,prediction_length=8).diagnostics[0]['achieved']


def test_live_weight_update_and_replay_row_mismatch_are_rejected():
    module,model=registered();controller=DynamicSteerer(model,config())
    r=SteeringRequest('level',.03);result=controller.steer(data()[:1],r,prediction_length=8)
    with pytest.raises(ValueError,match='number of interventions'):
        controller.replay(data()[:2],result)
    with torch.no_grad():module.banana.weight.add_(.01)
    with pytest.raises(SteeringUnavailable,match='state changed'):
        controller.steer(data()[:1],r,prediction_length=8)


@pytest.mark.parametrize('kind',TASKS)
def test_all_request_types_protect_unselected_horizon(kind):
    x=np.sin(2*np.pi*np.arange(32)/8)
    point=np.linspace(0,1,16)+np.sin(2*np.pi*np.arange(16)/8)
    baseline=point[:,None]+np.array([-1.,0.,1.])
    r=SteeringRequest(kind,strength=.2,forecast_window=(4,11),period=8,value=.5,lower=-.3,upper=.5)
    target=requested_change(baseline,x,[.1,.5,.9],r,counterfactual_forecast=baseline+.1)
    assert target.shape==baseline.shape and np.isfinite(target).all()
    assert np.all(target[:4]==0) and np.all(target[11:]==0)


def test_ignore_reference_is_independent_of_ignored_values():
    x=np.arange(16,dtype=float);changed=x.copy();changed[4:9]+=100
    r=SteeringRequest('context_ignore',1.,context_window=(4,9))
    assert np.array_equal(context_counterfactual(x,r),context_counterfactual(changed,r))
    focused=SteeringRequest('context_focus',context_window=(4,9),outside_weight=0.)
    changed=x.copy();changed[:4]-=100;changed[9:]+=50
    assert np.array_equal(context_counterfactual(x,focused),context_counterfactual(changed,focused))


def test_ignored_history_invariance_with_input_control():
    model=SteeringModel(lambda x,h:np.repeat(x.mean(1)[:,None],h,axis=1),model_id='history-mean')
    controller=DynamicSteerer(model,config())
    request=SteeringRequest('context_ignore',1.,context_window=(4,8))
    controller.fit(data(),request,prediction_length=8)
    x=data()[0:1];changed=x.copy();changed[:,4:8]+=.2
    first=controller.steer(x,request,prediction_length=8)
    second=controller.steer(changed,request,prediction_length=8)
    assert np.linalg.norm(first.forecast-second.forecast)<.01*np.linalg.norm(first.baseline-second.baseline)


def test_invalid_splits_capabilities_and_windows_fail_loudly():
    _,model=registered();controller=DynamicSteerer(model,config())
    with pytest.raises(ValueError,match='identical context'):
        controller.fit(data()[:2],SteeringRequest('level'),prediction_length=8,validation_contexts=data()[:1])
    with pytest.raises(ValueError):window_mask((4,4),8)
    with pytest.raises(ValueError):window_mask((0,9),8)
    with pytest.raises(ValueError,match='quantile outputs'):
        requested_change(np.ones((8,1)),data()[0],[.5],SteeringRequest('uncertainty'))


def test_stochastic_predictor_is_rejected_before_search():
    rng=np.random.default_rng(3)
    model=SteeringModel(lambda x,h:rng.normal(size=(len(x),h)))
    with pytest.raises(SteeringUnavailable,match='stochastic'):
        DynamicSteerer(model,config()).fit(data(),SteeringRequest('level'),prediction_length=8)


def test_reused_module_occurrence_and_hook_cleanup_on_failure():
    class Reused(torch.nn.Module):
        def __init__(self):
            super().__init__();self.odd=torch.nn.Linear(8,8,bias=False)
            with torch.no_grad():self.odd.weight.copy_(torch.eye(8))
        def forward(self,x):return self.odd(x)+2*self.odd(x)
    module=Reused().eval()
    model=SteeringModel(lambda x,h:module(torch.from_numpy(x)).detach().numpy(),module=module)
    x=np.ones((1,8),dtype='float32');sites=model.trace(x,8)
    assert [s.occurrence for s in sites[:2]]==[0,1]
    assert sites[2].occurrences==[0,1] and sites[2].width==16
    z=np.zeros((1,8));z[0,0]=.01
    first=model.evaluate(x,8,sites[0],np.eye(8),z)
    second=model.evaluate(x,8,sites[1],np.eye(8),z)
    base=model.forecast(x,8)
    assert np.allclose(second-base,2*(first-base),atol=1e-6)
    joint=model.evaluate(x,8,sites[2],np.eye(16),np.concatenate([z,z],axis=1))
    assert np.allclose(joint-base,3*(first-base),atol=1e-6)
    invalid=InterventionSite('odd',['args',0],0,[9],'single')
    with pytest.raises(SteeringUnavailable):model.evaluate(x,8,invalid,np.eye(9),np.zeros((1,9)))
    assert len(module.odd._forward_pre_hooks)==0


def test_screening_rejects_executed_but_unused_branch():
    class Branches(torch.nn.Module):
        def __init__(self):
            super().__init__();self.good=torch.nn.Linear(16,16);self.dead=torch.nn.Linear(16,16)
            with torch.no_grad():self.good.weight.copy_(torch.eye(16))
        def forward(self,x):
            y=self.good(x);self.dead(x);return y
    module=Branches().eval()
    model=SteeringModel(lambda x,h:module(torch.from_numpy(x)).detach().numpy()[:,:h],module=module)
    cfg=SearchConfig(max_candidates=1,include_input=False,rank=16,calibration_samples=1,validation_samples=1)
    controller=DynamicSteerer(model,cfg);request=SteeringRequest('level',.05)
    report=controller.fit(data(),request,prediction_length=8)[request.key(8)]
    assert report['chosen']['site'].startswith('module_input:good:')
    assert any(r['normalized_signed_effect']==0 for r in report['screening'])
    assert controller.steer(data()[:1],request,prediction_length=8).diagnostics[0]['achieved']


def test_automatic_joint_control_of_expanded_repeated_invocations():
    class Rollout(torch.nn.Module):
        def __init__(self):
            super().__init__();self.readout=torch.nn.Linear(8,8,bias=False)
            with torch.no_grad():self.readout.weight.copy_(torch.eye(8))
        def forward(self,x):
            first=self.readout(x)
            branches=first.repeat_interleave(3,dim=0)
            second=self.readout(branches).reshape(len(x),3,8).mean(1)
            return torch.cat([first,second],dim=1)
    module=Rollout().eval()
    model=SteeringModel(lambda x,h:module(torch.from_numpy(x)).detach().numpy()[:,:h],module=module)
    x=data()[:,:8]
    controller=DynamicSteerer(model,SearchConfig(max_candidates=3,rank=16,include_input=False,radius=.2))
    request=SteeringRequest('trend',.1)
    report=controller.fit(x[:6],request,prediction_length=16)[request.key(16)]
    chosen=controller.profiles[request.key(16)].site
    assert chosen.occurrences==[0,1] and chosen.invocation_multipliers==[1,3]
    result=controller.steer(x[6:],request,prediction_length=16)
    assert all(d['achieved'] for d in result.diagnostics)
    assert np.allclose(controller.replay(x[6:],result),result.forecast)


def test_exact_linear_rowspace_avoids_invisible_directions_and_accepts_root_module():
    module=torch.nn.Linear(16,4,bias=False).eval()
    with torch.no_grad():module.weight.copy_(torch.eye(16)[:4])
    model=SteeringModel(lambda x,h:module(torch.from_numpy(x)).detach().numpy(),module=module)
    controller=DynamicSteerer(model,SearchConfig(rank=4,max_candidates=1,include_input=False))
    request=SteeringRequest('level',.05)
    report=controller.fit(data()[:6],request,prediction_length=4)[request.key(4)]
    profile=controller.profiles[request.key(4)]
    assert 'exact row space' in report['basis_source']
    assert np.all(profile.basis[:,4:]==0)
    assert all(d['achieved'] for d in controller.steer(data()[6:],request,prediction_length=4).diagnostics)


def test_legacy_profile_numeric_keys_are_migrated_and_corruption_rejected(tmp_path):
    _,model=registered();controller=DynamicSteerer(model,config())
    request=SteeringRequest('seasonality',.03,period=8)
    controller.fit(data()[:6],request,prediction_length=8)
    path=tmp_path/'profile';controller.save(path)
    metadata=json.loads((path/'profiles.json').read_text())
    metadata['version']=1;metadata['fingerprint']=model.fingerprint(legacy=True)
    legacy=json.loads(metadata['profiles'][0]['key']);legacy['request']['period']=8
    metadata['profiles'][0]['key']=json.dumps(legacy,sort_keys=True)
    metadata['profiles'][0]['request']['period']=8
    (path/'profiles.json').write_text(json.dumps(metadata))
    restored=DynamicSteerer.load(path,model)
    assert request.key(8) in restored.profiles
    assert np.array_equal(controller.steer(data()[6:],request,prediction_length=8).forecast,
                          restored.steer(data()[6:],request,prediction_length=8,auto_discover=False).forecast)
    with (path/'profiles.npz').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        DynamicSteerer.load(path,model)


def test_inference_budget_returns_best_verified_result_with_failure_status():
    model=SteeringModel(lambda x,h:np.zeros((len(x),h)))
    controller=DynamicSteerer(model,config());request=SteeringRequest('level',.1)
    controller.fit(data(),request,prediction_length=8)
    controller.config.max_calls=3
    result=controller.steer(data()[:1],request,prediction_length=8,auto_discover=False)
    assert result.diagnostics[0]['budget_exhausted'] and not result.diagnostics[0]['achieved']
    assert result.diagnostics[0]['model_calls']==3
    assert np.array_equal(result.forecast,result.baseline)


def test_new_request_reuses_learned_response_without_repeating_discovery():
    _,model=registered();controller=DynamicSteerer(model,config())
    controller.fit(data()[:6],SteeringRequest('level',.03),prediction_length=8)
    def forbidden(*args,**kwargs):raise AssertionError('fresh site search should not be needed')
    controller._candidates=forbidden
    request=SteeringRequest('trend',.02,forecast_window='long')
    result=controller.steer(data()[6:],request,prediction_length=8)
    assert all(d['achieved'] for d in result.diagnostics)
    assert all('online transfer' in d['selection'] and d['validation_error'] is None for d in result.diagnostics)


@pytest.mark.parametrize("reverse_mapping", [False, True])
def test_readout_low_rank_mixed_control_and_empirical_fallback(tmp_path, reverse_mapping):
    from tsfm_lens.steering import CombinedRequest

    _, model = registered()
    order = np.arange(24)
    model.readout_rows = lambda h, q: {"banana": order[::-1].copy() if reverse_mapping else order}
    limits = SearchConfig(max_candidates=1, include_input=False, module_pattern="banana",
                          rank=2, radius=.2, calibration_samples=2, validation_samples=2)
    controller = DynamicSteerer(model, limits)
    request = CombinedRequest((SteeringRequest("level", .01, forecast_window="short"),
                               SteeringRequest("trend", -.01, forecast_window="short")))
    report = controller.fit(data()[:4], request, prediction_length=8,
                            validation_contexts=data()[4:6])[request.key(8)]
    assert report["local_readout"] == (not reverse_mapping)
    assert report["rank"] == 2
    path = tmp_path / "profile"
    controller.save(path)
    restored = DynamicSteerer.load(path, model)
    result = restored.steer(data()[6:], request, prediction_length=8, auto_discover=False)
    for i, info in enumerate(result.diagnostics):
        assert info["coefficient_norm"] <= limits.radius + 1e-8
        if not reverse_mapping:
            assert info["achieved"]
            np.testing.assert_allclose(result.forecast[i,4:], result.baseline[i,4:], atol=1e-5)
        expected = sum(requested_change(result.baseline[i], data()[6+i], result.quantiles, term)
                       for term in request.terms)
        np.testing.assert_allclose(result.requested_change[i], expected)
    np.testing.assert_allclose(restored.replay(data()[6:], result), result.forecast)


def test_combined_request_rejects_ambiguous_context_composition():
    from tsfm_lens.steering import CombinedRequest
    with pytest.raises(ValueError, match="two forecast requests"):
        CombinedRequest((SteeringRequest("level"), SteeringRequest("context_focus")))


def test_validation_selection_preserves_edit_headroom_at_probe_precision():
    _, model = registered()
    controller = DynamicSteerer(model, config())
    costly = dict(errors=[2e-6, 4e-6], error=3e-6, mean_coefficient_norm=.19)
    economical = dict(errors=[6e-6, 8e-6], error=7e-6, mean_coefficient_norm=.03)
    assert controller._select_candidate([costly, economical]) is economical
    controller.config.tolerance = 4e-6
    assert controller._select_candidate([costly, economical]) is costly
    economical.update(errors=[0., .11], error=.055)
    controller.config.tolerance = .1
    assert controller._select_candidate([costly, economical]) is costly


@pytest.mark.parametrize("transform", ["identity", "scale", "mix"])
def test_unknown_wide_head_with_reordered_rows(transform):
    with torch.random.fork_rng():
        torch.manual_seed(14)
        module = torch.nn.Linear(16, 32).eval()
    selected = [7, 23, 1, 19]

    def predict(x, h):
        y = module(torch.from_numpy(x)).detach().numpy()
        result = y[:, selected]
        if transform == "scale":
            return result * np.array([1., 2., 3., 4.])
        return result + y[:, :4] if transform == "mix" else result

    model = SteeringModel(predict, module=module)
    controller = DynamicSteerer(model, SearchConfig(rank=4, include_input=False, max_candidates=1))
    request = SteeringRequest("level", .03)
    report = controller.fit(data()[:4], request, prediction_length=4,
                            validation_contexts=data()[4:6])[request.key(4)]
    assert report["local_readout"] == (transform != "mix")
    if transform == "scale":
        assert "row space" in report["basis_source"]
    if transform != "mix":
        result = controller.steer(data()[6:], request, prediction_length=4)
        assert all(r["achieved"] for r in result.diagnostics)
        np.testing.assert_allclose(controller.replay(data()[6:], result), result.forecast)
