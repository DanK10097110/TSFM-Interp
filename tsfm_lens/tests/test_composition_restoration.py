"""Anchoring, gauge, numerical-stop and complete-family restoration decoys."""
import numpy as np
import pytest
from experiments.feature_search.composition_restoration import (
    anchored,random_edits,affine_remainder,nonlinear_remainder,numerical_gate,qualify,CANDIDATES)


def test_anchoring_is_affine_coordinate_invariant_and_self_identity():
    """Translation/nonorthogonal scaling preserve anchored composition exactly."""
    rng=np.random.default_rng(3)
    h,a,b,z=[rng.normal(size=(2,4,3)) for _ in range(4)]
    sm=np.array([2.,1.]);sa=np.array([1.,.8]);sb=np.array([1.5,.6])
    weight=anchored(h,a,b,z,sm,sa,sb)
    matrix=np.array([[1.,2.,0.],[0.,1.,.5],[.3,0.,2.]])
    offset=np.array([2.,-3.,1.])
    transform=lambda x:x@matrix+offset
    np.testing.assert_allclose(anchored(transform(h),transform(a),transform(b),transform(z),sm,sa,sb),transform(weight),atol=1e-14)
    np.testing.assert_allclose(anchored(h,h,z,z,sm,sm,np.zeros(2)),h,atol=1e-14)


def test_exact_affine_B_and_nonlinear_suffix_remainder():
    """Bias/mean correction remains explicit rather than an assumed head ceiling."""
    a=np.array([[1.,2.]]);b=np.array([[3.,1.]]);zero=np.array([[.2,-.1]])
    scales=dict(baseline=np.array([2.]),pure_phi=np.array([1.]),pure_psi=np.array([1.5]),background=np.array([.1]))
    means=dict(baseline=np.array([.3]),pure_phi=np.array([.2]),pure_psi=np.array([.4]),background=np.array([.1]))
    state=anchored(a,a,b,zero,scales['baseline'],scales['pure_phi'],scales['pure_psi'])
    for function in [lambda x:2*x+.7,lambda x:x*x+.7]:
        combined=2*function(state)+.3
        target=function(a)+.2+1.5*function(b)+.4-.1*function(zero)-.1
        B=affine_remainder(scales,means,function(zero))
        nonlinear=nonlinear_remainder(function,state,a,b,zero,(np.array([.5]),np.array([.75])),np.array([2.]))
        np.testing.assert_allclose(combined-target,B+nonlinear,atol=1e-14)
    np.testing.assert_allclose(nonlinear_remainder(lambda x:2*x+.7,state,a,b,zero,(np.array([.5]),np.array([.75])),np.array([2.])),0,atol=1e-14)


def test_nonlinear_invertible_encoding_can_fail_early_addition_with_perfect_model():
    """A restoration-compatible late cut cannot uniquely locate harmful layers."""
    encode=lambda x:x+x**3
    inverse=lambda y:2/np.sqrt(3)*np.sinh(np.arcsinh(3*np.sqrt(3)*y/2)/3)
    a=np.array([[.7]]);b=np.array([[1.1]]);zero=np.zeros_like(a);scale=np.ones(1)
    np.testing.assert_allclose(inverse(encode(a+b)),a+b,atol=1e-14)
    early=anchored(encode(a+b),encode(a),encode(b),encode(zero),scale,scale,scale)
    assert np.linalg.norm(inverse(early)-(a+b))>.1
    late=anchored(a+b,a,b,zero,scale,scale,scale)
    np.testing.assert_allclose(late,a+b,atol=1e-14)


def test_random_controls_match_each_token_radius_and_preserve_homogeneity():
    """Whole-state radius is insufficient when native RMSNorm acts per token."""
    rng=np.random.default_rng(13)
    h=rng.normal(size=(3,4,8));edit=rng.normal(size=h.shape)
    controls,flags=random_edits(h,edit,rng,2)
    for value in controls:
        np.testing.assert_allclose(np.sum(value**2,-1),np.sum(edit**2,-1),atol=1e-12)
        np.testing.assert_allclose(np.sum(value*h,-1),np.sum(edit*h,-1),atol=1e-12)
        np.testing.assert_allclose(np.sum((h+value)**2,-1),np.sum((h+edit)**2,-1),atol=1e-12)
    assert flags.shape==(3,)
    for seed in range(len(h)):
        for value in controls:
            np.testing.assert_allclose(value[seed]@value[seed].T,edit[seed]@edit[seed].T,atol=1e-11)
            np.testing.assert_allclose(value[seed]@h[seed].T,edit[seed]@h[seed].T,atol=1e-11)
            np.testing.assert_allclose((h[seed]+value[seed])@(h[seed]+value[seed]).T,(h[seed]+edit[seed])@(h[seed]+edit[seed]).T,atol=1e-11)
    same_h=np.repeat(h[:,:1],4,axis=1);same_edit=np.repeat(edit[:,:1],4,axis=1)
    homogeneous,_=random_edits(same_h,same_edit,rng,2)
    for value in homogeneous:np.testing.assert_allclose(value,np.repeat(value[:,:1],4,axis=1),atol=1e-12)
    _,flags=random_edits(h,.2*h,rng,2)
    assert flags.all()
    zero,flags=random_edits(np.zeros_like(h),np.zeros_like(edit),rng)
    assert flags.all()
    np.testing.assert_equal(zero,0)


def test_numerical_anchor_gate_requires_every_candidate_and_late_cut():
    """One unstable actual-background anchor stops rather than replacing/excluding it."""
    checks=dict(transport={'0':0.},normalized_transport={'0':0.},B_invariance={'0':0.},identity_closure={'0':0.},raw_input_waveform_RMS=[0.],
        raw_input_state_RMS=[0.],self_patch_state={'0':0.},maximum_anchor_sensitivity=.009,anchor_sensitivity={'0':[.009],'20':[0.]})
    assert numerical_gate(checks)
    checks['anchor_sensitivity']['20']=[.0101]
    assert not numerical_gate(checks)
    checks['anchor_sensitivity']['20']=[0.]
    checks['anchor_sensitivity']['0']=[0.]
    checks['B_invariance']['20']=.0001
    assert not numerical_gate(checks)
    checks['B_invariance']['20']=0.
    checks['normalized_transport']['20/background']=2e-5
    assert not numerical_gate(checks)


def test_complete_Holm8_point_gates_and_only_earliest_candidate_selection():
    """Tiny effects, small ceilings and casewise p cannot rescue a full-family failure."""
    native=np.ones(16);component=np.ones(16)*.1
    edits={cut:np.ones(16)*.2 for cut in CANDIDATES}
    random={cut:np.ones(16)*.5 for cut in CANDIDATES}
    suffix={cut:np.ones(16)*.01 for cut in CANDIDATES}
    result=qualify(native,component,edits,random,suffix)
    assert result['family_size']==8 and result['earliest_qualifying_cut']==0
    identifiable={c:np.ones(16,dtype=bool) for c in CANDIDATES}
    identifiable[0][3]=False
    blocked=qualify(native,component,edits,random,suffix,identifiable)
    assert blocked['earliest_qualifying_cut']==5
    assert blocked['claims']['0/random_advantage']['p']==1
    assert 'untestable' in blocked['claims']['0/random_advantage']
    edits[0]=np.ones(16)*.99
    result=qualify(native,component,edits,random,suffix)
    assert result['earliest_qualifying_cut']==5
    assert not result['claims']['0/native_gain']['passed']
    small=qualify(np.ones(16)*.05,np.zeros(16),{c:np.zeros(16) for c in CANDIDATES},random,suffix)
    assert small['earliest_qualifying_cut'] is None
    n=8
    result=qualify(native[:n],component[:n],{c:edits[5][:n] for c in CANDIDATES},
                   {c:random[c][:n] for c in CANDIDATES},{c:suffix[c][:n] for c in CANDIDATES})
    assert result['earliest_qualifying_cut'] is None
    assert result['claims']['0/native_gain']['p']<.05
    assert result['claims']['0/native_gain']['p_holm']>=.05


def test_native_capture_disables_autograd_and_keeps_normalized_head_reference():
    """A zero-scale physical output must not hide an incorrect normalized head."""
    import torch
    from experiments.feature_search.composition_restoration import native_capture,suffix
    class Layer(torch.nn.Module):
        def forward(self,x,mask,cache=None):return x+.1,None
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer=torch.nn.Linear(64,8)
            self.stacked_xf=torch.nn.ModuleList([Layer() for _ in range(20)])
            self.output_projection_point=torch.nn.Linear(8,1280)
            self.aridx=5
        def decode(self,horizon,inputs,masks):
            assert not torch.is_grad_enabled()
            z=inputs.reshape(len(inputs),4,32)
            h=self.tokenizer(torch.cat([z,torch.zeros_like(z)],-1))
            for layer in self.stacked_xf:h,_=layer(h,None,None)
            return self.output_projection_point(h).reshape(len(h),4,128,10),None,None
    model=Tiny()
    prediction,states=native_capture(model,np.ones((2,128),dtype=np.float32))
    expected=states['normalized_head'][:,-1].reshape(2,128,10)[:,:64,5]
    np.testing.assert_allclose(prediction,expected,atol=1e-7)
    np.testing.assert_allclose(suffix(model,states[5],5),expected,atol=1e-7)
    assert states[5].shape==(2,4,8)


def test_raw_input_anchor_exposes_mean_chart_residual():
    """At raw input, composing centered components reproduces mixture up to means."""
    a=np.array([[[1.,2.,3.]]]);b=np.array([[[.4,.7,.9]]]);x0=np.ones_like(a)*.1
    mm,ma,mb,m0=1.2,1.,.5,.1
    sm,sa,sb=2.,1.,1.5
    mixed=(a+b-x0-mm)/sm
    first=(a-ma)/sa;second=(b-mb)/sb;zero=np.zeros_like(a)
    composed=anchored(mixed,first,second,zero,np.array([sm]),np.array([sa]),np.array([sb]))
    np.testing.assert_allclose(composed-mixed,(mm-ma-mb+m0)/sm,atol=1e-14)


def test_random_geometry_runtime_validation_and_rank_failure_are_visible():
    """Failed QR references are flagged, never retried or treated as random zeros."""
    from experiments.feature_search.composition_restoration import geometry_errors
    rng=np.random.default_rng(81)
    h=rng.normal(size=(2,4,12));edit=rng.normal(size=h.shape)
    controls,flags,diagnostics=random_edits(h,edit,rng,return_diagnostics=True)
    assert not flags.any()
    assert all(max(row['float64'].values())<1e-8 and max(row['injected_float32'].values())<1e-5 for seed in diagnostics for row in seed)
    damaged=edit[0].copy();damaged[0]*=2
    assert geometry_errors(h[0],edit[0],damaged)['edit_gram']>.01
    class ZeroRng:
        def normal(self,size):return np.zeros(size)
    _,failed,details=random_edits(h,edit,ZeroRng(),return_diagnostics=True)
    assert failed.all()
    assert any('rank-deficient' in row.get('reason','') for seed in details for row in seed)


def test_point_native_margin_and_small_aggregate_ceiling_are_separate_gates():
    """A large relative recovery cannot replace either declared absolute margin."""
    suffix={c:np.ones(16)*.01 for c in CANDIDATES}
    random={c:np.ones(16)*.3 for c in CANDIDATES}
    insufficient=qualify(np.ones(16)*.2,np.ones(16)*.1,{c:np.ones(16)*.12 for c in CANDIDATES},random,suffix)
    assert insufficient['earliest_qualifying_cut'] is None
    assert insufficient['claims']['0/native_gain']['p_holm']<.05
    small=qualify(np.ones(16)*.2,np.ones(16)*.15,{c:np.zeros(16) for c in CANDIDATES},random,suffix)
    assert small['earliest_qualifying_cut'] is None
    assert small['claims']['0/native_gain']['passed']


def test_resource_and_numerical_failures_override_conditional_selection():
    from experiments.feature_search.composition_restoration import effective_cut
    result=dict(numerical_checks=dict(passed=True),pilot=dict(selection=dict(earliest_qualifying_cut=15)))
    assert effective_cut(result)==15
    result['resource_limit_exceeded']=True
    assert effective_cut(result) is None
    result['resource_limit_exceeded']=False
    result['numerical_checks']['passed']=False
    assert effective_cut(result) is None


def test_model_and_state_bundle_released_before_statistic_import(monkeypatch,tmp_path):
    """The execution boundary must not keep weights alive during heavy analysis imports."""
    import json,weakref
    from experiments.feature_search import composition_restoration as worker
    class Model:pass
    observed=[]
    weights=tmp_path/'weights';weights.write_bytes(b'no model loaded')
    manifest=dict(checkpoint=str(weights),checkpoint_weight_sha256=worker.replay.digest(weights),source_sha256={},available_memory_kib=10000000)
    monkeypatch.setattr(worker.replay,'manifest',lambda *args:(None,None,None,weights,manifest))
    def load(_):
        model=Model();observed.append(weakref.ref(model));return model
    monkeypatch.setattr(worker.replay,'load_model',load)
    monkeypatch.setattr(worker,'calibrated_run',lambda model,*args:(dict(passed=True),dict(model_reference=model)))
    def forecasts(model,*args):return {},dict(example=np.zeros((1,64))),dict(example=1)
    monkeypatch.setattr(worker,'pilot_forecasts',forecasts)
    def statistics(**kwargs):
        assert observed[-1]() is None
        return dict(earliest_qualifying_cut=15)
    monkeypatch.setattr(worker,'qualify',statistics)
    calibration=tmp_path/'calibration';calibration.mkdir()
    declaration=dict(driver_sha256=worker.replay.digest(worker.__file__),helper_sha256=worker.replay.digest(worker.replay.__file__),
        checkpoint_weight_sha256=manifest['checkpoint_weight_sha256'],source_sha256={})
    (calibration/'protocol.json').write_text(json.dumps(declaration),encoding='utf-8')
    (calibration/'results.json').write_text(json.dumps(dict(numerical_checks=dict(passed=True))),encoding='utf-8')
    worker.run(tmp_path,tmp_path/'output','pilot',calibration)
    result=json.loads((tmp_path/'output/results.json').read_text(encoding='utf-8'))
    assert result['effective_earliest_cut']==15
    assert 'after_model_and_states_released' in result['memory_boundaries']


def test_execution_refactor_preserves_original_scientific_ast():
    """The resource fix moves statistics only; native operations remain identical."""
    import ast
    from pathlib import Path
    from experiments.feature_search import composition_restoration as worker
    root=Path(__file__).resolve().parents[2]
    original=root/'tsfm_lens/runs/feature_search_stage49_restoration_memory_diagnosis/frozen_resource_failed_driver.py'
    if not original.is_file():pytest.skip('Optional frozen producer source is retained with the local controlled experiment')
    functions=lambda p:{n.name:n for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    old,new=functions(original),functions(Path(worker.__file__))
    names=['protocol','anchored','random_edits','geometry_errors','affine_remainder','nonlinear_remainder','make_data','normalizers',
        'capture_hooks','suffix','native_capture','canonical_background','calibrated_run','numerical_gate','qualify']
    for name in names:assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False),name
    def body(function,removed):
        nodes=[n for n in function.body if not isinstance(n,ast.Return) and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==removed for t in n.targets))]
        return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
    assert body(old['pilot'],'selected')==body(new['pilot_forecasts'],'statistical_inputs')
