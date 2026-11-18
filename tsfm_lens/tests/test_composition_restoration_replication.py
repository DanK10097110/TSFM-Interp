"""Selected-cut replication guards; all model-free except a tiny hook fixture."""
import copy
import numpy as np
import pytest
from experiments.feature_search import composition_restoration_replication as study


def validity():
    return [dict(valid=True,shard_index=i,seed_ids=list(ids),numerical_checks=dict(passed=True),
        primary15_random_geometry_pass=True,model_released_before_return=True,peak_resident_bytes=100,
        wall_seconds=1.) for i,ids in enumerate(study.SHARDS)]


def arrays(native=1.,component=.8,edited=.6,random=(0.,2.),suffix=.04):
    wave=lambda loss:np.full((64,64),np.sqrt(loss))
    return dict(seed_ids=np.asarray(study.SEEDS,dtype=np.int64),native=wave(native),actual_component_sum=wave(component),
        truth=wave(0),common_scale=np.ones(64),cut15_anchored=wave(edited),cut15_random0=wave(random[0]),
        cut15_random1=wave(random[1]),cut15_suffix_remainder=wave(suffix),cut20_anchored=wave(0))


def test_keyed_random_draws_ignore_restart_order_and_shard_boundaries():
    rng=np.random.default_rng(1);h=rng.normal(size=(4,32));edit=rng.normal(size=(4,32))
    a=study.keyed_random(h,edit,3700000,15);study.keyed_random(h,edit,3700032,15)
    b=study.keyed_random(h,edit,3700000,15);c=study.keyed_random(h,edit,3700001,15)
    np.testing.assert_array_equal(a[0],b[0]);assert a[0].shape==(2,1,4,32)
    assert not np.array_equal(a[0],c[0])
    assert not a[1].any() and len(a[2][0])==2
    with pytest.raises(ValueError):study.keyed_random(h,edit,3700000,10)


def test_holm2_uses64_seeds_and_mean_of_losses_not_mean_predictions():
    result=study.pooled_decision(arrays(),validity())
    assert result['qualified'] and result['candidate_cut']==15 and result['family_size']==2
    assert set(result['claims'])=={'native_gain','random_advantage'}
    assert result['claims']['native_gain']['n_seeds']==64
    assert result['claims']['native_gain']['p']==pytest.approx(2.**-63,abs=0,rel=1e-12)
    assert result['claims']['native_gain']['p_holm']==pytest.approx(2.**-62,abs=0,rel=1e-12)
    assert result['mean_losses']['random_mean']==pytest.approx(1.)
    assert result['gates']['recovery_unclipped']==pytest.approx(2.)
    failed=study.pooled_decision(arrays(edited=1.2),validity())
    assert not failed['qualified']  # Perfect late20 cannot rescue primary15.


def test_same_mean_suffix_reference_and_unclipped_recovery_gates():
    cases=[arrays(edited=.91),arrays(random=(.64,.64)),arrays(suffix=.051),
           arrays(component=.91),arrays(component=.6,edited=.704)]
    for data in cases:assert not study.pooled_decision(data,validity())['qualified']
    assert study.pooled_decision(arrays(component=1.1),validity())['gates']['recovery_unclipped'] is None


def test_any_bad_or_missing_shard_cannot_be_dropped_or_partially_tested():
    base=validity()
    bad=copy.deepcopy(base);bad[2]['numerical_checks']['passed']=False
    cases=[bad,base[:3],list(reversed(base))]
    bad=copy.deepcopy(base);bad[0]['primary15_random_geometry_pass']=False;cases.append(bad)
    bad=copy.deepcopy(base);bad[3]['peak_resident_bytes']=study.CAP+1;cases.append(bad)
    for records in cases:
        result=study.pooled_decision(arrays(),records)
        assert result['status']=='failed_incomplete' and result['claims']=={} and not result['qualified']
        assert not study.controller_validity(records,1)
    data=arrays();data['seed_ids']=data['seed_ids'].astype(float);data['seed_ids'][0]+=.9
    assert study.pooled_decision(data,base)['claims']=={}
    data=arrays();data['seed_ids']=data['seed_ids'][::-1]
    assert study.pooled_decision(data,base)['claims']=={}
    for field,value in [('truth',np.zeros((64,32))),('common_scale',np.ones((64,1)))]:
        data=arrays();data[field]=value;assert study.pooled_decision(data,base)['claims']=={}
    data=arrays();del data['cut15_anchored'];assert study.pooled_decision(data,base)['claims']=={}


def test_resource_failure_overrides_conditional_success_without_nested_claims():
    result=study.pooled_decision(arrays(),validity());assert result['qualified']
    failed=study.resource_override(result,False,'memory')
    assert failed['claims']=={} and not failed['qualified'] and 'gates' not in failed


def test_capture_stores15_and20_at_correct_indices_without_cut0():
    import torch
    class Token(torch.nn.Module):
        def forward(self,x):return torch.zeros(len(x),4,1280)
    class Block(torch.nn.Module):
        def forward(self,x):return x+1,None
    class Model(torch.nn.Module):
        aridx=5
        def __init__(self):
            super().__init__();self.tokenizer=Token();self.stacked_xf=torch.nn.ModuleList([Block() for _ in range(20)]);self.output_projection_point=torch.nn.Identity()
        def forward(self,x):
            h=self.tokenizer(x)
            for block in self.stacked_xf:h,_=block(h)
            return self.output_projection_point(h)
    model=Model();cache={}
    with study.partial_capture(model,cache):model(torch.ones(2,4,64))
    assert set(cache)=={15,20,'head','raw'}
    assert torch.all(cache[15][0]==15) and torch.all(cache[20][0]==20)
    assert cache['head'][0].shape==(2,64)


def test_scientific_declarations_cannot_disagree_with_hardcoded_analysis():
    p=dict(seed_ids=list(study.SEEDS),candidate=15,cuts=[15,20],late_control=20,forbidden_candidate_search=[0,5,10],
        periods=[32,16],context=128,horizon=64,conditions=['baseline','pure_phi','pure_psi','background'],family_size=2,
        random_count=2,random_seed_key=study.RANDOM_KEY,random_scope=study.RANDOM_SCOPE,
        random_geometry_FP64=1e-8,random_geometry_injected_FP32=1e-5,geometry_reference_floor=1.,
        point_gates=copy.deepcopy(study.POINT_GATES),numerical_gates=copy.deepcopy(study.NUMERICAL_GATES),resources=copy.deepcopy(study.RESOURCES))
    assert study.declarations_match(p)
    for key,value in [('candidate',20),('horizon',32),('family_size',8),('random_scope','late20random')]:
        bad=copy.deepcopy(p);bad[key]=value;assert not study.declarations_match(bad)
    bad=copy.deepcopy(p);bad['point_gates']['native_gain']=.01;assert not study.declarations_match(bad)
