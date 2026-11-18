"""Known-answer checks for adaptive residual preservation and hidden composition."""
import numpy as np
from experiments.feature_search.selective_tree import decompose,fuse,local_metrics


def test_offgrid_tone_is_peeled_without_losing_residual_or_level():
    t=np.arange(-128,0);rng=np.random.default_rng(2);x=3+8*np.sin(2*np.pi*17.35*t/128+.4)+rng.normal(0,.05,128)
    levels,details=decompose(x)
    assert len(details)>=1 and abs(details[0]['frequency']-17.35)<.01
    assert len(levels[1])==2 and np.std(levels[1][-1])<.1
    for level in levels:np.testing.assert_allclose(np.sum(level,axis=0),x,atol=1e-12)
    assert abs(np.mean(levels[1][-1])-3)<.02


def test_noise_and_constant_are_not_forced_into_three_splits():
    for x in [np.ones(128),np.random.default_rng(4).normal(size=128)]:
        levels,details=decompose(x);assert len(details)==0 and len(levels)==1


def test_anchored_fusion_recovers_affine_model_and_is_translation_invariant():
    rng=np.random.default_rng(4);a=rng.normal(size=(3,7));linear=rng.normal(size=(7,5));offset=np.full(5,8.)
    states=a@linear+offset;expected=a.sum(0)@linear+offset
    np.testing.assert_allclose(fuse(states,offset,[1,1,1]),expected)
    weights=[.2,.6,1];shift=rng.normal(size=5)
    np.testing.assert_allclose(fuse(states+shift,offset+shift,weights)-shift,fuse(states,offset,weights))


def test_local_curvature_distinguishes_linear_from_nonlinear_response():
    base=np.ones(8)*2;eps=.01;delta=np.ones(8)*eps
    linear=local_metrics(base,base+delta,base-delta,eps)
    nonlinear=local_metrics(base,base+delta+.5*eps,base-delta+.5*eps,eps)
    assert abs(linear['sensitivity']-.5)<1e-12 and linear['curvature_ratio']<1e-10
    assert abs(nonlinear['curvature_ratio']-.5)<1e-9


def test_policy_selection_does_not_use_test_losses():
    from experiments.feature_search.selective_analysis import choose_config
    rows=[dict(source='a',split='fit'),dict(source='b',split='test')];configs=['a','b'];losses={'a':np.array([1.,100.]),'b':np.array([2.,0.])}
    assert choose_config(configs,losses,rows,[0])=='a'
    losses['a'][1]=1e9;losses['b'][1]=-1e9
    assert choose_config(configs,losses,rows,[0])=='a'


def test_geometry_weights_use_fit_states_and_recover_known_composition(tmp_path):
    import json
    from experiments.feature_search.selective_analysis import fit_geometry
    (tmp_path/'rows.json').write_text(json.dumps([dict(source='fit',split='fit'),dict(source='test',split='test')]))
    (tmp_path/'plans.json').write_text(json.dumps([dict(levels=[[0],[1,2]]),dict(levels=[[0],[1,2]])]))
    z=np.full((1,1,4),3.);tone=z+np.array([[[1.,0,1,0]]]);resid=z+np.array([[[0,1.,0,1.]]]);full=z+1.6*(tone-z)+.8*(resid-z)
    values={}
    for i in [0,1]:values.update({f'{i}_full':full.copy(),f'{i}_zero':z.copy(),f'{i}_depth1':np.stack([tone,resid],axis=1)})
    for model in ['bolt','timesfm']:
        (tmp_path/f'{model}_worker.json').write_text(json.dumps(dict(cuts=['0'])))
        np.savez(tmp_path/f'{model}_states.npz',**values)
    first=fit_geometry(tmp_path);np.testing.assert_allclose(first['bolt']['1']['0']['weights'],[1.6,.8],atol=.01)
    values['1_full']=full*1e6
    for model in ['bolt','timesfm']:np.savez(tmp_path/f'{model}_states.npz',**values)
    second=fit_geometry(tmp_path)
    assert first==second


def test_forecast_policy_ties_keep_native_instead_of_unsupported_edit():
    from experiments.feature_search.selective_analysis import choose_config
    native=('native',0,'none','none',0.);fusion=('fusion',3,'0','equal',1.)
    rows=[dict(source='a',split='fit')];losses={native:np.ones(1),fusion:np.ones(1)}
    assert choose_config([fusion,native],losses,rows,[0])==native
