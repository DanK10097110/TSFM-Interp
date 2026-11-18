"""Native head replay known answers with quantile, bias and cancellation decoys."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest


def _module():
    """Load only the CPU head replay instrument."""
    root=Path(__file__).parents[1]/'experiments/feature_search';sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('head_branch_audit',root/'head_branch_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_point_channel_respects_different_flattening_layouts():
    """The extra TimesFM channel and Bolt quantile-major layout are explicit decoys."""
    m=_module();tf=np.arange(1280)[None,:];bolt=np.arange(576)[None,:]
    np.testing.assert_equal(m.point_channel(tf,'timesfm')[0],np.arange(32)*10+5)
    np.testing.assert_equal(m.point_channel(bolt,'bolt')[0],np.arange(32)+4*64)
    with pytest.raises(ValueError):m.point_channel(np.zeros((1,128)), 'timesfm')


def test_replay_includes_both_branch_biases_and_uses_last_position():
    """An unread loud first position and nonzero branch biases expose plausible shortcuts."""
    m=_module();h=np.array([[[100,100],[.1,.2]],[[200,200],[-.5,.4]]]);weights={'hidden_layer.weight':np.array([[1,-.5],[-1,2],[.3,.4]]),'hidden_layer.bias':np.array([.1,-.2,.5]),'output_layer.weight':np.array([[.2,-.3,.6],[1,.5,-.4]]),'output_layer.bias':np.array([.7,-.8]),'residual_layer.weight':np.array([[.5,-.25],[-.2,.1]]),'residual_layer.bias':np.array([.4,-.5])}
    skip,nl=m.branches(h,weights,'relu');x=h[:,-1];hidden=np.maximum(x@weights['hidden_layer.weight'].T+weights['hidden_layer.bias'],0)
    np.testing.assert_allclose(skip,x@weights['residual_layer.weight'].T+weights['residual_layer.bias'],atol=1e-14)
    np.testing.assert_allclose(nl,hidden@weights['output_layer.weight'].T+weights['output_layer.bias'],atol=1e-14)


def test_signed_projection_and_energy_include_cancellation():
    """A huge branch may cancel another;absolute percentages give the wrong answer."""
    m=_module();skip=np.tile([10.,0.],(5,1));nonlinear=np.tile([-9.,1.],(5,1));total=skip+nonlinear
    result=m.attribution(total,{'skip':skip,'nonlinear':nonlinear})
    assert result['branches']['skip']['signed_projection_share']['value']==5
    assert result['branches']['nonlinear']['signed_projection_share']['value']==-4
    assert result['cross_energy']['skip__nonlinear']['value']==-180
    assert result['projection_share_sum']==1
    assert abs(result['energy_closure_error'])<1e-12


def test_source_population_normalization_and_constant_decoy():
    """TimesFM population running statistics differ from sample std and Bolt zero floor."""
    m=_module();x=np.random.default_rng(15).normal(size=(8,128)).astype(np.float32)
    for model in ['bolt','timesfm']:
        loc,scale=m.normalizers(x,model)
        np.testing.assert_allclose(loc,x.mean(1),atol=1e-7)
        np.testing.assert_allclose(scale,x.std(1),atol=2e-7)
    constant=np.full((2,128),3,dtype=np.float32)
    assert np.all(m.normalizers(constant,'timesfm')[1]==0)
    np.testing.assert_allclose(m.normalizers(constant,'bolt')[1],1e-5,rtol=1e-7)
