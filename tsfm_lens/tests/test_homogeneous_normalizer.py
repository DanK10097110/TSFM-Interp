"""Affine/radial and nonlinear decoys for the homogeneous TimesFM replay."""
import numpy as np
import pytest
from experiments.feature_search.homogeneous_normalizer import radial_identity,attribution,weighted_values


def test_affine_bias_native_residual_survives_but_fixed_normalizer_is_zero():
    """Native scalar normalization can create an affine-bias composition residual."""
    u=np.array([1.,0.]);v=np.array([0.,2.])
    function=lambda z:np.array([2*z[0]+.3,3*z[1]-.7])
    native,fixed,radial=radial_identity(function,u,v,2.,1.,1.5)
    assert np.linalg.norm(native)>.1
    np.testing.assert_allclose(fixed,0,atol=1e-14)
    np.testing.assert_allclose(native,radial,atol=1e-14)


def test_bilinear_cross_feature_remains_under_fixed_normalizer():
    """A true cross-feature term is not removed by subtracting F(0)."""
    function=lambda z:np.array([z[0]*z[1]+.3,z[0]-z[1]])
    native,fixed,radial=radial_identity(function,np.array([1.,0.]),np.array([0.,2.]),2.,1.,1.5)
    np.testing.assert_allclose(fixed,[4.,0.],atol=1e-14)
    np.testing.assert_allclose(native,fixed+radial,atol=1e-14)


def test_signed_energy_cross_terms_and_small_native_denominator():
    """Cancellation shares may exceed one or be negative without being errors."""
    native=np.array([[1.,2.],[0.,0.]])
    component=np.array([[2.,4.],[1.,1.]])
    result=attribution(native,dict(first=component,second=native-component),np.ones(2))
    assert result['components']['first']['aggregate_signed_projection_share']==pytest.approx(2.)
    assert result['components']['second']['aggregate_signed_projection_share']==pytest.approx(-1.)
    assert result['cross_normalized_energy']['first__second'][0]<0
    assert result['vector_closure_max']==0 and result['energy_closure_max']==0
    assert result['small_native_residual']==[False,True]
    result=attribution(np.zeros((1,2)),dict(a=np.zeros((1,2))),np.ones(1))
    assert result['components']['a']['aggregate_signed_projection_share'] is None


def test_identical_values_null_routing_and_heterogeneous_decoy():
    """QK reweighting cannot distinguish homogeneous V but can alter heterogeneous V."""
    weights=np.array([[1.,0.],[.2,.8]])
    identical=np.array([[2.,3.],[2.,3.]])
    np.testing.assert_allclose(weighted_values(weights,identical),identical)
    heterogeneous=np.array([[2.,3.],[-2.,1.]])
    assert not np.allclose(weighted_values(weights,heterogeneous)[0],weighted_values(weights,heterogeneous)[1])
    with pytest.raises(ValueError,match='row-stochastic'):
        weighted_values(np.ones((2,2)),identical)


def test_collapsed_value_path_matches_direct_single_token_source_blocks():
    """The emulator keeps value/out, both residual paths, FFN and native channel."""
    import torch
    from types import SimpleNamespace
    from experiments.feature_search.homogeneous_normalizer import collapsed
    torch.manual_seed(4)
    linear=lambda a,b:torch.nn.Linear(a,b)
    attention=SimpleNamespace(fuse_qkv=False,value=linear(8,8),out=linear(8,8))
    layer=SimpleNamespace(pre_attn_ln=torch.nn.Identity(),attn=attention,post_attn_ln=torch.nn.Identity(),
        pre_ff_ln=torch.nn.Identity(),ff0=linear(8,12),activation=torch.nn.SiLU(),ff1=linear(12,8),post_ff_ln=torch.nn.Identity())
    model=SimpleNamespace(tokenizer=linear(64,8),stacked_xf=[layer],output_projection_point=linear(8,1280),aridx=5)
    patch=torch.randn(3,32)
    with torch.inference_mode():
        h=model.tokenizer(torch.cat([patch,torch.zeros_like(patch)],-1))
        h=attention.out(attention.value(h))+h
        h=layer.ff1(layer.activation(layer.ff0(h)))+h
        expected=model.output_projection_point(h).reshape(3,128,10)[:,:64,5].numpy()
    np.testing.assert_allclose(collapsed(model,patch),expected,atol=1e-7)
