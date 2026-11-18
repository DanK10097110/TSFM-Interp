"""Conditional value homogeneity with positional-value and roundoff decoys."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'experiments/feature_search/patch_homogeneity.py'
spec=importlib.util.spec_from_file_location('patch_homogeneity',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_identical_vs_alternating_tokens_and_scale_invariance():
    """Identical32patches and alternating16patches have distinct diagnostics."""
    rng=np.random.default_rng(5);base=rng.normal(size=(12,1,7));same=np.repeat(base,8,axis=1)
    stats=audit.token_variation(same)
    assert stats['max_inter_token_abs_difference']==0
    assert stats['within_token_variance_fraction']<1e-30
    alternating=same.copy();alternating[:,1::2]+=2
    stats=audit.token_variation(alternating)
    assert stats['within_token_variance_fraction']>.1
    assert stats['within_parity_variance_fraction']<1e-30
    assert stats['even_odd_mean_separation_mse']==4
    scaled=audit.token_variation(alternating*7)
    np.testing.assert_allclose(stats['within_token_variance_fraction'],scaled['within_token_variance_fraction'],atol=1e-14)


def test_arbitrary_attention_weights_preserve_equal_values_but_positional_values_break_it():
    """Q/K can differ arbitrarily; adding position toV breaks the premise."""
    rng=np.random.default_rng(7);logits=rng.normal(size=(9,8))*4
    value=rng.normal(size=(1,5));identical=np.repeat(value,8,axis=0)
    output=audit.attention_average(logits,identical)
    np.testing.assert_allclose(output,np.repeat(value,9,axis=0),atol=1e-14)
    positioned=identical+np.arange(8)[:,None]
    assert np.ptp(audit.attention_average(logits,positioned),axis=0).max()>1
    import torch
    from timesfm.torch.transformer import _dot_product_attention
    q=torch.tensor(rng.normal(size=(1,8,2,3)),dtype=torch.float64)
    k=torch.tensor(rng.normal(size=(1,8,2,3)),dtype=torch.float64)
    v=torch.tensor(rng.normal(size=(1,1,2,3)),dtype=torch.float64).repeat(1,8,1,1)
    native=_dot_product_attention(q,k,v)
    np.testing.assert_allclose(native.numpy(),v.numpy(),atol=1e-14)


def test_context_numerical_difference_is_not_reported_as_exact_identity():
    """A tiny but nonzero context effect remains explicitly nonzero."""
    metadata=[dict(condition=c) for c in ['mixed','pure_phi','pure_psi']]
    base=np.ones((3,32));later=base.copy();later[0,0]+=1e-6
    result=audit.compare_context_outputs({128:base,256:later,512:base},metadata)
    assert not result['256_vs_128']['mixed']['exact_equal']
    np.testing.assert_allclose(result['256_vs_128']['mixed']['max_abs_difference'],1e-6,atol=1e-14)
    assert result['512_vs_128']['mixed']['exact_equal']
    assert all(audit.native_source_audit()['checks'].values())
