"""Known-answer controls for native normalization and omitted-inverse-information tests."""
import numpy as np
from experiments.feature_search.response_surface_normalization import (
    project, native_metadata, normalize_with, restore, ridge_grouped)


def test_fourier_projection_separates_dc_and_both_offgrid_tones():
    """A large DC nuisance and neighboring off-grid tones must not leak into coefficients."""
    t = np.arange(-128, 0); coefficients = np.array([.4, -.8, 1.2, .3])
    basis = np.column_stack([v for f in (128/24,128/19.2) for v in
        (np.sin(2*np.pi*f*t/128),np.cos(2*np.pi*f*t/128))])
    for spans in (1,4):
        actual = project((99+basis@coefficients)[None], spans)
        np.testing.assert_allclose(actual, np.broadcast_to(coefficients, actual.shape), atol=2e-12)


def test_normalized_difference_inverse_requires_observed_background_term():
    """Different endpoint scales produce a background contribution even at fixed signal."""
    cb = np.arange(16).reshape(1,4,4)/10
    signal = np.array([[.1,.2,-.3,.4]])
    sq = np.array([[2.,3.,4.,5.]]); sb = np.array([[1.,2.,1.,3.]])
    delta = (cb+signal[:,None])/sq[...,None]-cb/sb[...,None]
    np.testing.assert_allclose(restore(delta.reshape(1,-1),'normalized_delta',sq,sb,cb),signal,atol=1e-15)
    assert np.max(abs((sq[...,None]*delta).mean(1)-signal))>.1


def test_endpoint_inverse_uses_each_native_prefix_scale():
    """Using only the final TimesFM scale cannot invert earlier normalized patches."""
    cq = np.arange(16).reshape(1,4,4)/7; cb = cq*.3
    sq = np.array([[.5,1.,2.,4.]]); sb = np.array([[4.,2.,1.,.5]])
    paired = np.stack([cq/sq[...,None],cb/sb[...,None]],axis=1).reshape(1,-1)
    np.testing.assert_allclose(restore(paired,'normalized_pair',sq,sb,cb),(cq-cb).mean(1),atol=1e-15)


def test_timesfm_stats_are_cumulative_prefix_not_global_or_perpatch():
    """Actual lightweight native kernels preserve the four distinct normalization spans."""
    x = np.concatenate([np.arange(32)*s+mu for s,mu in [(1.,0.),(2.,30.),(3.,-40.),(.5,80.)]])[None]
    z, mu, sigma = native_metadata(x,'timesfm')
    np.testing.assert_allclose(mu[0],[x[0,:n].mean() for n in [32,64,96,128]],atol=1e-5)
    np.testing.assert_allclose(sigma[0],[x[0,:n].std() for n in [32,64,96,128]],rtol=1e-6)
    np.testing.assert_array_equal(z,normalize_with(x,mu,sigma))
    assert np.ptp(sigma)>5 and mu.shape==(1,4)
    zb, mb, sb = native_metadata(x,'bolt')
    assert mb.shape==(1,1)
    np.testing.assert_allclose(sb[0,0],x.std(),rtol=1e-6)
    np.testing.assert_array_equal(zb,normalize_with(x,mb,sb))


def test_omitted_scale_plant_recovers_with_fair_inverse_not_direct_linear_fit():
    """A perfectly accessible normalized feature can fail direct physical linear decoding."""
    rng = np.random.default_rng(4953); groups = np.repeat(np.arange(8),15)
    normalized = rng.normal(size=(120,4)); scale = np.exp(rng.normal(0,1,len(groups)))[:,None]
    physical = normalized*scale
    direct, _, _, _ = ridge_grouped([normalized],physical,groups)
    pred, _, _, _ = ridge_grouped([normalized],normalized,groups)
    restored = restore(pred,'normalized_delta',scale,scale,np.zeros((120,1,4)))
    direct_error = np.mean((direct-physical)**2)/np.mean(physical**2)
    corrected_error = np.mean((restored-physical)**2)/np.mean(physical**2)
    assert direct_error>.1 and corrected_error<1e-4


def test_grouped_scaling_and_metadata_block_are_not_learned_from_test_source():
    """Train/test source separation applies to hidden and metadata scaling alike."""
    rng = np.random.default_rng(4);groups=np.repeat(np.arange(8),15)
    hidden=rng.normal(size=(120,7));metadata=rng.normal(size=(120,4));y=hidden[:,:2]
    pred, shuffled, mean, folds = ridge_grouped([hidden,metadata],y,groups)
    changed=metadata.copy();changed[groups==0]*=1e6
    _, _, _, other = ridge_grouped([hidden,changed],y,groups)
    assert folds[0]['block_scales']==other[0]['block_scales']
    assert all(f['source_id'] not in f['train_sources'] for f in folds)
    assert np.mean((pred-y)**2)<.01 and np.mean((shuffled-y)**2)>.1
    assert pred.shape==shuffled.shape==mean.shape==y.shape
