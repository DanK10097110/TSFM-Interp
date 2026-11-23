import numpy as np
import torch
from tsfm_lens.sae.models import TopKSAE

from research.activation_transfer.boundary.analyze import Ridge, ci, metric_vectors, representations
from research.activation_transfer.boundary.audit_forecasts import pooled, sufficient_stats
from research.activation_transfer.boundary.capture import shapes
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.boundary.smooth import smooth_codes


def test_one_topk_crossing_rotates_the_reconstruction():
    sae = TopKSAE(2, 2, 1)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2))
        sae.W_dec.copy_(torch.eye(2))
    x = np.array([[1.001,1.]],np.float32)
    y = np.array([[.999,1.002]],np.float32)
    r = representations(sae,x,y)
    assert r["switched"].item()
    assert r["fraction_left"].item() == 1.
    np.testing.assert_allclose(r["decoded"],[[-1.001,1.002]],rtol=1e-6)
    np.testing.assert_allclose(r["fixed_decoded"],[[-.002,0.]],atol=1e-6)
    z = representations(sae,x,x)
    assert not z["switched"].any()
    assert not z["decoded"].any()


def test_smooth_gate_is_sparse_and_continuous_at_atom_exchange():
    pre = torch.tensor([[3.,1.01,1.,-1.],[3.,.99,1.,-1.]])
    soft,deg = smooth_codes(pre,2)
    assert not deg.any()
    for code in soft.values():
        assert torch.equal(code>0,torch.tensor([[True,True,False,False],[True,False,True,False]]))
        assert torch.max(torch.abs(code[1]-code[0])) < .022
    hard = pre.clone()
    hard[0,2:]=0.
    hard[1,1]=0.
    hard[1,-1]=0.
    assert torch.max(torch.abs(hard[1]-hard[0])) > 1.


def test_clean_duplicate_and_low_firing_case():
    x=torch.tensor([[3.,0.,-1.,-.1],[0.,0.,-1.,-4.]])
    soft, _=smooth_codes(x,2)
    for z in soft.values():
        assert torch.isfinite(z).all()
        assert z[1].abs().sum()==0


def test_series_bootstrap_and_known_linear_transport():
    rng=np.random.default_rng(817)
    x=rng.normal(size=(12,4))
    y=x@np.array([[1.,2.],[2.,1.],[1.,2.],[-1.,0.]])+2.
    train=Ridge(x[:8],y[:8],4)
    np.testing.assert_allclose(train.predict(train.transform(x[8:]),0),y[8:],atol=1e-10)
    assert ci(np.ones((4,2)))=={"mean":1.,"ci95":[1.,1.],"n_items":4}
    refs=np.array([[1.,0.],[0.,1.]])
    np.testing.assert_allclose(metric_vectors(-refs,refs)["cosine"],-1)


def test_patch_indices_exclude_reg_and_respect_left_padding():
    class Adapter:
        def token_slice(self,n):
            return slice(1,n-1)
    assert last_window_slice(Adapter(),np.array([2,3]))(6)==slice(3,5)


def test_smooth_input_scale_and_nonleaking_ids():
    ids=np.array(["4100","3600"])
    cue=shapes("m4",ids)
    np.testing.assert_array_equal(shapes("m4",ids[::-1])[::-1],cue)
    np.testing.assert_allclose(np.mean(cue**2,axis=-1),1.,atol=1e-7)
    np.testing.assert_allclose(cue[:,:,[0,-1]],0.,atol=1e-7)
    assert not np.allclose(cue[0],cue[1])


def test_forecast_energy_audit_keeps_scale_and_negative_direction():
    y=np.arange(12.).reshape(2,2,3)
    v=pooled(sufficient_stats(-2*y,y).sum(0))
    np.testing.assert_allclose([v["cosine"],v["relative_rmse"],v["gain"],v["energy_ratio"]],
                               [-1,3,-2,4])


def test_norm_control_matches_energy_and_retains_the_hard_gate():
    from research.activation_transfer.boundary.gate_controls import norm_match
    hard=torch.tensor([[4.,2.,0.],[0.,0.,0.]])
    soft=torch.tensor([[3.,.2,0.],[0.,0.,0.]])
    matched=norm_match(hard,soft)
    torch.testing.assert_close(matched.norm(dim=1),soft.norm(dim=1))
    assert torch.equal(matched>0,hard>0)
    torch.testing.assert_close(matched[0,0]/matched[0,1],torch.tensor(2.))


def test_paired_map_selects_on_validation_alone():
    from research.activation_transfer.boundary.analyze import train_validate_one
    rng=np.random.default_rng(74)
    xs=rng.normal(size=(192,2,4))
    ys=xs@np.array([[2.,0.],[0.,-2.],[1.,1.],[-2.,3.]])+1.
    untouched=ys.copy()
    untouched[176:]=rng.normal(size=untouched[176:].shape)*100
    a,penalty,rank=train_validate_one(xs,ys,xs[176:],4)
    b,penalty2,rank2=train_validate_one(xs,untouched,xs[176:],4)
    assert rank==rank2==4
    assert penalty==penalty2==0.
    np.testing.assert_allclose(a,b)
    np.testing.assert_allclose(a,ys[176:].reshape(-1,2),atol=1e-10)


def test_frequency_priority_and_projection_scale():
    from research.activation_transfer.boundary.fixed import normalized_columns, popular_atoms
    codes = np.array([[2.,5.,7.,0.],[2.,4.,0.,0.],[0.,0.,100.,0.]])
    np.testing.assert_array_equal(popular_atoms(codes,2),[2,1])
    equal = np.array([[1.,1.,0.],[1.,1.,3.]])
    np.testing.assert_array_equal(popular_atoms(equal,2),[0,1])
    basis = normalized_columns(np.array([[0.,-3.],[4.,4.]]))
    np.testing.assert_allclose(np.linalg.norm(basis,axis=0),[1.,1.])


def test_matching_recovers_permuted_positive_atoms_and_change():
    from research.activation_transfer.boundary.seed_identity import apply_match, fit_scale, match_features, random_targets
    rng=np.random.default_rng(39)
    features=np.eye(19)
    order=rng.permutation(19)
    a,b,sim=match_features(features,features[order])
    np.testing.assert_allclose(sim,1.)
    z=rng.uniform(0,3,(30,19))
    target=2.5*z[:,order]
    scale=fit_scale(z[:,a],target[:,b])
    np.testing.assert_allclose(scale,2.5)
    np.testing.assert_allclose(apply_match(z,target.shape,a,b,scale),target,rtol=1e-6)
    np.testing.assert_array_equal(np.sort(random_targets(b,np.arange(19),13)),np.sort(b))


def test_random_rotation_preserves_learned_gram():
    from research.activation_transfer.boundary.fixed_geometry import orthogonal_parts
    rng=np.random.default_rng(14)
    m=rng.normal(size=(30,4))
    q,r=orthogonal_parts(m)
    u,_=orthogonal_parts(rng.normal(size=(30,4)))
    np.testing.assert_allclose(q@r,m,rtol=1e-5,atol=1e-6)
    rotated=u@r
    np.testing.assert_allclose(rotated.T@rotated,m.T@m,rtol=1e-5,atol=1e-6)


def test_crossing_selection_counts_independent_items_and_prefers_pulse():
    from research.activation_transfer.boundary.crossing import contexts, select_switched
    z=np.array([[True,False],[True,False],[True,False],[True,False]])
    alt=np.repeat(z[:,None],2,axis=1)
    alt[1,1]=alt[1,1,::-1]; alt[2,0]=alt[2,0,::-1]; alt[2,1]=alt[2,1,::-1]
    alt[3,0]=alt[3,0,::-1]
    idx,family=select_switched(z,alt,2)
    np.testing.assert_array_equal(idx,[1,2]); np.testing.assert_array_equal(family,[1,0])
    x=np.zeros((2,80),np.float32); cue=np.ones((2,64),np.float32)
    out=contexts(x,cue,np.array([0.,.5]))
    assert not out[:,:16].any(); assert not out[0].any()
    np.testing.assert_allclose(out[1,-64:],.015)


def test_clean_translation_discards_intercept():
    from research.activation_transfer.boundary.seed_translation import clean_map
    rng=np.random.default_rng(12)
    x=rng.normal(size=(210,4))+2
    weights=rng.normal(size=(4,3))
    y=x@weights+19.
    delta=rng.normal(size=(17,2,4))
    pred,penalty=clean_map(x,y,delta)
    assert penalty==0
    np.testing.assert_allclose(pred,(delta@weights).reshape(-1,3),atol=1e-12)


def test_decoder_refit_recovers_planted_clean_mapping():
    from research.activation_transfer.boundary.decoder_refit import corrected_decoder
    rng=np.random.default_rng(19)
    z=rng.normal(size=(200,5)).astype('float32')
    d=rng.normal(size=(5,3)).astype('float32'); b=np.array([1.,2.,-3.],dtype='float32')
    target=z@d+b
    fitted,bias,pen=corrected_decoder(z,target,np.zeros_like(d),np.array([4.,8.,1.],dtype='float32'))
    assert pen==0
    np.testing.assert_allclose(fitted,d,atol=2e-6)
    np.testing.assert_allclose(bias,b,atol=2e-6)
