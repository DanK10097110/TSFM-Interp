import numpy as np

from research.activation_transfer.shape_transfer.response.analyze import moments
from research.activation_transfer.shape_transfer.response.model import goals, limit_norm, match_norm, ridge_response


def test_common_response_bridge_recovers_cross_item_rotation():
    rng=np.random.default_rng(441)
    target=rng.normal(size=(32,8,3)); w=rng.normal(size=(3,4))
    source=target@w
    got,penalty=ridge_response(target,source)
    g=target.reshape(-1,3).T@target.reshape(-1,3)
    expected=np.linalg.solve(g+penalty*np.eye(3),g@w)
    np.testing.assert_allclose(got,expected,atol=1e-12)
    # Histories have different Jacobians even though their population average is small.
    unpaired,_=ridge_response(target,np.broadcast_to(source.mean(0),source.shape))
    assert np.linalg.norm(unpaired-got)>.5*np.linalg.norm(got)


def test_unseen_waveforms_have_equal_energy_outside_original_span():
    wave=goals()
    np.testing.assert_allclose(np.sqrt(np.mean(wave**2,axis=1)), .05)
    np.testing.assert_allclose(wave[3:]@wave[:3].T,0.,atol=1e-15)
    np.testing.assert_allclose(wave[3]@wave[4],0.,atol=1e-15)


def test_common_radius_and_equal_dose_keep_original_directions():
    v=np.array([[.3,.4],[0.,-7.]])
    capped=limit_norm(v,2.)
    np.testing.assert_allclose(capped, [[.3,.4],[0.,-2.]])
    base=np.array([[2.,0.],[-1.,0.]])
    matched=match_norm(v,base)
    np.testing.assert_allclose(matched, [[1.2,1.6],[0.,-1.]])


def test_source_template_cannot_score_history_specific_recovery():
    rng=np.random.default_rng(81)
    native=2.+.25*rng.normal(size=(40,2,8))
    pred=np.broadcast_to(native.mean(0),native.shape)
    weights=np.random.default_rng(13).multinomial(40,np.full(40,.025),size=9)
    assert (moments(pred,native,weights,False)['skill']>.9).all()
    s=moments(pred,native,weights,True)
    np.testing.assert_allclose(s['skill'],0.)
    item_idx=np.repeat(np.arange(len(native)),weights[0]); r=native[item_idx]; p=native[item_idx][::-1]
    same=moments(p,r,np.ones((1,len(r))),True)
    pp=p-p.mean(0);rr=r-r.mean(0)
    np.testing.assert_allclose(same['skill'][0],1-((pp-rr)**2).sum((0,2))/(rr**2).sum((0,2)))
