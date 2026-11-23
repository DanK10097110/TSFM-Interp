import numpy as np

from research.activation_transfer.moment.dictionary_transfer.semantics.core import measurements, nominate, predicted
from research.activation_transfer.moment.dictionary_transfer.semantics.evaluate import auc_many, indices
from research.activation_transfer.moment.dictionary_transfer.semantics.physical import waves


def test_patch_mean_and_signed_ols_span_isolate_shift_and_ramp():
    t=np.arange(480)
    raw=np.stack((22+.1*t+np.cos(t/5), -7+.02*t+np.sin(t/3)))
    baseline_native=measurements(raw)[1]
    for addition, wanted in [(np.repeat(2.,8),[2-16/480,0]),(np.linspace(-2,2,8),[0,4])]:
        changed=raw.copy()
        changed[:,-8:]+=addition
        normalized,physical,patch=measurements(changed)
        np.testing.assert_allclose(physical-baseline_native,np.tile(wanted,(2,1)),atol=1e-12)
        np.testing.assert_allclose(normalized,physical/(changed.std(1,keepdims=True)+1e-5))


def test_training_signed_selectors_remain_distinct_on_tied_rankings():
    u=np.tile(np.linspace(-2,2,40),3)
    train,val=np.arange(80),np.arange(80,120)
    z=np.maximum(-u,0)
    # The last two atoms have the same ranking on two identical input statistics.
    y=np.column_stack((np.tile([0.,1.],60),z,z))
    winners,ranked=nominate(np.column_stack((u,u)),y,np.array([30,10,20]),train,val)
    assert [winners[k]["id"] for k in ("level","trend")]==[10,20]
    assert winners["level"]["polarity"]==-1 and winners["level"]["validation"]["auc"]==1
    x=u[train]
    b=np.linalg.lstsq(np.column_stack((np.ones(len(x)),x)),z[train],rcond=None)[0]
    got=predicted(np.array([-10.,10.]),winners["level"])
    np.testing.assert_allclose(got,np.maximum(0,b[0]+b[1]*np.array([-10.,10.])))


def test_tied_auc_and_bootstrap_keep_every_station_origin_together():
    fired=np.array([0,0,1,1])
    scores=np.array([0.,5.,5.,7.])
    np.testing.assert_allclose(auc_many(fired,scores,np.array([[0,1,2,3]])),[.875])
    clusters=np.repeat(np.arange(5),4)
    ix=indices("fixture",clusters)
    assert ix.shape==(2001,20)
    for row in ix[1:]:
        counts=np.bincount(row,minlength=20).reshape(5,4)
        assert np.all(counts==counts[:,:1])


def test_physical_scores_identical_centered_wave_and_no_change():
    truth=np.tile(np.arange(16),(8,1)).astype(float)
    rows=np.arange(8)
    z=waves(truth,dict(offset=truth+300,zero=np.zeros_like(truth)),np.ones(8,bool),rows,rows)
    assert z["eligible"]
    for lead in ("1–8","9–16"):
        result=z["centered_leads"][lead]["physical_skill"]
        assert result["offset"]["value"]==1 and result["zero"]["value"]==0
    assert not waves(truth,{},rows<7,rows,rows)["eligible"]
