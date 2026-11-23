"""Exercise sign, within-history identity and forecast/past shape distinctions."""
import numpy as np

from research.activation_transfer.moment.crosscoder.directions import compare,sample
from research.activation_transfer.moment.crosscoder.patch_probe import compare as probe_compare
from research.activation_transfer.moment.crosscoder.readout_slots import per_history,writes
from research.activation_transfer.moment.crosscoder.mechanism import gate_predictions


def test_distractor_means_exclude_matched_identity_and_preserve_sign():
    g=np.random.default_rng(71)
    ids=np.array([[0,3,1,4],[1,4,0,2]])
    source=g.normal(size=(5,8))
    target=-source.copy()
    readings=probe_compare(source,target,ids)
    np.testing.assert_allclose(readings["shape"],-1.,atol=1e-12)
    np.testing.assert_array_equal(readings["level_sign"],0)
    panel=compare(dict(small=source[ids],large=target[ids],dense=source[ids]))
    for r,features in enumerate(ids):
        for c,f in enumerate(features):
            others=np.delete(features,c)
            expected_shape=np.mean([np.corrcoef(source[f],target[h])[0,1] for h in others])
            expected_level=np.mean([np.sign(source[f].mean())==np.sign(target[h].mean()) for h in others])
            np.testing.assert_allclose(readings["other_shape"][r,c],expected_shape)
            np.testing.assert_allclose(readings["other_level_sign"][r,c],expected_level)
            np.testing.assert_allclose(panel["within_history"][r,c],expected_shape)
    np.testing.assert_allclose(panel["dense"],1.)


def test_doses_and_gate_boundary():
    ds=np.eye(3);dt=np.diag([2.,3.,5.]);indices=np.array([1,2])
    q=np.array([0.,2.])
    out=writes(ds,dt,indices,np.array([0,1]),q)
    for z in out.values():np.testing.assert_array_equal(z[0],0.)
    np.testing.assert_array_equal(out["small"][1],[0,0,-2.])
    np.testing.assert_array_equal(out["paired"][1],[0,0,-10.])
    np.testing.assert_array_equal(out["decoy"][1],[0,-10.,0])
    pred=gate_predictions(np.array([[1.,10.,11.,10.],[-1.,-2.,3.,.1]]))
    np.testing.assert_array_equal(pred,[[0,10,11],[0,0,3]])


def test_same_feature_at_different_readouts_is_scored_separately():
    shape=np.array([1.,2.,-1.,0.,1.,2.,1.,4.])
    source=np.broadcast_to(shape,(5,4,8))
    rec=source.copy()+7
    rec[:,2:]=2-source[:,2:]
    vals=per_history(dict(small=source, paired=rec, decoy=-rec))
    np.testing.assert_allclose(vals["paired"],np.tile([1,1,-1,-1],(5,1)))
    np.testing.assert_allclose(vals["decoy"],-vals["paired"])
    r,p,q=sample(np.repeat(["daily","traffic"],128))
    assert len(set(r))==64 and np.all(p!=q)
