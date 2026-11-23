import numpy as np
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import apply,train
from research.activation_transfer.moment.dictionary_transfer.weather.detector.score import detection


def test_closed_receiving_threshold_keeps_true_zero_and_raw_open_amplitude():
    np.testing.assert_allclose(apply(np.array([-1.,.3,.7,1.]),.5),[0.,0.,.7,1.])


def test_dose_scaling_uses_only_fit_rows():
    x=np.array([[0.,2.],[1.,2.],[2.,2.],[3.,2.],[4.,2.],[5.,30.]])
    y=np.array([[0.],[0.],[2.],[3.],[4.],[5.]])
    f=train(x,y,np.arange(4),np.arange(4,6))
    np.testing.assert_allclose(f['center'],[1.5,2.])
    np.testing.assert_equal(f['inverse_sd'][1],0.)
    assert f['lam'][0] in (100.,10.,1.,.1,.01)


def test_all_history_detection_counts_extra_recipient_firing():
    got=detection(np.array([0.,2.,1.,0.]),np.array([1.,2.,0.,0.]))
    assert (got['tp'],got['fp'],got['fn'],got['tn'])==(1,1,1,1)
    assert got['precision']==.5 and got['recall']==.5
