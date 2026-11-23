import numpy as np

from research.activation_transfer.shape_transfer.forecast_loss.run import loss_change


def test_identical_removal_opposite_accuracy_and_crossed_zero():
    delta = np.array([[-2.,-.2]])
    err1 = np.array([[1.5,1.0]])
    err2 = -err1
    for err in (err1,err2):
        for squared,power in ((True,2),(False,1)):
            actual = np.mean(np.abs(err+delta)**power - np.abs(err)**power,axis=-1)
            np.testing.assert_allclose(loss_change(err,delta,squared),actual)
    assert loss_change(err1,delta).item()<0<loss_change(err2,delta).item()
