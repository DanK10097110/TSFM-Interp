import numpy as np
from research.activation_transfer.forecast_requests.run import desired, scale_factors
from research.activation_transfer.forecast_requests.summary import item_scores


def test_calibration_uses_shared_gain_and_respects_sign():
    goal=desired()
    effects=np.broadcast_to(goal*.25,(7,*goal.shape)).copy()
    effects[:,1]*=-1
    np.testing.assert_allclose(scale_factors(effects,goal),[4,0,4])


def test_metric_penalizes_wrong_direction_and_spillover():
    goal=desired()[1]
    rows=np.stack([goal,-goal,goal+.1])
    vals=item_scores(rows,goal)
    np.testing.assert_allclose(vals["skill"],[1,-3,-3])
    np.testing.assert_allclose(vals["gain"],[1,-1,1])
    np.testing.assert_allclose(vals["collateral_sq"],[0,0,4],atol=1e-13)
