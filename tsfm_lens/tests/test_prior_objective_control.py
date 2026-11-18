"""Known analytic optima distinguish objective effects from continuation priors."""
import numpy as np
from experiments.feature_search.prior_objective_control import absolute_risk,median_distance


def test_two_point_median_and_mean_differ():
    tone=np.array([[2.,-4.]])
    for p,best in ((.25,np.zeros_like(tone)),(.75,tone)):
        assert np.allclose(absolute_risk(best,tone,p),.75)
        assert np.all(absolute_risk(p*tone,tone,p)>absolute_risk(best,tone,p))
        assert np.allclose(median_distance(best,tone,p),0)
    assert np.allclose(absolute_risk(np.zeros_like(tone),tone,.25),.75)
    assert np.allclose(absolute_risk(np.zeros_like(tone),tone,.75),2.25)


def test_half_probability_has_a_median_interval_per_time():
    tone=np.array([[2.,-4.]])
    for gain in (0.,.1,.5,.9,1.):
        assert np.allclose(absolute_risk(gain*tone,tone,.5),1.5)
        assert np.allclose(median_distance(gain*tone,tone,.5),0)
    assert np.all(median_distance(1.1*tone,tone,.5)>0)
