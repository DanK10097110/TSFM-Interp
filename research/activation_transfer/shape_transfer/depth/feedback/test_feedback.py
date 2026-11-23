import numpy as np

from research.activation_transfer.shape_transfer.depth.feedback.run import limit,loss


def test_controller_distinguishes_level_from_shape_and_caps_the_total_vector():
    goal=np.zeros((2,8))
    y=np.array([[1.]*8,[1.,-1.]*4])
    np.testing.assert_allclose(loss(y,goal,3.),[1.,9.])
    v=np.array([[3.,4.],[-.5,.25]])
    np.testing.assert_allclose(limit(v,2.),[[1.2,1.6],[-.5,.25]])
