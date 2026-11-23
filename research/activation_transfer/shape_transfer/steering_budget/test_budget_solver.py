import numpy as np
from scipy.optimize import minimize

from research.activation_transfer.shape_transfer.steering_budget.run import coefficients


def test_bounded_ridge_reallocates_length_and_matches_independent_primal_solution():
    A=np.array([[[6.,0.,0.],[0.,1.,2.]],[[0.,0.,0.],[0.,0.,0.]]])
    g=np.array([[1.,3.],[2.,1.]])
    lim=np.array([.2,.4])
    radial,optimal=coefficients(A,g,lim)
    def objective(x):
        return np.sum((A[0]@x-g[0])**2)+.001*36*np.sum(x**2)
    oracle=minimize(objective,np.zeros(3),constraints=[dict(type="ineq",fun=lambda x:.04-np.sum(x**2))],
                    method="SLSQP",options=dict(ftol=1e-13))
    assert oracle.success
    np.testing.assert_allclose(optimal[0],oracle.x,atol=2e-7)
    assert objective(optimal[0]) < objective(radial[0])-.03
    np.testing.assert_array_equal(optimal[1],0.)
