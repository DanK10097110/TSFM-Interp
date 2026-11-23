import numpy as np
from research.activation_transfer.shape_transfer.rolling.fixed_2622.intervene import project


def test_signed_projection_respects_old_budget_in_both_directions():
    basis=np.array([2.,-1.,0.])
    assert project(basis,-3*basis,.4)==-.4
    assert project(basis,.2*basis,.4)==.2
    assert project(np.zeros(3),basis,.4)==0.
