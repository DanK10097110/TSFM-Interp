"""Known answers distinguish relative per-atom skill, pooled energy and pooled absolute error."""
import numpy as np

from research.activation_transfer.moment.review.check import summary


def test_good_high_energy_atom_and_bad_low_energy_atom():
    site = np.arange(8)
    event_row = np.repeat(site, 2)
    atom = np.tile([0,1], len(site))
    source = np.column_stack((-np.ones(len(atom)), np.ones(len(atom)))) * (1+3*atom[:,None])
    estimate = np.where(atom[:,None]==1, source, 0.) + 104.
    actual = summary(source, {"candidate":estimate}, event_row, atom, np.ones(len(atom),bool), site, True)["candidate"]
    np.testing.assert_allclose(
        [actual["macro"], actual["energy_weighted"], actual["l1_skill"]],
        [1/2, 16/17, 4/5], rtol=1e-12,
    )
