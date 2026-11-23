import numpy as np

from research.activation_transfer.moment.dictionary_transfer.more_seeds.run import ORIGINAL, displacements
from research.activation_transfer.moment.dictionary_transfer.more_seeds.score import summarize_receiver


def test_native_gate_and_donor_norm_give_distinct_physical_writes():
    orig = dict(rows=np.array([0, 1]), cols=np.array([0, 0]), z=np.array([[2.], [3.]]), donor_norm=np.array([2., 3.]))
    match = dict(target_z=np.array([[0.], [7.]]), single_vector=np.array([[0., 1.]]), group_vector=np.array([[.25, -.5]]))
    moves = displacements(orig, match)
    np.testing.assert_allclose(moves['corr_own'], [[0, 0], [0, -7]])
    np.testing.assert_allclose(moves['corr_forced'], [[0, -2], [0, -3]])
    np.testing.assert_allclose(moves['group16'], [[-.5, 1.], [-.75, 1.5]])


def test_equal_feature_skill_and_pooled_celsius_preserve_different_weights():
    donor = np.array([[1., -1.], [10., -10.]])
    matched = np.array([[1., -1.], [0., 0.]])
    orig = dict(rows=np.arange(2), cols=np.arange(2), station=np.arange(2), atoms=np.array([9, 13]), z=np.eye(2),
                target_z=np.eye(2), input_sd=np.ones(2), noop=np.zeros((2, 2)), donor=donor,
                corr_own=matched, corr_forced=matched, group16=donor, corr=np.ones(2), group_r2=np.ones(2))
    res = summarize_receiver(ORIGINAL, orig, np.arange(2), np.ones((3, 2)), np.ones((3, 2)))
    row = res['metrics']['celsius_centered']['arms']['corr_own']
    assert row['macro_skill']['point'] == .5
    np.testing.assert_allclose(row['pooled_skill'], 1/101)
    np.testing.assert_allclose(row['error_rms'], np.sqrt(50))
