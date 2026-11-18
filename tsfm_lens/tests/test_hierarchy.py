"""Known-answer hierarchy controls with a geometrically misleading decoy."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).parents[1] / "experiments/feature_search/hierarchy.py"
_spec = importlib.util.spec_from_file_location("hierarchy", _path)
hierarchy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hierarchy)


def test_cosine_recovers_planted_groups_and_response_size_correction():
    """A larger group must not win merely because it contains more children."""
    atoms = np.array([[1, .01], [1, -.01], [1, 0], [.01, 1], [-.01, 1]])
    membership = hierarchy.cosine_hierarchy(atoms, 2)
    assert np.argmax(membership, axis=1).tolist()[:3].count(
        np.argmax(membership[0])) == 3
    assert np.argmax(membership[0]) != np.argmax(membership[-1])
    np.testing.assert_allclose(hierarchy.parent_responses(np.ones((2, 5)), membership), 1)


def test_decoder_atoms_are_training_rows():
    """Transpose or code-stacking regressions break the atom membership shape."""
    atoms = np.random.default_rng(4).normal(size=(9, 4))
    weights = hierarchy.decoder_hierarchy(atoms, 3, seed=1, steps=3)
    assert weights.shape == (9, 3)
    assert np.isfinite(weights).all()
    assert np.all(np.count_nonzero(weights, axis=1) <= 1)


def test_gate_refuses_seed_instability_and_missing_measurement():
    """Mean-only stability has a strong decoy that hides one unstable seed."""
    assert hierarchy.promotion_gate([.03]*3, [0]*3, [.9]*3)
    assert not hierarchy.promotion_gate([.03]*3, [0]*3, [1, 1, .7])
    assert not hierarchy.promotion_gate([.03]*3, [0]*3, [])
    assert not hierarchy.promotion_gate([.03]*3, [0, 0, -.03], [.9]*3)


def test_dead_atom_is_not_silently_assigned():
    """Invalid decoder inputs fail instead of becoming an artificial cluster."""
    with pytest.raises(ValueError, match="zero"):
        hierarchy.cosine_hierarchy(np.array([[1, 0], [0, 1], [0, 0]]), 2)


def test_random_control_preserves_parent_sizes_but_moves_atoms():
    """A random control must retain the confusable unequal-size groups."""
    membership = np.eye(2)[[0, 0, 0, 1, 1]]
    control = hierarchy.random_hierarchy(membership, 17)
    np.testing.assert_equal(control.sum(axis=0), membership.sum(axis=0))
    assert not np.array_equal(control, membership)
