"""Known-answer, leakage and inferential checks for the comparison harness."""

import numpy as np
import pytest

from tsfm_lens.experiments.feature_search.statistics import (
    GroupedSplit, compare_effects, evidence_verdict, fit_ridge, grouped_split,
    incremental_predictivity, paired_effect,
)


def planted(seed=19):
    """Rotated shared concept, model-private concept and input-correlated decoy."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(300), 2)
    latent = rng.normal(size=(300, 3))[groups]
    x, shared, private = latent.T
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    model_a = np.column_stack((shared, private, x)) @ rotation
    model_b = np.column_stack((shared, x)) @ np.array([[0.6, -0.8], [0.8, 0.6]])
    decoy = (3 * x)[:, None]
    return groups, x[:, None], shared, private, model_a, model_b, decoy


def test_known_answer_rotated_shared_private_and_input_decoy():
    """Direct response transfer survives rotation; private response stays private."""
    groups, inputs, shared, private, a, b, decoy = planted()
    split = grouped_split(groups, seed=4)
    for partition in (split.train, split.validation, split.evaluation):
        assert all(np.sum(groups[partition] == g) == 2 for g in np.unique(groups[partition]))
    answers = {}
    for name, features, response in (
        ("shared_a", a, shared), ("shared_b", b, shared),
        ("private_a", a, private), ("private_b", b, private),
        ("decoy", decoy, shared),
    ):
        result = incremental_predictivity(inputs, features, response, groups, split=split,
                                          n_boot=199, n_permutations=199)
        answers[name] = result["effect"]
    for name in ("shared_a", "shared_b", "private_a"):
        assert answers[name]["lo"] > 0.5
        assert answers[name]["p"] == 1 / 200
    assert answers["private_b"]["hi"] < 0.1
    assert answers["decoy"]["hi"] < 0.01
    assert answers["decoy"]["lo"] > -0.01


def test_evaluation_values_do_not_change_ridge_or_validation_choice():
    """A planted confirmation-label leak changes scores but cannot change fit."""
    groups, inputs, shared, _, a, _, _ = planted()
    split = grouped_split(groups)
    model, choices = fit_ridge(a, shared, split)
    changed_x, changed_y = a.copy(), shared.copy()
    changed_x[split.evaluation] = 1e8
    changed_y[split.evaluation] = -1e8
    changed, changed_choices = fit_ridge(changed_x, changed_y, split)
    assert choices == changed_choices
    np.testing.assert_array_equal(model.predict(a), changed.predict(a))
    np.testing.assert_array_equal(model[0].mean_, changed[0].mean_)


def test_group_weighting_plus_one_holm_and_equivalence():
    """Replication inside one generating seed cannot manufacture evidence."""
    groups = np.array([0] * 40 + [1, 2, 3])
    values = np.array([4.] * 40 + [-1., -1., -1.])
    effect = paired_effect(values, np.zeros_like(values), groups, n_boot=99, n_permutations=99)
    assert effect["value"] == 0.25
    assert effect["p"] >= 0.01
    results = compare_effects({"a": (values, values * 0), "b": (values * 0, values * 0)},
                              groups, n_boot=99, n_permutations=99)
    assert results["a"]["p_holm"] >= results["a"]["p"]
    assert results["b"]["p"] == 1
    assert evidence_verdict(results["b"], equivalence_margin=0.1) == "interval_within_margin"
    assert evidence_verdict(dict(lo=-2, hi=2, p_holm=1), equivalence_margin=0.1) == "not_detected"
    assert evidence_verdict(dict(lo=1, hi=2, p_holm=0.01), equivalence_margin=0.1) == "detected"


def test_reject_seed_leakage_and_invalid_inputs():
    """Custom row splits must not separate perturbations of a generating seed."""
    groups, inputs, shared, _, a, _, _ = planted()
    bad = GroupedSplit(np.arange(0, 200), np.arange(200, 401), np.arange(401, 600))
    with pytest.raises(ValueError, match="seeds cross"):
        incremental_predictivity(inputs, a, shared, groups, split=bad)
    with pytest.raises(ValueError):
        paired_effect([1, np.nan, 3], [0, 0, 0], [0, 1, 2])
    for kwargs in ({"n_boot": 1.5}, {"n_permutations": True}):
        with pytest.raises(ValueError):
            paired_effect([1, 2, 3], [0, 0, 0], [0, 1, 2], **kwargs)
    for labels in ([0, np.nan, 2], ["", "a", "b"], [None, 1, 2]):
        with pytest.raises(ValueError):
            paired_effect([1, 2, 3], [0, 0, 0], labels)
