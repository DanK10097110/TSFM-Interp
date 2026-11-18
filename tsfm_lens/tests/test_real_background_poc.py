"""Known-answer controls for source-heldout real-background pilot measurements."""
import numpy as np
import pytest
from experiments.feature_search.real_background_poc import (
    signal, response_metrics, baseline, history_control, donor_indices, screen,
    supervised_screen, DATASETS,
)


def fixture():
    """Three dependent datasets, with disjoint fit/evaluation source IDs."""
    x = np.random.default_rng(8).normal(size=(24, 128))
    metadata = [dict(dataset=dataset, id=f'{dataset}-{j}') for dataset in DATASETS for j in range(8)]
    train = np.array([i for i in range(24) if i % 8 < 4])
    test = np.array([i for i in range(24) if i % 8 >= 4])
    return x, metadata, train, test


def test_response_phase_and_off_frequency_have_known_answer():
    expected = signal(128)[128:]
    perfect = response_metrics(np.stack([expected, 2*expected]), expected)
    np.testing.assert_allclose(perfect['error'], [0, 1], atol=1e-12)
    np.testing.assert_allclose(perfect['amplitude'], [1, 2], atol=1e-12)
    np.testing.assert_allclose(perfect['phase'], 0, atol=1e-12)
    np.testing.assert_allclose(perfect['off_target'], 0, atol=1e-12)
    opposite = response_metrics(-expected[None], expected)
    np.testing.assert_allclose(abs(opposite['phase']), np.pi, atol=1e-12)
    np.testing.assert_allclose(opposite['error'], [4], atol=1e-12)


def test_physical_phase_is_identical_across_histories_and_linear_baseline():
    short, long = signal(128), signal(256)
    np.testing.assert_array_equal(short, long[-160:])
    x, _, _, _ = fixture()
    for scale, shift in [(.1, -5), (10, 5)]:
        for kind in ['fourier', 'seasonal']:
            np.testing.assert_allclose((baseline(scale*x+shift, kind)-shift)/scale,
                                       baseline(x, kind), atol=1e-11)
    response = baseline(x+short[:128], 'fourier')-baseline(x, 'fourier')
    np.testing.assert_allclose(response, np.broadcast_to(short[128:], response.shape), atol=1e-12)


def test_irrelevant_history_preserves_distribution_and_recent_junction_but_changes_old_order():
    x = np.arange(2*256).reshape(2, 256)
    changed = history_control(x)
    np.testing.assert_array_equal(changed[:, 96:], x[:, 96:])
    np.testing.assert_array_equal(np.sort(changed[:, :128]), np.sort(x[:, :128]))
    assert not np.array_equal(changed[:, :96], x[:, :96])
    with pytest.raises(ValueError):
        history_control(x[:, :128])


def test_donor_selection_cannot_use_heldout_perfect_self_match():
    x, _, train, test = fixture()
    near, far = donor_indices(x, train, test)
    assert set(near).issubset(set(train)) and set(far).issubset(set(train))
    assert not set(near).intersection(test)
    assert np.all(near != far)


def test_dataset_gate_cannot_be_rescued_by_pooled_success_or_training_rows():
    _, metadata, train, test = fixture()
    values = np.full(24, 10.)
    values[test[-4:]] = -.1
    result = screen(values, metadata, test, .1)
    assert np.mean(values[test]) > .1 and not result['passed']
    values[test[-4:]] = .2
    assert screen(values, metadata, test, .1)['passed']
    values[test[-1]] = np.nan
    assert not screen(values, metadata, test, .1)['passed']


def test_supervised_predictions_do_not_fit_heldout_targets():
    x, metadata, train, test = fixture()
    interaction = np.random.default_rng(2).normal(size=(24, 2))
    harm = interaction[:, 0]+.1
    first = supervised_screen(x, interaction, harm, train, test, metadata)
    altered = harm.copy(); altered[test] += 100
    second = supervised_screen(x, interaction, altered, train, test, metadata)
    for key in first['predictions']:
        np.testing.assert_array_equal(first['predictions'][key], second['predictions'][key])
    assert second['passed'] is False
