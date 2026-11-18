"""Known-answer and deliberate-decoy tests for the no-oracle adapter."""
import numpy as np
import pytest
from experiments.feature_search.fourier_adapter import (
    decompose, groups, grouping_orders, combine, interaction, first_crossing,
    conditional_interval, classical,
)
from experiments.feature_search.fourier_poc import Bank, plan, synthetic


def test_reconstruction_including_trend_dc_nyquist_and_offgrid():
    t = np.arange(128)
    x = 4+.02*t+np.cos(np.pi*t)+np.sin(2*np.pi*5.3*t/128)
    for policy in ['mean', 'linear']:
        leaves = decompose(x, policy)
        np.testing.assert_allclose(leaves.sum(0), x, atol=1e-12)
        assert np.linalg.norm(leaves[-1]) > 1
        assert abs(leaves[0].mean()-x.mean()) < 1e-12
    with pytest.raises(ValueError): decompose(np.full(128, np.nan))


def test_all_partitions_preserve_leaves_and_nesting():
    x = np.random.default_rng(2).normal(size=128)
    leaves = decompose(x)
    orders, matching = grouping_orders(leaves, 11)
    for order in orders.values():
        previous = None
        for count in [1, 2, 4, 8]:
            parts = groups(order, count)
            assert sorted(v for group in parts for v in group) == list(range(8))
            np.testing.assert_allclose(sum(leaves[list(g)].sum(0) for g in parts), x, atol=1e-12)
            if previous:
                assert all(set(parts[2*j]) | set(parts[2*j+1]) == set(parent) for j, parent in enumerate(previous))
            previous = parts
    assert matching['energy_match_valid'] == (matching['energy_fraction_max_mismatch'] <= .1)
    with pytest.raises(ValueError): groups(np.zeros(8), 2)


def test_affine_closure_and_bias_decoy():
    x = np.random.default_rng(3).normal(size=128)
    leaves = decompose(x)
    f = lambda v: 3+np.arange(1, 33)*v.mean()+v[-32:]
    zero = f(np.zeros(128))
    preds = np.array([f(leaf) for leaf in leaves])
    np.testing.assert_allclose(combine(preds, zero), f(x), atol=1e-12)
    assert not np.allclose(combine(preds, zero, corrected=False), f(x))
    np.testing.assert_allclose(interaction(f(x), f(leaves[0]), f(leaves[1:].sum(0)), zero), 0, atol=1e-12)


def test_input_only_plan_and_bank_identity():
    x = np.random.default_rng(8).normal(size=128)
    bank = Bank(); first = plan(x, 'all', 1, bank)
    size = len(bank.inputs); second = plan(x, 'all', 1, bank)
    assert first == second and len(bank.inputs) == size
    for scheme in first['schemes'].values():
        np.testing.assert_array_equal(bank.inputs[scheme['1'][0]['bank_id']], x.astype(np.float32))
    assert bank.add(np.zeros(128)) == bank.add(-np.zeros(128))


def test_nonmonotonic_crossing_and_censoring():
    assert first_crossing({'1': .7, '2': .2, '4': .4, '8': .1}) == 2
    assert first_crossing({'1': .7, '2': .4, '4': .3, '8': .26}) is None


def test_uncertainty_resamples_sources_not_dependent_rows():
    original = conditional_interval([0., 2.], ['a', 'b'], draws=99)
    duplicated = conditional_interval([0., 0., 0., 2.], ['a', 'a', 'a', 'b'], draws=99)
    assert original == duplicated
    assert original['mean'] == 1 and original['independent_sources'] == 2


def test_synthetic_phase_pairs_have_equal_spectra_and_valid_crest():
    records, values = synthetic()
    for seed in sorted(set(r['seed'] for r in records)):
        ix = {r['family']: i for i, r in enumerate(records) if r['seed'] == seed and r['family'] != 'distractor'}
        a = values[ix['stationary']][:128]
        b = values[ix['phase_matched']][:128]
        np.testing.assert_allclose(abs(np.fft.rfft(a)), abs(np.fft.rfft(b)), atol=1e-11)
        assert records[ix['phase_matched']]['phase_control_valid']
        assert not np.allclose(a, b)
    x = np.sin(2*np.pi*5*np.arange(-128, 0)/128)
    np.testing.assert_allclose(classical(x), np.sin(2*np.pi*5*np.arange(32)/128), atol=1e-10)
