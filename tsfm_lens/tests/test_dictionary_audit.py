"""Audit guards distinguish state fidelity, delta fidelity, effects and null validity."""
import numpy as np
from experiments.feature_search import dictionary_audit as audit


def test_paired_score_cancels_shared_offset_but_detects_erased_response():
    h = np.array([[[10., 2.]], [[8., 2.]]])
    shifted = audit.paired_score(h, h+100.)
    assert shifted['desired_energy'] == 2. and shifted['relative_error'] == 0.
    erased = np.repeat(h.mean(0, keepdims=True), 2, axis=0)
    assert audit.paired_score(h, erased)['relative_error'] == 1.
    assert audit.paired_score(erased, erased)['relative_error'] is None


def test_validity_does_not_turn_invalid_geometry_or_zero_effect_into_inactivity():
    for selected in (0., 2.):
        row = dict(exact_gates=True, causal_selectivity={'sinusoid': dict(reach=True, budgets={'1':
            dict(random_valid=False, response_change_nmse=selected, random_response_change_nmse=selected)})})
        result = audit.validity_cell(row, 'sinusoid', 1)
        assert not result['causal_selectivity_claim_scorable']
        assert not result['descriptive_comparison_numerically_available']
        assert 'random_geometry_invalid_fallback_not_a_null' in result['reasons']
        assert 'not recorded' in result['actual_perturbed_activation_edit_norm']
        assert result['selected_response_change_nonzero'] == (selected > 0)
