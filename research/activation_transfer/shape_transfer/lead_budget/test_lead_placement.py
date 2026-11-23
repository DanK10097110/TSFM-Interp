import numpy as np

from research.activation_transfer.shape_transfer.lead_budget.run import AUDIT, select


def test_audit_leads_are_held_out_and_selection_finds_the_only_predictive_row():
    rng = np.random.default_rng(9912)
    gradients = rng.normal(size=(16,32,5))
    signal = gradients[:,15].copy()
    for lead, amplitude in zip(AUDIT, [-2, -1, 1, 2]):
        gradients[:,lead] = amplitude * signal
    winner, options, error = select(gradients)
    assert len(options) == 495
    assert not np.isin(options,AUDIT).any()
    assert 15 in winner
    assert error[~(options==15).any(axis=1)].min() > 1000 * error.min()
