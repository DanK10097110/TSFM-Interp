import numpy as np

from research.activation_transfer.moment.crosscoder.architecture_check.score import group_summary


def test_false_alarm_on_silent_feature_counts_in_pooled_skill():
    truth = np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.]])
    pred = np.array([[1.,2.],[1.,2.],[1.,2.],[1.,2.]])
    arms = dict(state_exact=pred,direct=truth,explicit_gate=truth,raw32=truth)
    summary = group_summary(truth,arms,np.array([8,9]),np.array([0,0,1,1]))
    assert summary['n_bootstrap_clusters'] == 2
    assert summary['never_fired'] == [9]
    estimate=summary['arms']['state_exact']
    assert estimate['macro_skill']['point'] == 1.
    assert estimate['pooled_skill']['point'] == -3.
    assert estimate['false_alarm_energy_on_never_fired'] == 16.
