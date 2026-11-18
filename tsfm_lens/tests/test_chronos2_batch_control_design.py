"""Fixed-shape target identity and a genuinely mixed-group decoy."""
import numpy as np
from experiments.feature_search import chronos2_batch_control_design as design


def test_all_target_slots_hold_exact_context_and_kernel_shape():
    contexts=np.random.default_rng(1).normal(size=(512,128)).astype(np.float32)
    base,cases=design.fixed_batches(contexts)
    assert len(cases)==16
    assert {r['target'] for r in cases}==set(range(8))
    for row in cases:
        assert row['batch'].shape==base.shape==(8,128)
        np.testing.assert_array_equal(row['batch'][row['target']],base[row['target']])
        assert np.isfinite(row['batch']).all()


def test_installed_unique_group_mask_is_independent_but_bad_group_decoy_is_not():
    # Different valid rowlevels make groupmixing unmistakable.
    contexts=np.arange(512,dtype=np.float64)[:,None]+np.arange(128)[None,:]/100
    base,cases=design.fixed_batches(contexts)
    unique=design.known_group_forecast(base,np.arange(8))
    bad=design.known_group_forecast(base,np.zeros(8,dtype=int))
    bad_deltas=[]
    for row in cases:
        target=row['target']
        assert design.known_group_forecast(row['batch'],np.arange(8))[target]==unique[target]
        bad_deltas.append(abs(design.known_group_forecast(row['batch'],np.zeros(8,dtype=int))[target]-bad[target]))
    assert max(bad_deltas)>1
