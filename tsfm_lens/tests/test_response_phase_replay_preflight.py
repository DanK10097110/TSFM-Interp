"""Original batch ordering, bitwise replay and fail-closed recovery eligibility."""
import numpy as np
from experiments.feature_search.response_phase_replay_preflight import original_batch,array_hash,differences,classify


def test_original_batch_preserves_source_list_order_and_row():
    """Non-sorted node IDs expose accidental sorting and size-only reconstruction."""
    nodes=[dict(index=i,source_id=s) for i,s in [(4,0),(30,1),(7,0),(1,0),(9,0),(8,0)]]
    assert original_batch(nodes,0,1)==([4,7,1,9],2)
    assert original_batch(nodes,0,8)==([8],0)
    x=np.arange(16,dtype=np.float32).reshape(4,4)
    assert array_hash(x)!=array_hash(x[[1,0,2,3]])
    assert array_hash(x)!=array_hash(x.reshape(2,8))


def test_differences_keep_one_ulp_and_full_arrays():
    """A tiny difference must remain nonexact even when ordinary allclose accepts it."""
    x=np.ones((4,32),np.float32);y=x.copy();y[2,7]=np.nextafter(y[2,7],np.float32(2))
    result=differences(y,x)
    assert not result['exact'] and result['unequal_entries']==1
    assert result['max_abs']==float(y[2,7]-x[2,7])
    np.testing.assert_allclose(result['l2'],result['max_abs'],atol=0,rtol=0)
    assert differences(x,x)['exact']


def test_recovery_needs_four_exact_original_repeats_and_reproduced_geometry_effect():
    """Neither unexplained old mismatch nor unreproduced failure authorizes a child run."""
    good=dict(same_geometry_exact=True,current_prefix_exact=True,single_forward=True,repeat_exact=True,target_query_identical=True,
              mixed_target_vs_historical={'exact':False})
    records=[dict(good) for _ in range(4)]
    assert classify(records)['eligible']
    assert not classify(records[:3])['eligible']
    for key in ('same_geometry_exact','current_prefix_exact','single_forward','repeat_exact','target_query_identical'):
        bad=[dict(good) for _ in range(4)];bad[1][key]=False
        assert not classify(bad)['eligible']
    unchanged=[dict(good,mixed_target_vs_historical={'exact':True}) for _ in range(4)]
    assert not classify(unchanged)['eligible']
