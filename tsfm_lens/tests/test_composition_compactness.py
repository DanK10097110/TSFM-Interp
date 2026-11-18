"""Fixed-fold, uncentered-projector and reused-cohort screening decoys."""
import numpy as np
import pytest
from experiments.feature_search.composition_compactness import fit_crossfold,decide,failed_execution,IDS,RANKS


def delta_fixture():
    rng=np.random.default_rng(21)
    return rng.normal(size=(16,4,12)),np.asarray(IDS)


def test_folds_use_seed_identity_and_projectors_are_shared_across_tokens():
    delta,ids=delta_fixture()
    projected,mean,valid,records=fit_crossfold(delta,ids)
    assert all(valid.values())
    for record in records:
        evaluation=np.flatnonzero(ids%2==record['evaluation_parity'])
        assert len(record['training_ids'])==len(record['evaluation_ids'])==8
        assert all(seed%2!=record['evaluation_parity'] for seed in record['training_ids'])
        for rank in RANKS:
            basis=np.asarray(record['bases'][str(rank)])
            np.testing.assert_allclose(projected[rank][evaluation],delta[evaluation]@basis.T@basis,atol=1e-12)
    permutation=np.random.default_rng(8).permutation(16)
    reordered,other,_,_=fit_crossfold(delta[permutation],ids[permutation])
    inverse=np.argsort(permutation)
    for rank in RANKS:np.testing.assert_allclose(reordered[rank][inverse],projected[rank],atol=1e-12)
    np.testing.assert_allclose(other[inverse],mean,atol=1e-12)


def test_heldout_delta_changes_cannot_change_its_training_basis():
    delta,ids=delta_fixture()
    _,baseline,_,records=fit_crossfold(delta,ids)
    modified=delta.copy();modified[ids%2==0]+=100.
    _,mean,_,changed=fit_crossfold(modified,ids)
    first=next(row for row in records if row['evaluation_parity']==0)
    second=next(row for row in changed if row['evaluation_parity']==0)
    assert first['training_ids']==second['training_ids']
    np.testing.assert_array_equal(first['bases']['4'],second['bases']['4'])
    np.testing.assert_array_equal(baseline[ids%2==0],mean[ids%2==0])


def test_uncentered_rank_one_mean_is_preserved_and_null_ranks_unscorable():
    direction=np.arange(1.,13.)
    delta=np.broadcast_to(direction,(16,4,12)).copy()
    projected,mean,valid,records=fit_crossfold(delta,np.asarray(IDS))
    assert valid[1] and not valid[2] and not valid[4]
    np.testing.assert_allclose(projected[1],delta,atol=1e-12)
    np.testing.assert_allclose(mean,delta,atol=1e-12)
    assert all(row['numerical_rank']==1 for row in records)
    assert all('2' not in row['bases'] and '4' not in row['bases'] for row in records)
    bad_ids=np.asarray(IDS).copy();bad_ids[-1]=bad_ids[0]
    with pytest.raises(ValueError,match='distinct'):
        fit_crossfold(delta,bad_ids)


def test_all_point_gates_mean_control_and_whole_rank_geometry_validity():
    native=np.ones(16);full=np.full(16,.1);mean=np.full(16,.3)
    loss={r:np.full(16,.04) for r in RANKS};random={r:np.full(16,.2) for r in RANKS}
    valid={r:True for r in RANKS}
    result=decide(native,full,mean,loss,random,valid)
    assert result['smallest_passing_rank']==1
    assert result['ranks']['1']['recovery']>1  # Full edit is a reference, not a bound.
    valid[1]=False
    assert decide(native,full,mean,loss,random,valid)['smallest_passing_rank']==2
    assert decide(native,full,np.full(16,.05),loss,random,valid)['smallest_passing_rank'] is None
    loss[2]=np.full(16,.061);loss[4]=np.full(16,.2)
    assert decide(native,full,mean,loss,random,valid)['smallest_passing_rank'] is None
    assert set(result['ranks'])=={'1','2','4'} and 'claims' not in result


def test_nonfinite_or_missing_seed_cannot_silently_drop_or_qualify():
    native=np.ones(16);full=np.full(16,.1);mean=np.full(16,.3)
    loss={r:np.full(16,.04) for r in RANKS};random={r:np.full(16,.2) for r in RANKS};valid={r:True for r in RANKS}
    loss[1][3]=np.nan
    result=decide(native,full,mean,loss,random,valid)
    assert not result['ranks']['1']['valid_all16'] and not result['ranks']['1']['passed']
    loss[2]=loss[2][:-1]
    with pytest.raises(ValueError,match='missing'):
        decide(native,full,mean,loss,random,valid)


def test_resource_failure_masks_conditional_passing_rank():
    conditional=dict(smallest_passing_rank=1,ranks={'1':{'passed':True}})
    result=failed_execution(conditional,'peak memory')
    assert result['smallest_passing_rank'] is None
    assert not result['execution_valid'] and result['status']=='failed'
    assert result['conditional_smallest_rank']==1
