"""Synthetic controls with decoys and analytically known answers."""
from itertools import permutations

import numpy as np

from research.activation_transfer.shape_transfer.support_confirmation.run import limit_norm
from research.activation_transfer.shape_transfer.support_confirmation.score import reduce_draws
from research.activation_transfer.shape_transfer.support_confirmation.supports import count_overlap,rank_frequency,select_supports


def test_frequency_uses_only_positive_occurrences_and_breaks_ties_by_feature_id():
    observed=np.array([[3,1,5],[0,2,4],[1,3,5],[3,2,4]])
    rank,counts=rank_frequency(observed,8)
    np.testing.assert_array_equal(counts,[1,2,2,3,2,2,0,0])
    np.testing.assert_array_equal(rank,[3,1,2,4,5,0,6,7])


def test_low_overlap_is_global_derangement_and_ladder_has_exact_membership():
    rng=np.random.default_rng(8419302);frequent=np.arange(150)[::-1]
    common=np.arange(8)
    native=np.stack([np.r_[common,rng.choice(np.arange(8,100),24,replace=False)] for _ in range(5)])
    arms,indices=select_supports(native,frequent)
    feasible=[]
    for perm in permutations(range(5)):
        if any(i==p for i,p in enumerate(perm)):continue
        feasible.append(sum(len(set(u)&set(v)) for u,v in zip(native,native[list(perm)])))
    assignment=indices['minimum_overlap']
    assert set(assignment)==set(range(5))
    assert (assignment!=np.arange(5)).all()
    assert sum(count_overlap(native,arms['minimum_overlap']))==min(feasible)
    for k in (0,8,16,24):
        mixed=arms[f'own_{k}']
        assert all(len(set(r))==32 for r in mixed)
        np.testing.assert_equal(count_overlap(mixed,native),k)
    for i,own in enumerate(native):
        np.testing.assert_array_equal(arms['own_0'][i],frequent[~np.isin(frequent,own)][:32])


def test_norm_cap_keeps_direction_and_preserves_writes_already_inside():
    v=np.array([[3.,4.],[-.15,.2],[0.,0.]])
    actual,original,scale=limit_norm(v,np.array([2.,1.,1.]))
    np.testing.assert_allclose(actual,[[1.2,1.6],[-.15,.2],[0.,0.]])
    np.testing.assert_allclose(original,[5,.25,0])
    np.testing.assert_allclose(scale,[.4,1,1])


def test_seed_contrasts_average_outcomes_and_pair_each_bootstrap_draw():
    first=(.3,np.array([.2,.54]),np.array([.4,.1]))
    second=(.7,np.array([.6,.06]),np.array([0.,.9]))
    avg=reduce_draws([first,second])
    np.testing.assert_allclose(avg[0],.5);np.testing.assert_allclose(avg[1],[.4,.3]);np.testing.assert_allclose(avg[2],[.2,.5])


def test_orthogonalizing_removes_shared_axis_and_discards_rank_deficient_rows():
    from research.activation_transfer.shape_transfer.support_confirmation.residual import orthogonal_part
    # A tilted candidate axis retains only e2. The second axis lies wholly in removal span.
    e=np.eye(4)
    axes=np.stack([(e[0]+e[1])/np.sqrt(2),e[2]])[None]
    remove=np.stack([e[0],e[2]])[None]
    retained,strength,rank,cosines=orthogonal_part(axes,remove)
    np.testing.assert_array_equal(rank,[1])
    np.testing.assert_allclose(strength,[[1/np.sqrt(2),0]],atol=1e-12)
    np.testing.assert_allclose(np.sort(cosines),[[1/np.sqrt(2),1]],atol=1e-12)
    np.testing.assert_allclose(retained[0].T@retained[0],np.outer(e[1],e[1]),atol=1e-12)


def test_derivative_swap_keeps_requested_waveform_and_dc_supplies_only_its_mean():
    from research.activation_transfer.shape_transfer.support_confirmation.derivative import fit_goal
    native=np.eye(2)[None];wrong=np.array([[[-1.,0.],[0.,1.]]]);wave=np.array([[1.,-1.]])
    budget=np.array([10.]);expected=1/1.001
    read=lambda j,m:fit_goal(j,native,wave,budget,1.,m)
    np.testing.assert_allclose(read(native,'balanced'),[[expected,-expected]])
    np.testing.assert_allclose(read(wrong,'balanced'),[[-expected,-expected]])
    np.testing.assert_allclose(read(wrong,'dc'),np.zeros_like(wave),atol=1e-10)
