"""Adversarial fixtures for site-block counting and season-matched station reassignments."""
import numpy as np

from research.activation_transfer.shape_transfer.support_confirmation.noaa.run import station_derangement
from research.activation_transfer.shape_transfer.support_confirmation.noaa.score import cluster_skill,eligible_mask


def test_station_eligibility_counts_unique_sites_instead_of_repeated_origins():
    station=np.repeat(np.arange(4),4);a=np.zeros((16,2),dtype=bool)
    a[:4,0]=True;a[[0,4,8],1]=True
    np.testing.assert_equal(eligible_mask(a,station,minimum=3),[False,True])


def test_station_derangement_preserves_calendar_position_when_input_order_is_scrambled():
    station=np.repeat(np.arange(4),4);origin=np.tile(np.arange(4),4)
    perm=np.random.default_rng(331).permutation(16);station=station[perm];origin=origin[perm]
    other=station_derangement(station,origin,12)
    np.testing.assert_array_equal(origin[other],origin)
    assert np.all(station[other]!=station)
    assert len(np.unique(other))==16
    for s in range(4):assert len(np.unique(station[other][station==s]))==1


def test_station_equal_skill_reweights_four_repeated_active_origins_as_one_site():
    station=np.repeat([0,1],4);active=np.zeros((8,1),dtype=bool);active[:5]=True
    truth=np.ones((8,1,1));pred=np.zeros_like(truth);pred[:4]=1.
    draws=np.array([[1,1],[2,0],[0,2]])
    freq=np.ones((3,1),dtype=int)
    flat,b1,_=cluster_skill(active,pred,truth,station,draws,freq,False)
    equal,b2,_=cluster_skill(active,pred,truth,station,draws,freq,True)
    np.testing.assert_allclose([flat,equal],[.8,.5])
    np.testing.assert_allclose(b1,[.8,1,0]);np.testing.assert_allclose(b2,[.5,1,0])
