import numpy as np
import pytest
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import groups
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import agreements,native_directions
from research.activation_transfer.moment.crosscoder.cross_family_directions.precision import four


def test_native_intervals_cover_all_480_without_gaps():
    edges=np.arange(0,481,16);spans=np.c_[edges[:-1],edges[1:]]
    pairs=groups(spans)
    np.testing.assert_array_equal(pairs,np.arange(30).reshape(15,2))
    # A gap inside one pooled window leaves its outside endpoints unchanged.
    malformed=spans.copy();malformed[5,0]+=1
    with pytest.raises(AssertionError):groups(malformed)


def test_other_coordinate_excludes_self_and_centering_removes_level():
    s=np.array([[[1.,2.,3.],[3.,2.,1.],[1.,0.,1.]]]);t=s+15
    same,wrong=agreements(s,t,np.array([[1,1,0]],dtype=bool))
    np.testing.assert_allclose(same[0,:2],[1.,1.])
    np.testing.assert_allclose(wrong[0,:2],[-1.,-1.]);assert np.isnan(wrong[0,2])


def test_relative_view_scales_are_restored_in_true_hidden_units():
    params=dict(source_decoder=np.array([[2.,1.]]),target_decoder=np.array([[.2,3.]]),scale=np.array([.05,4.]))
    a,b=native_directions(params)
    np.testing.assert_allclose(a,[[.1,.05]]);np.testing.assert_allclose(b,[[.8,12.]])


def test_precision_uses_four_distinct_firing_coordinates_and_matched_other():
    atoms=np.tile(np.arange(5),(20,1));dose=(atoms<4).astype(float)
    rows,i,j=four(np.tile([0,1],10),atoms,dose)
    np.testing.assert_array_equal(rows,np.repeat(np.arange(20),4))
    assert (dose[rows,i]>0).all() and (dose[rows,j]>0).all() and (i!=j).all()
    for group in i.reshape(20,4):assert len(np.unique(group))==4
