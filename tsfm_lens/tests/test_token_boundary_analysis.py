"""Reject a curvature ratio made from nearly uniform FP32 probabilities."""
import numpy as np
from experiments.feature_search.token_boundary_analysis import precision_eligible


def test_probability_roundoff_is_not_evidence_of_nonsmoothness():
    base=np.full(4096,1/4096,dtype=np.float32)
    tiny=base.copy();tiny[0]=np.nextafter(tiny[0],np.float32(np.inf))
    assert not precision_eligible(base,base,tiny)[0]
    strong=base.copy();strong[0]+=.01;strong[1:]-=.01/4095
    assert precision_eligible(base,base,strong)[0]
