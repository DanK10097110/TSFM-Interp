"""Selected-edit and output-reach decoys: full-FFN reach is not unit reach."""
import numpy as np
import pytest
from experiments.feature_search.ffn_selectivity_causal_audit import edit_audit


def test_zero_or_dead_selected_edit_does_not_inherit_full_FFN_reach():
    background=np.ones((2,4));clean=background.copy();clean[:,2]+=2
    forecast=np.array([1.,2.])
    zero=edit_audit(background,clean,clean,[0],forecast,forecast)
    assert zero['actual_selected_edit_L2']==0 and not zero['output_reach_gt_1e6']
    patched=clean.copy();patched[:,2]=background[:,2]
    dead=edit_audit(background,clean,patched,[2],forecast,forecast)
    assert dead['actual_selected_edit_L2']==pytest.approx(np.sqrt(8))
    assert dead['selected_edit_nonzero'] and not dead['output_exactly_changed'] and not dead['output_reach_gt_1e6']
    live=edit_audit(background,clean,patched,[2],forecast,forecast+.1)
    assert live['output_reach_gt_1e6'] and live['output_reach_relative']==pytest.approx(.05)


def test_selected_support_rejects_collateral_or_wrong_reset():
    background=np.ones((2,4));clean=background+2;patched=clean.copy();patched[:,1]=1
    edit_audit(background,clean,patched,[1],np.ones(2),np.zeros(2))
    collateral=patched.copy();collateral[:,0]=1
    with pytest.raises(ValueError,match='Unselected'):
        edit_audit(background,clean,collateral,[1],np.ones(2),np.zeros(2))
    wrong=patched.copy();wrong[:,1]=0
    with pytest.raises(ValueError,match='Selected coordinates'):
        edit_audit(background,clean,wrong,[1],np.ones(2),np.zeros(2))
