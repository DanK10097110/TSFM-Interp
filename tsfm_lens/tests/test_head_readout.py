"""Native head capture contracts and frozen-probe known answers with decoys."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch


def _module():
    """Load the standalone native-head instrument."""
    root=Path(__file__).parents[1]/'experiments/feature_search';sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('head_readout',root/'head_readout.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_exact_head_container_last_position_and_ragged_rejection():
    """A same-element-count wrong shape must never be silently reshaped."""
    m=_module();x=torch.arange(2*4*8).reshape(2,4,8).float()
    assert torch.equal(m.head_tensor((x,),batch=2,positions=4,width=8),x[:,-1])
    with pytest.raises(TypeError):m.head_tensor([x],batch=2,positions=4,width=8)
    with pytest.raises(TypeError):m.head_tensor((x,x),batch=2,positions=4,width=8)
    with pytest.raises(ValueError):m.head_tensor((x.reshape(2,8,4),),batch=2,positions=4,width=8)
    with pytest.raises(ValueError):m.head_tensor((x[:1],),batch=2,positions=4,width=8)


def test_frozen_head_probe_signal_and_heldout_label_decoy():
    """Train signal beats loud nuisance and fresh labels never enter the fit API."""
    m=_module();rng=np.random.default_rng(5);old=rng.normal(size=(240,9));fresh=rng.normal(size=(80,9));old[:,0]*=1000;fresh[:,0]*=1000;targets=old[:,4:8];truth=fresh[:,4:8];train=np.arange(240)<120;validation=(np.arange(240)>=120)&(np.arange(240)<180)
    fit=m.fit_frozen_head(old,fresh,targets,train,validation)
    altered=targets.copy();altered[180:]*=-100;decoy=m.fit_frozen_head(old,fresh,altered,train,validation)
    assert np.array_equal(fit['old_prediction'],decoy['old_prediction'])
    assert np.array_equal(fit['fresh_prediction'],decoy['fresh_prediction'])
    assert np.mean((fit['fresh_prediction']-truth)**2)<1e-6
    rebuilt=(fresh-fit['scaler_mean'])/fit['scaler_scale']@fit['coefficients'].T+fit['intercept']
    assert np.allclose(rebuilt,fit['fresh_prediction'],atol=1e-12,rtol=1e-12)
