"""Algebraic diagnostic bounds and raw-before-gate recovery safeguards."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.feature_search import response_phase_horizon_recovery as recovery


def test_mse_change_bound_is_tight_for_aligned_perturbation():
    """The cross term cannot be dropped; fixed-predictor NMSE changes are bounded."""
    error=np.arange(1,129,dtype=float).reshape(4,32)/100;delta=.2*error;energy=.64
    observed=(np.sum((error+delta)**2)-np.sum(error**2))/energy
    bound=recovery.mse_change_bound(error,delta,energy)
    np.testing.assert_allclose(bound,observed,rtol=1e-14)
    assert recovery.mse_change_bound(error,np.zeros_like(delta),energy)==0


def test_response_difference_subtracts_background_and_triangle_bounds_it():
    """Background numerical changes contribute to every signal's response difference."""
    reference=np.zeros((5,32));current=np.full((5,32),.02);current[0]=-.01
    group=dict(amplitude=.1,cohort='FIT',source='fixture',source_id=0,input_period=19.2,physical_amplitude=.1,base=0,indices=[1,2,3,4])
    delta,rows=recovery.response_bounds(current,reference,[group]);r=rows[0]
    expected=np.sqrt(4*32)*.03
    np.testing.assert_allclose(r['response_delta_l2'],expected,rtol=1e-14)
    np.testing.assert_allclose(r['triangle_l2_bound'],expected,rtol=1e-14)
    np.testing.assert_allclose(r['u'],expected/.8,rtol=1e-14)


def test_failed_prefix_is_persisted_before_gate_rejection():
    """A new capability failure cannot repeat the original missing-delta problem."""
    class Fake:
        native=SimpleNamespace(pipeline=SimpleNamespace(model_prediction_length=64))
        def pair(self,x):return np.zeros((len(x),64),np.float32),np.ones((len(x),32),np.float32),[1,1]
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);recorder=recovery.Recorder(out,Fake());query=np.ones((4,128),np.float32)
        with pytest.raises(ValueError,match='prefix32'):recorder.pair(query,'bad')
        assert (out/'bad.npz').exists() and (out/'query_records.json').exists()
        raw=np.load(out/'bad.npz');np.testing.assert_array_equal(raw['prefix_difference'],-np.ones((4,32)))
        np.testing.assert_array_equal(raw['query'],query)


def test_grant_must_bind_amendment_not_only_old_protocol():
    """The original grant cannot silently authorize a different comparator contract."""
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);recovery.write(out/'protocol.json',{});recovery.write(out/'amendment.json',{})
        grant=dict(protocol_sha256=recovery.sha(out/'protocol.json'),explicit_slot_release=True,slot=2,models=['bolt'])
        recovery.write(out/'grant.json',grant)
        with pytest.raises(PermissionError,match='amendment'):recovery.require_recovery_grant(out,out/'grant.json')
        grant['amendment_sha256']=recovery.sha(out/'amendment.json');recovery.write(out/'grant.json',grant)
        assert recovery.require_recovery_grant(out,out/'grant.json')['slot']==2
