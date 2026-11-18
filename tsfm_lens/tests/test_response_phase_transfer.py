"""Known-answer old-only fits, transport, source inference and guarded capture."""
import contextlib
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.feature_search import response_phase_transfer as transfer
from experiments.feature_search.response_phase_study import prediction, basis, J


def test_scalar_uses_waveform_geometry_and_bounds():
    """Anisotropy at a fractional window defeats trace/2 and phase-ratio shortcuts."""
    f=128/24; h=np.array([[.1,.8],[-.2,1.7]])
    amps=np.array([.05,.2]); y=np.stack([prediction(h,a,f) for a in amps])
    fit=transfer.fit_training(y,amps,f)
    ideal=prediction(np.eye(2),1.,f); target=prediction(h,1.,f)
    expected=float(np.dot(ideal.ravel(),target.ravel())/np.dot(ideal.ravel(),ideal.ravel()))
    assert abs(expected-np.trace(h)/2)>.005
    np.testing.assert_allclose(fit['scalar_unconstrained'],expected,atol=1e-14)
    for gain, bounded in [(-.4,0.),(1.7,1.)]:
        fit=transfer.fit_training(np.stack([prediction(gain*np.eye(2),a,f) for a in amps]),amps,f)
        np.testing.assert_allclose(fit['scalar_unconstrained'],gain,atol=1e-14)
        assert fit['scalar_bounded']==bounded
        assert fit['training_nmse']['scalar_unconstrained']<1e-26


def test_complex_full_and_zero_with_even_decoy():
    """Independent normal equations recover fullH; even terms cannot tilt paired fits."""
    f=128/19.2; a=.1; h=np.array([[.3,.7],[-.8,1.2]])
    y=prediction(h,a,f); y[[0,2]]+=.17; y[[1,3]]-=.09
    fit=transfer.fit_training(np.stack([y,2*y]),[a,2*a],f)
    np.testing.assert_allclose(fit['matrices']['full_H'],h,atol=1e-14)
    np.testing.assert_array_equal(fit['matrices']['zero'],np.zeros((2,2)))
    b=basis(f); gram=b.T@b
    q=np.trace(gram@h)/np.trace(gram)
    r=np.trace(J.T@gram@h)/np.trace(gram)
    np.testing.assert_allclose(fit['complex_qr'],[q,r],atol=1e-14)
    np.testing.assert_allclose(fit['matrices']['complex_linear'],q*np.eye(2)+r*J,atol=1e-14)
    assert fit['training_nmse']['full_H']<fit['training_nmse']['complex_linear']
    assert fit['training_nmse']['complex_linear']<fit['training_nmse']['scalar_unconstrained']
    ideal=prediction(np.eye(2),a,f)
    np.testing.assert_allclose(fit['training_nmse']['zero'],np.sum(y*y)/np.sum(ideal*ideal))


def test_delay_transport_sign_and_center():
    """A known delay follows negative phase slope; no refit needed at new frequencies."""
    center=128/19.2; tau=2.; omega=2*np.pi*center/128
    rot=lambda angle:np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    h=.8*rot(-omega*tau)
    np.testing.assert_array_equal(transfer.delayed_map(h,center,center),h)
    for f in (center-.12,center+.12):
        np.testing.assert_allclose(transfer.delayed_map(h,center,f),.8*rot(-2*np.pi*f/128*tau),atol=1e-14)


def test_window_absolute_origin_and_even_decoy():
    """Late-window timestamps must not reset; a DC decoy needs the explicit joint fit."""
    f=128/19.2; a=.1; h=np.array([[.7,.2],[-.1,.9]])
    y=prediction(h,a,f); y[[0,2]]+=.02
    for lo,hi in [(0,16),(16,32),(0,32)]:
        result=transfer.window_decomposition(y,a,f,lo,hi)
        np.testing.assert_allclose(result['H'],h,atol=1e-14)
        assert result['even_nmse']>0
    y[0]+=.4; y[2]-=.4
    result=transfer.window_decomposition(y,a,f,16,32)
    np.testing.assert_allclose(result['H_dc_adjusted'],h,atol=1e-14)
    assert np.linalg.norm(np.asarray(result['H'])-h)>.1


def test_source_bootstrap_and_family_four():
    """Source-paired intervals preserve type balance and widen for the fixed family."""
    types=np.tile(np.arange(4),4); draws=transfer.source_draws(types,1024)
    assert draws.shape==(1024,16) and draws.max()==15
    for kind in range(4): assert np.all(np.sum(types[draws]==kind,axis=1)==4)
    values=np.linspace(-1,3,16); result=transfer.paired_interval(values,draws)
    boot=np.array([sum(values[i] for i in row)/16 for row in draws])
    np.testing.assert_allclose(result['family95_percentile'],np.quantile(boot,[.00625,.99375]))
    assert result['family95_percentile'][0]<result['percentile95'][0]
    assert result['family95_percentile'][1]>result['percentile95'][1]
    assert result['wins']==12 and result['family']==4


def test_fresh_design_and_phase_swap_gate():
    """The prospective six-frequency bank has400 nodes and rejects a phase permutation."""
    bank,nodes,sources,groups,gates=transfer.fresh_design()
    assert bank.shape==(400,128) and len(nodes)==400 and len(groups)==96
    assert [s['seed'] for s in sources]==list(range(4954300,4954316))
    assert [s['background_type'] for s in sources]==list(range(4))*4
    assert gates['nominal_rms_over_peak_max']<1e-5
    assert all(len([g for g in groups if g['source_id']==sid])==6 for sid in range(16))
    wrong=bank.copy(); ids=groups[0]['indices']; wrong[[ids[0],ids[1]]]=wrong[[ids[1],ids[0]]]
    with pytest.raises(ValueError,match='input gate'): transfer.input_gates(wrong,groups)


def test_worker_requires_grant_before_model_load():
    """Absent or mismatched slot grants must fail before checkpoint validation/loading."""
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)
        with patch.object(transfer,'validate',side_effect=AssertionError('Reached native preparation')):
            with pytest.raises(PermissionError): transfer.worker(out,'bolt',None)


class FakeNative:
    """Deterministic receiver for capture instrumentation, never a model-evidence source."""
    name='bolt'
    def __init__(self): self.anchor=None
    def predict(self,x,replacement=None):
        h=np.asarray(x,np.float32)[:,None,:]
        if replacement is not None: h=replacement
        y=h[:,0,:32].copy()
        if self.anchor is not None: y+=(x.mean(1)-self.anchor.mean(1))[:,None]*.5
        return y.astype(np.float32),h


@contextlib.contextmanager
def fake_normalization(native,anchor,policy):
    """Preserve exact self identity while giving native/locked distinct perturbed responses."""
    native.anchor=anchor
    try: yield
    finally: native.anchor=None


def test_capture_saved_arithmetic_and_identity_gate():
    """A locked-policy effect survives capture; a broken self-normalizer is rejected."""
    bank,nodes,_,_,_=transfer.fresh_design(); bank=bank[:25];nodes=nodes[:25]
    p=dict(seconds_per_model=600,worker_cap=3*2**30)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)
        with patch.object(transfer,'normalization',fake_normalization):
            result=transfer.capture_bank(out,FakeNative(),p,bank,nodes)
        assert not any(result['gates'].values()) and result['batch_replay_exact']
        native=np.load(out/'bolt_native_forecast.npy'); locked=np.load(out/'bolt_locked_forecast.npy')
        np.testing.assert_array_equal(native,bank[:,:32])
        expected=bank[:,:32]+(bank.mean(1)-bank[0].mean())[:,None]*.5
        np.testing.assert_array_equal(locked,expected)
        assert np.max(abs(native-locked))>0
    @contextlib.contextmanager
    def broken(native,anchor,policy):
        native.anchor=anchor+1
        try: yield
        finally: native.anchor=None
    with tempfile.TemporaryDirectory() as directory, patch.object(transfer,'normalization',broken):
        with pytest.raises(ValueError,match='identity gate'): transfer.capture_bank(Path(directory),FakeNative(),p,bank,nodes)
