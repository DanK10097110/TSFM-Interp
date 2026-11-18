"""Known-answer phase instrument tests, including a high-variance nuisance decoy."""
import importlib.util
from pathlib import Path
import numpy as np


def _module():
    """Load the experiment independently from package installation."""
    path=Path(__file__).parents[1]/'experiments/feature_search/phase_features.py'
    spec=importlib.util.spec_from_file_location('phase_features',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_phase_pair_normalization_and_harmonic_known_answer():
    """Phase shifts preserve normalizers; an offset decoy must not bias phase."""
    m=_module();x,future,records=m.make_phase_corpus()
    diagnostics=m.normalization_diagnostics(x)
    assert diagnostics['mean_pair_max_abs']<1e-6
    assert diagnostics['std_pair_max_abs']<1e-6
    measured=m.harmonic_fit(future)
    expected=np.array([r['phase'] for r in records])
    assert np.max(abs(np.angle(np.exp(1j*(measured['phase']-expected)))))<1e-10
    assert np.max(abs(measured['amplitude']-np.array([r['amplitude'] for r in records])))<1e-10
    assert len(set(r['seed'] for r in records))==192


def test_raw_phase_rowspace_unscales_coefficients():
    """Anisotropic nuisance units confound a standardized-coordinate transplant."""
    m=_module();rng=np.random.default_rng(15);n=400
    phase=rng.uniform(-np.pi,np.pi,n);latent=rng.normal(size=(n,8));latent[:,:2]=np.column_stack([np.sin(phase),np.cos(phase)])
    q,_=np.linalg.qr(rng.normal(size=(8,8)));units=np.diag([.01,.1,1,2,10,100,300,1000]);mix=q@units
    x=latent@mix;train=np.arange(n)<200;valid=(np.arange(n)>=200)&(np.arange(n)<300)
    fit=m.fit_phase_probe(x,phase,train,valid)
    true=np.linalg.inv(mix)[:,:2]
    projection=fit['frame']@fit['frame'].T
    assert np.linalg.norm((np.eye(8)-projection)@true)/np.linalg.norm(true)<.001
    assert np.mean((fit['prediction'][300:]-latent[300:,:2])**2)<1e-5
