"""Distinguish true attenuation from phase error and amplification."""
import numpy as np
from experiments.feature_search.attenuation_analysis import project


def test_attenuation_phase_and_amplification():
    """Orthogonal integer-cycle tones supply confusable known-answer controls."""
    t=np.arange(32);s=np.sin(2*np.pi*4*t/128);c=np.cos(2*np.pi*4*t/128)
    a=project(.4*s,s,4);np.testing.assert_allclose(a['gain'],.4);assert a['bounded_scalar_residual']<1e-20
    b=project(c,s,4);np.testing.assert_allclose(b['bounded_scalar_residual'],1);assert b['off_frequency_residual']<1e-20
    d=project(2*s,s,4);assert d['outside_conservative_range'];np.testing.assert_allclose(d['bounded_scalar_residual'],1);assert d['unrestricted_scalar_residual']<1e-20
