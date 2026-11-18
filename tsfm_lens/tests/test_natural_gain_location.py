"""An exactly corrected tone must not hide a newly introduced orthogonal error."""
import numpy as np
from experiments.feature_search.natural_gain_location import partition


def test_tone_gain_and_collateral_harm_partition_exactly():
    t=np.arange(32);f=5.35
    tone=np.sin(2*np.pi*f*t/128)
    basis=np.column_stack([tone,np.cos(2*np.pi*f*t/128)])
    nuisance=np.random.default_rng(1).normal(size=32);nuisance-=basis@np.linalg.lstsq(basis,nuisance,rcond=None)[0]
    nuisance*=np.sqrt(np.mean(tone**2)/np.mean(nuisance**2))
    target=np.ones(32)
    r,_=partition(target+tone,target+2*nuisance,target,f)
    assert np.isclose(r['tone_gain'],np.mean(tone**2))
    assert np.isclose(r['other_gain'],-4*np.mean(tone**2))
    assert np.isclose(r['gain'],-3*np.mean(tone**2))
    assert np.isclose(r['other_update']/r['update_energy'],.8)
