"""Known-answer controls for expanded decomposition and matched training."""
import numpy as np
from experiments.feature_search.coherence_study import parts, Bank
from experiments.feature_search.coverage_training import dataset, CYCLES


def test_shift_splits_nearby_pair_while_original_does_not():
    t=np.arange(-128,0);a=np.sin(2*np.pi*5*t/128);b=np.cos(2*np.pi*6*t/128)
    normal=parts(a+b,'fft');shifted=parts(a+b,'shift')
    np.testing.assert_allclose(normal.sum(0),a+b,atol=1e-12)
    np.testing.assert_allclose(normal[1],a+b,atol=1e-12)
    np.testing.assert_allclose(shifted[0],a,atol=1e-12)
    np.testing.assert_allclose(shifted[1],b,atol=1e-12)


def test_bridge_and_long_context_are_lossless_with_nyquist():
    for n in [128,256]:
        t=np.arange(n);x=4+.02*t+np.cos(np.pi*t)+np.sin(2*np.pi*5.35*t/128)
        for mode in ['fft','shift','bridge']:
            p=parts(x,mode);np.testing.assert_allclose(p.sum(0),x,atol=1e-12)
            assert np.linalg.norm(p[-1])>1


def test_bank_does_not_alias_different_lengths_or_values(tmp_path):
    b=Bank();a=b.add(np.zeros(128));c=b.add(np.zeros(256));d=b.add(np.ones(128))
    assert len({a,c,d})==3 and b.add(-np.zeros(128))==a
    b.save(tmp_path/'bank.npz')
    from experiments.feature_search.coherence_study import load_bank
    recovered=load_bank(tmp_path/'bank.npz')
    assert [len(x) for x in recovered]==[128,256,128]


def test_coverage_changes_pairs_but_exactly_matches_component_exposures_and_singles():
    x0,y0,m0=dataset(4,6,False);x1,y1,m1=dataset(4,6,True)
    heldout={(0,2),(1,3)}
    assert not any(tuple(r['components']) in heldout for r in m0)
    assert any(tuple(r['components']) in heldout for r in m1)
    for block in range(6):
        for f in range(4):
            assert sum(f in r['components'] for r in m0 if r['block']==block)==3
            assert sum(f in r['components'] for r in m1 if r['block']==block)==3
    ids=[i for i,r in enumerate(m0) if len(r['components'])==1]
    np.testing.assert_array_equal(x0[ids],x1[ids]);np.testing.assert_array_equal(y0[ids],y1[ids])


def test_holm_preserves_full_family_and_monotonicity():
    from experiments.feature_search.coherence_analysis import holm
    rows=[dict(p=p,mean=.2) for p in [.02,.001,.5,.004]]
    result=holm(rows)
    np.testing.assert_allclose([r['holm_p'] for r in result],[.04,.004,.5,.012])
    assert [r['passes'] for r in result]==[True,True,False,True]


def test_selector_choices_do_not_read_test_futures():
    from experiments.feature_search.coherence_analysis import choose_models
    rng=np.random.default_rng(7);x=rng.normal(size=(24,4));d=rng.normal(size=(24,2))
    loss=np.tile([1.,9.],(24,1));fit=np.arange(24)<12;domains=np.full(24,'a');hind=loss.copy()
    first=choose_models(x,d,loss,fit,domains,hind)
    loss[~fit]=[1000,-1000]
    second=choose_models(x,d,loss,fit,domains,hind)
    for name in first: np.testing.assert_array_equal(first[name],second[name])


def test_response_uses_intervention_baseline_and_composition_subtracts_zero():
    from experiments.feature_search.coherence_analysis import response_score,predictions
    expected=np.arange(32)/31;anchor=np.full(32,20.)
    assert response_score(anchor+expected,anchor,expected)['response_nmse']<1e-25
    f={'forecasts':np.stack([np.full(32,2.),np.full(32,3.),np.full(32,4.),np.full(32,5.),np.full(32,6.)])}
    p=predictions(dict(root=1,zero=0,methods={'fft':[1,2,3,4]}),f)
    np.testing.assert_array_equal(p['fft'],np.full(32,12.))


def test_classical_recovers_unknown_frequencies_with_large_distractor():
    from experiments.feature_search.coherence_classical import forecast
    t=np.arange(-128,32);y=.5*np.sin(2*np.pi*5.35*t/128+.7)+8*np.cos(2*np.pi*17.35*t/128-.3)
    pred=forecast(y[:128]);assert np.mean((pred-y[128:])**2)<1e-8


def test_response_distinguishes_gain_error_from_orthogonal_distortion():
    from experiments.feature_search.coherence_analysis import response_score
    expected=np.ones(32);orthogonal=3*np.tile([-1,1],16)
    r=response_score(2*expected+orthogonal,np.zeros(32),expected)
    assert r['gain_error']==1 and r['orthogonal_error']==9
    assert r['response_nmse']==r['gain_error']+r['orthogonal_error']==10
