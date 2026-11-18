"""Known-answer phase modes, conjugation and dangerous coarse-grid aliases."""
import numpy as np
import pytest
from experiments.feature_search import phase_orbit as study


def grid(n,origins=(.31,-.27)):
    return np.meshgrid(origins[0]+2*np.pi*np.arange(n)/n,
                       origins[1]+2*np.pi*np.arange(n)/n,indexing='ij')


def test_complex_fundamental_conjugation_and_equivariance_are_distinct():
    phi,psi=grid(8)
    good=study.mode_summary(np.exp(1j*phi),(.31,-.27),1.,'phi')
    bad=study.mode_summary(np.exp(-1j*phi),(.31,-.27),1.,'phi')
    assert good['wrong_low_order_energy']<1e-25
    np.testing.assert_allclose(good['correct_complex_gain'],[1,0],atol=1e-14)
    assert bad['wrong_low_order_energy']==pytest.approx(1)
    assert bad['correct_gain_error']==pytest.approx(1)
    assert study.mode_class(-1,1,'phi')['semantic']=='genuine_joint'
    assert study.mode_class(-1,1,'phi')['equivariance']=='compatible'
    assert study.mode_class(2,0,'psi')['semantic']=='wrong_axis_or_harmonic'
    assert study.mode_class(2,0,'psi')['equivariance']=='compatible'
    assert study.mode_class(-1,0,'phi')['equivariance']=='violating'


def test_alias_decoys_expose_false_fundamental_and_offgrid_failure():
    origins=(0.,0.)
    p8,s8=grid(8,origins);p16,s16=grid(16,origins)
    coarse,modes=study.orbit_fft(np.exp(9j*p8),origins)
    fine,fmodes=study.orbit_fft(np.exp(9j*p16),origins)
    assert abs(coarse[1,0])==pytest.approx(1)
    assert abs(fine[1,0])<1e-14
    phases=np.array([[.11,.37],[.63,-.71]])
    guessed=study.reconstruct(coarse,modes,phases)
    assert np.linalg.norm(guessed-np.exp(9j*phases[:,0]))>.5
    # sin7ψ appears entirely low-order on8, becomes high-order on16.
    c8,_=study.orbit_fft(np.sin(7*s8),origins)
    c16,_=study.orbit_fft(np.sin(7*s16),origins)
    assert abs(c8[0,1])>.49 and abs(c16[0,1])<1e-14


def test_non_tiling_full_gram_oracle_positive_complex_phase():
    _,future,rows=study.make_orbit((40,20),bundles=(0,))
    fit=study.temporal_components(future,(40,20))
    expected=np.asarray([[r['a']*np.exp(1j*r['phi']),r['b']*np.exp(1j*r['psi'])] for r in rows])
    np.testing.assert_allclose(fit['z'],expected,atol=1e-13)
    assert abs(fit['residual']).max()<1e-13
    assert not np.allclose(fit['gram'],np.diag(np.diag(fit['gram'])))
    summary=study.mode_summary(fit['z'][:,0].reshape(8,8),(rows[0]['origin_phi'],rows[0]['origin_psi']),rows[0]['a'],'phi')
    assert summary['correct_gain_error']<1e-25
    assert summary['parseval_error']<1e-13


def test_frozen_amplitudes_seeds_nested_grid_and_independent_offgrid():
    x,y,rows=study.make_orbit((32,16))
    assert x.shape==(512,128) and y.shape==(512,64)
    assert {r['generator_seed'] for r in rows}==set(range(3500000,3500008))
    assert rows[0]['offset']==np.random.default_rng(3500000).uniform(-2,2)
    for r in rows:
        assert r['a']/r['b']==pytest.approx(study.RATIOS[r['bundle']//2])
        assert (r['a']**2+r['b']**2)/2==pytest.approx(r['scale']**2)
    _,fine,frows=study.make_orbit((32,16),grid=16,bundles=(0,))
    np.testing.assert_array_equal(y[:64].reshape(8,8,64),fine.reshape(16,16,64)[::2,::2])
    _,_,off=study.make_orbit((32,16),offgrid=True,bundles=(0,))
    assert len(off)==16 and all(r['offgrid'] for r in off)
    assert off[0]['phi']!=rows[0]['phi']


def test_native_preprocessing_preserves_all_patch_states_and_c2_inverse():
    x,_,_=study.make_orbit((40,20),bundles=(0,))
    for model in ('timesfm','bolt','chronos2'):
        normalized,mu,sigma=study.normalized_input(x,model)
        assert normalized.shape==x.shape
        assert np.isfinite(normalized).all()
        if model=='timesfm':
            assert mu.shape==(64,4) and sigma.shape==(64,4)
            assert np.max(abs(mu[:,0]-mu[:,-1]))>.01
        else:
            affine=(x-mu)/sigma
            np.testing.assert_allclose(normalized,np.arcsinh(affine) if model=='chronos2' else affine,atol=2e-6)
    with pytest.raises(ValueError):study.orbit_fft(np.ones((8,7)),(0,0))


def test_stable_low_modes_do_not_imply_complete_response_explanation():
    origins=(.31,-.27);p8,_=grid(8);p16,_=grid(16)
    phases=np.random.default_rng(1).uniform(-np.pi,np.pi,(16,2))
    def response(phi):return np.exp(1j*phi)+.3*np.exp(4j*phi)
    value=study.alias_stability(response(p8),response(p16),origins,response(phases[:,0]),phases,1.)
    assert value['stable_low_order']
    assert not value['adequate_complex_coverage']
    assert value['beyond_order3_energy']==pytest.approx(.09)
    alias=study.alias_stability(np.exp(9j*p8),np.exp(9j*p16),origins,np.exp(9j*phases[:,0]),phases,1.)
    assert not alias['stable_low_order']


def test_interestingness_requires_same_bundles_and_fixed_six_of_eight():
    first=np.array([.1]*6+[0.,0.]);second=np.zeros(8)
    assert study.pilot_interestingness(first,second)['promotes_alias_validation']
    assert not study.pilot_interestingness(np.array([.1]*5+[0.]*3),second)['promotes_alias_validation']
    assert not study.pilot_interestingness(np.full(8,.1),np.full(8,.09))['promotes_alias_validation']
    with pytest.raises(ValueError):study.pilot_interestingness(first[:6],second[:6])


def test_omitted_temporal_harmonic_can_mimic_incompatible_target_mode():
    phi,psi=grid(8)
    t=np.arange(64)
    waveform=np.cos(3*(2*np.pi*t[None,None,:]/40+phi[:,:,None]))
    fitted=study.temporal_components(waveform.reshape(64,64),(40,20))
    z,modes=study.orbit_fft(fitted['z'][:,0].reshape(8,8),(.31,-.27))
    assert abs(z[3,0])>.01  # Finite-window leakage despite joint full Gram.
    assert study.mode_class(3,0,'phi')['equivariance']=='violating'
    full,_=study.orbit_fft(waveform,(.31,-.27))
    assert study.waveform_mode_compatibility(full[3,0],3,0,40)<1e-25
    # Wrong temporal frequency is a confusable decoy for the waveform check.
    assert study.waveform_mode_compatibility(np.exp(2j*np.pi*t/40),3,0,40)>.5


def test_primary_full_waveform_parseval_dc_and_conjugation_decoys():
    _,future,rows=study.make_orbit((40,20),bundles=(0,))
    truth=future.reshape(8,8,64);r=rows[0];origins=(r['origin_phi'],r['origin_psi'])
    oracle=study.waveform_budget(truth,truth,origins,r['scale'],40)
    assert oracle['scaled_MSE']==0 and sum(oracle['error_categories'].values())==0
    assert oracle['primary_D']<1e-25  # Nonzero true offset must NOT count.
    phase,modes=study.orbit_fft(truth,origins)
    t=np.arange(64)
    np.testing.assert_allclose(phase[1,0],r['a']/(2j)*np.exp(2j*np.pi*t/40),atol=1e-14)
    np.testing.assert_allclose(phase[-1,0],phase[1,0].conj(),atol=1e-14)
    phi,psi=grid(8,origins)
    conjugate=r['offset']+r['a']*np.sin(2*np.pi*t/40-phi[:,:,None])+r['b']*np.sin(2*np.pi*t/20+psi[:,:,None])
    bad=study.waveform_budget(conjugate,truth,origins,r['scale'],40)
    assert bad['low_order_nonexpected_phase_energy']<1e-25
    assert bad['expected_axis_time_character_residual_energy']>.1
    assert bad['primary_D']>.1
    assert bad['error_Parseval_absolute_error']<1e-13
    biased=study.waveform_budget(truth+3,truth,origins,r['scale'],40)
    assert biased['error_categories']['phase_DC']==pytest.approx(9/r['scale']**2)
    assert biased['primary_D']<1e-25
    gain=study.waveform_budget(2*(truth-r['offset'])+r['offset'],truth,origins,r['scale'],40)
    assert gain['primary_D']<1e-25
    assert gain['expected_axis_gain_rotation_error']>.5
    for value in (oracle,bad,biased,gain):
        assert value['primary_D']<=value['scaled_MSE']+1e-12
        assert value['expected_axis_gain_rotation_error']>=-1e-12
    tiny=next(v for v in oracle['modes'] if (v['k'],v['l'])==(2,1))
    assert tiny['physical_time_character_residual_fraction'] is None
    assert tiny['physical_time_character_residual_energy']>=0


def test_full_waveform_modes_keep_omitted_harmonic_and_joint_error_disjoint():
    phi,psi=grid(8);t=np.arange(64);origins=(.31,-.27)
    extra=np.cos(3*(2*np.pi*t/40+phi[:,:,None]))
    joint=.7*np.cos(2*np.pi*t/40+phi[:,:,None]+psi[:,:,None])
    prediction=extra+joint
    result=study.waveform_budget(prediction,np.zeros_like(prediction),origins,1.,40)
    assert result['error_categories']['higher_axis']==pytest.approx(.5)
    assert result['error_categories']['joint']==pytest.approx(.7**2/2)
    assert result['primary_D']==pytest.approx(.5+.7**2/2)
    assert result['error_Parseval_absolute_error']<1e-13
    assert result['primary_D']<=result['scaled_MSE']+1e-12
    assert result['expected_axis_gain_rotation_error']>=-1e-12
    mode=next(v for v in result['modes'] if (v['k'],v['l'])==(3,0))
    assert mode['physical_time_character_residual_fraction']<1e-25
    # Persistence is stationary as a rule but fails this waveform character.
    assert study.waveform_mode_compatibility(np.ones(64),1,0,40)>.5


def test_primary_waveform_stability_preserves_time_axis_and_coarse_offgrid():
    p8,s8=grid(8);p16,s16=grid(16);t=np.arange(64);origins=(.31,-.27)
    phases=np.random.default_rng(2).uniform(-np.pi,np.pi,(16,2))
    def wave(phi):return np.sin(2*np.pi*t/40+phi[...,None])
    result=study.waveform_stability(wave(p8),wave(p16),origins,wave(phases[:,0]),phases,1.)
    assert result['stable_low_order'] and result['adequate_waveform_coverage']
    assert result['offgrid_coarse_waveform_RMS']<1e-13
    alias=study.waveform_stability(wave(9*p8),wave(9*p16),origins,wave(9*phases[:,0]),phases,1.)
    assert not alias['stable_low_order']


def test_native_analysis_oracle_retains_full_future_and_normalizer_profiles():
    from experiments.feature_search import phase_orbit_native as native
    context,future,rows=study.make_orbit((40,20))
    records,normalization=native.analyze_case(future,context,future,rows,(40,20),'bolt')
    assert len(records)==8 and normalization[0].shape==(512,128)
    assert all(r['primary_waveform_budget']['primary_D']<1e-20 for r in records)
    assert all(r['primary_waveform_budget']['error_Parseval_absolute_error']==0 for r in records)
    assert all(len(r['native_sigma_phase_profile']['energy'])==8 for r in records)
    with pytest.raises(ValueError,match='512×64'):
        native.analyze_case(future[:,:32],context,future,rows,(40,20),'bolt')


def test_saved_prediction_refuses_swapped_corpus_and_phase_order(tmp_path):
    import json
    from experiments.feature_search import phase_orbit_native as native
    source=tmp_path/'source';source.mkdir();stem='p32_16'
    x,y,rows=study.make_orbit((32,16))
    np.savez_compressed(source/f'{stem}.npz',context=x,clean_future=y)
    (source/f'{stem}.json').write_text(json.dumps(rows),encoding='utf-8')
    (source/'protocol.json').write_text('{}',encoding='utf-8')
    payload=dict(prediction=y,model=np.asarray('bolt'),case=np.asarray(stem),
        phase_row_ids=native.row_ids(rows,stem),corpus_sha256=np.asarray(native.digest(source/f'{stem}.npz')),
        metadata_sha256=np.asarray(native.digest(source/f'{stem}.json')),
        core_protocol_sha256=np.asarray(native.digest(source/'protocol.json')))
    path=tmp_path/'forecast.npz';np.savez_compressed(path,**payload)
    np.testing.assert_array_equal(native.load_prediction(path,'bolt',stem,source)['prediction'],y)
    payload['phase_row_ids']=payload['phase_row_ids'][::-1];np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match='phase-row'):native.load_prediction(path,'bolt',stem,source)
    payload['phase_row_ids']=payload['phase_row_ids'][::-1]
    payload['corpus_sha256']=np.asarray('swapped-corpus');np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match='binding'):native.load_prediction(path,'bolt',stem,source)
