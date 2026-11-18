"""Known-answer crossover replication guards, without native model loading."""
import json
import numpy as np
import pytest
from experiments.feature_search import frequency_crossover_replication as study


def fixture(n=3):
    """Use non-tiling40/20 to expose projection and pure-error shortcuts."""
    config=study.protocol()
    config.update(n_seeds=n,n_boot=19)
    data,metadata=study.make_case(config,(40,20),128)
    return config,data,metadata


def test_actual_background_preserves_affine_bias_and_oracle_zero():
    """Using the true offset instead of f(background) silently doubles bias."""
    config,data,metadata=fixture()
    truth=data['clean_future']
    oracle,diagnostic=study.component_metrics(truth,data,metadata,(40,20),config)
    assert oracle['L'].max()==0
    assert np.abs(oracle['G']).max()<1e-25
    assert oracle['off_target_energy'].max()<1e-25
    assert diagnostic['basis_condition']<1.72
    biased,_=study.component_metrics(truth+.3,data,metadata,(40,20),config)
    np.testing.assert_allclose(biased['S'],biased['L'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(biased['G'],0,atol=1e-14)
    ratio=study.residual_fraction(biased['L'],biased['S'],config)
    assert ratio['value']==pytest.approx(1.)
    assert study.residual_fraction(oracle['L'],oracle['S'],config)['value'] is None


def test_separable_frequency_bias_has_nonzero_legacy_interaction_at_zero_gain():
    """Pureφ distortion reproduces mixed failure with exact additive forecasts."""
    config,data,metadata=fixture()
    prediction=data['clean_future']+.5*data['component_future'][:,0]
    value,_=study.component_metrics(prediction,data,metadata,(40,20),config)
    assert value['L'].min()>.01
    np.testing.assert_allclose(value['J'],.5*value['L'],atol=1e-14)
    np.testing.assert_allclose(value['G'],0,atol=1e-14)
    assert value['component_sum_direct_discrepancy'].max()<1e-25
    assert value['pure_psi'].max()==0
    biased,_=study.component_metrics(prediction+.3,data,metadata,(40,20),config)
    assert np.abs(biased['J']).max()>.01
    np.testing.assert_allclose(biased['G'],0,atol=1e-14)


def test_ratio_bootstrap_preserves_seed_pairing_not_mean_individual_ratios():
    """Aggregate losses and paired resampling have a planted exact answer."""
    config,_,_=fixture()
    value=study.residual_fraction(np.array([1.,3.]),np.array([.5,1.5]),config)
    assert value['value']==value['lo']==value['hi']==.5
    unbalanced=study.residual_fraction(np.array([1.,9.]),np.array([1.,0.]),config)
    assert unbalanced['value']==.1
    assert unbalanced['value']!=np.mean([1.,0.])
    with pytest.raises(ValueError,match='vectors'):
        study.residual_fraction(np.ones((2,2)),np.ones((2,2)),config)


def test_exact_effect_mapping_and_ranking_flip_requires_both_signed_gaps():
    """A negative difference of two positive gaps is not a ranking flip."""
    def row(loss,gain,off):return {key:np.asarray([value]) for key,value in [('L',loss),('G',gain),('off_target_energy',off)]}
    values={32:dict(timesfm=row(4,3,4),bolt=row(2,1,1)),40:dict(timesfm=row(1,1,0),bolt=row(5,4,3))}
    mapped=study.effects(values)
    np.testing.assert_equal([mapped[key][0] for key in study.EFFECTS],[2,-4,-6,2,-3,-5,3,-3,-6,3,4])
    claims={f'c128/{key}':dict(passed=True,mean=dict(value=value)) for key,value in [('g32',4),('g40',1),('g40_minus_g32',-3)]}
    assert not study.ranking_flip(claims,128)
    claims['c128/g40']['mean']['value']=-1
    assert study.ranking_flip(claims,128)
    claims['c128/g32']['passed']=False
    assert not study.ranking_flip(claims,128)


def test_whole22_family_identity_and_primary_no_sensitivity_rescue():
    """All cases stay present, including nulls, and row permutations are refused."""
    config,_,_=fixture()
    cases,predictions={},{}
    for periods in config['pairs']:
        for context in config['contexts']:
            name=f'p{periods[0]}_{periods[1]}_c{context}'
            data,metadata=study.make_case(config,periods,context)
            cases[name]=(data,metadata)
            predictions[name]={m['name']:dict(prediction=data['clean_future'],row_ids=data['row_ids']) for m in config['models']}
    result,_=study.analyze(config,cases,predictions)
    assert len(result['claims'])==22
    assert all(not value['passed'] for value in result['claims'].values())
    assert all(value['scope'].startswith('sensitivity') for key,value in result['claims'].items() if key.startswith('c512'))
    assert not result['primary_ranking_flip']
    altered=dict(config,contexts=[128])
    with pytest.raises(ValueError,match='family changed'):
        study.analyze(altered,cases,predictions)
    name=next(iter(predictions))
    predictions[name]['bolt']['row_ids']=predictions[name]['bolt']['row_ids'][::-1]
    with pytest.raises(ValueError,match='identity'):
        study.analyze(config,cases,predictions)


def test_frozen_background_and_component_future_cannot_be_rebound(tmp_path):
    """Matching IDs and newly computed hashes cannot hide changed true components."""
    out=tmp_path/'prepared'
    payload=study.prepare(out)
    _,_,cases=study.load_prepared(out)
    assert len(cases)==4
    name=next(iter(payload['sources']))
    path=out/f'{name}.npz'
    with np.load(path) as source:data={key:source[key] for key in source.files}
    data['component_future']=data['component_future'].copy()
    data['component_future'][5,0,0]=.5
    np.savez_compressed(path,**data)
    payload['sources'][name]['corpus_sha256']=study.screen.digest(path)
    (out/'protocol.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='physical input/background'):
        study.load_prepared(out)


def test_positive_native_gain_and_off_target_only_coupling():
    """Native mixture error and a leaked first harmonic have planted signs/units."""
    config,data,metadata=fixture()
    prediction=data['clean_future'].copy()
    t=np.arange(config['horizon'])
    wave=np.sin(2*np.pi*t/40+.7)
    for index,row in enumerate(metadata):
        if row['condition'] in ('baseline','minus','plus'):
            prediction[index]+=.5+row['phase_shift']*.4*wave
    value,_=study.component_metrics(prediction,data,metadata,(40,20),config)
    np.testing.assert_allclose(value['G'],value['L'],atol=1e-14)
    assert value['G'].min()>0
    expected=[np.mean((.4*wave)**2)/(row['b']**2/2) for row in metadata[::6]]
    np.testing.assert_allclose(value['off_target_energy'],expected,atol=1e-13,rtol=1e-12)
    assert value['target_derivative_error'].max()<1e-25
    assert np.abs(value['derivative_energy_closure_error']).max()<1e-12


@pytest.mark.parametrize('n,expected',[(16,True),(8,False)])
def test_negative_passing_gain_is_component_sum_harm_not_positive_gain(n,expected):
    """Two-sided detection must never become a direction-free improvement label."""
    config=study.protocol()
    config.update(n_seeds=n,n_boot=9)
    cases,predictions={},{}
    for periods in config['pairs']:
        for context in config['contexts']:
            name=f'p{periods[0]}_{periods[1]}_c{context}'
            data,metadata=study.make_case(config,periods,context)
            prediction=data['clean_future'].copy()
            for index,row in enumerate(metadata):
                if row['condition'] in ('pure_phi','pure_psi'):
                    prediction[index]+=.5
            cases[name]=(data,metadata)
            predictions[name]={model['name']:dict(prediction=prediction,row_ids=data['row_ids']) for model in config['models']}
    result,_=study.analyze(config,cases,predictions)
    assert result['claims']['c128/G_timesfm32']['passed'] is expected
    assert result['claims']['c128/G_timesfm32']['mean']['value']<0
    assert not any(result['positive_gain_evidence'].values())
