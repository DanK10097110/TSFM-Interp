"""Known full-Gram and cross-frequency decoys for the prospective Stage49 grid."""
import numpy as np
import pytest
from experiments.feature_search.frequency_generalization import (
    protocol, make_case, basis, metrics, analyze, ENDPOINTS)


def test_full_gram_recovers_nontiling_target_without_false_off_frequency():
    """Nonorthogonal harmonics must be fitted jointly, never by raw projections."""
    config = protocol()
    config['n_seeds'] = 3
    for periods in config['pairs']:
        data, metadata = make_case(config, periods, 128)
        value, diagnostic = metrics(data['clean_future'],data['clean_future'],metadata,periods,config['delta'])
        assert diagnostic['basis_condition'] < 1.72
        assert diagnostic['groundtruth_fit_residual'] < 1e-12
        assert value['off_target_energy'].max() < 1e-25
        assert value['target_derivative_error'].max() < 1e-25
        assert np.abs(value['derivative_energy_closure_error']).max() < 1e-12
        assert value['normalized_mse'].max() == 0
    with pytest.raises(ValueError,match='distinct'):
        basis((16,16),np.arange(64))


def test_planted_leak_energy_uses_waveform_not_orthogonal_coefficient_shortcut():
    """A non-tiling leaked sine has energy differing from coefficient-norm/2."""
    config = protocol()
    config['n_seeds'] = 3
    periods = (48,24)
    data,metadata = make_case(config,periods,128)
    prediction = data['clean_future'].copy()
    wave = np.sin(2*np.pi*np.arange(64)/48+.7)
    for index,row in enumerate(metadata):
        prediction[index] += row['phase_shift']*.4*wave
    value,_ = metrics(prediction,data['clean_future'],metadata,periods,config['delta'])
    expected = np.array([np.mean((.4*wave)**2)/(row['b']**2/2) for row in metadata[::3]])
    np.testing.assert_allclose(value['off_target_energy'],expected,rtol=1e-12,atol=1e-12)
    assert not np.allclose(np.mean(wave**2),.5,atol=.001)
    assert value['target_derivative_error'].max() < 1e-25
    assert np.abs(value['derivative_energy_closure_error']).max() < 1e-12
    bad = [dict(row) for row in metadata]
    bad[1]['psi'] += .1
    with pytest.raises(ValueError,match='latent'):
        metrics(prediction,data['clean_future'],bad,periods,config['delta'])


def test_rightaligned_extent_complete_family_row_identity_and_closed_negatives():
    """Identical future and nested context do not license shuffled prediction IDs."""
    config = protocol()
    config.update(n_seeds=3,n_boot=9)
    cases,predictions = {},{}
    for periods in config['pairs']:
        short,meta = make_case(config,periods,128)
        long,long_meta = make_case(config,periods,512)
        assert meta == long_meta
        np.testing.assert_array_equal(short['clean_future'],long['clean_future'])
        np.testing.assert_array_equal(short['context'],long['context'][:,-128:])
        for context,data in [(128,short),(512,long)]:
            name = f'{periods}-{context}'
            cases[name] = (data,meta,periods)
            predictions[name] = {model['name']:dict(prediction=data['clean_future'],row_ids=data['row_ids']) for model in config['models']}
    result,_ = analyze(config,cases,predictions)
    assert result['primary_family_size'] == 108
    assert len(result['claims']) == 108
    assert all(effect['p_holm'] == 1 and not effect['passed'] for effect in result['claims'].values())
    assert not result['candidates_for_fresh_replication']
    incomplete = dict(cases)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError,match='family changed'):
        analyze(config,incomplete,predictions)
    first = next(iter(predictions))
    predictions[first]['bolt']['row_ids'] = predictions[first]['bolt']['row_ids'][::-1]
    with pytest.raises(ValueError,match='identity'):
        analyze(config,cases,predictions)


def test_promotion_requires_nonlocal_frequency_pair_not_only_neighbor_contexts():
    """Repeated contexts and31/33 neighbors cannot establish broad generalization."""
    from experiments.feature_search.frequency_generalization import candidate_patterns
    def entry(pair):
        return dict(model_pair='timesfm-minus-bolt',endpoint='off_target_energy',periods=pair,passed=True)
    claims={'a':entry((31,16)),'b':entry((33,16)),'c':entry((31,16))}
    assert candidate_patterns(claims) == []
    claims['d']=entry((40,20))
    selected=candidate_patterns(claims)
    assert len(selected)==1
    assert len(selected[0]['period_pairs'])==3


def test_global108_not_casewise_sign_correction():
    """Nine unanimous seeds are casewise persuasive but insufficient globally."""
    config=protocol()
    config.update(n_seeds=9,n_boot=9)
    cases,predictions={},{}
    for periods in config['pairs']:
        for context in config['contexts']:
            data,metadata=make_case(config,periods,context)
            name=f'{periods}-{context}'
            cases[name]=(data,metadata,periods)
            predictions[name]={model['name']:dict(prediction=data['clean_future']+bias,row_ids=data['row_ids'])
                for model,bias in zip(config['models'],[0.,.5,1.])}
    result,_=analyze(config,cases,predictions)
    losses=[v for v in result['claims'].values() if v['endpoint']=='normalized_mse']
    assert all(v['p'] < .05 and v['p_holm'] >= .05 and not v['passed'] for v in losses)


def test_prepared_input_rejects_hash_consistent_wrong_context(tmp_path):
    """A manually rebound array cannot substitute for the frozen generator."""
    import json
    from experiments.feature_search.frequency_generalization import prepare,load_prepared,digest
    out=tmp_path/'prepared'
    prepare(out)
    payload,config,cases=load_prepared(out)
    assert len(cases)==12
    name=next(iter(payload['sources']))
    path=out/f'{name}.npz'
    with np.load(path) as source:
        data={key:source[key] for key in source.files}
    data['context']=data['context'].copy()
    data['context'][0,0]+=.5
    np.savez_compressed(path,**data)
    payload['sources'][name]['corpus_sha256']=digest(path)
    (out/'protocol.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='physical generator'):
        load_prepared(out)
