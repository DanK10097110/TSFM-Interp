"""Known-answer feature interference with confusable bias, aliases and row order."""
import json

import numpy as np
import pytest

from tsfm_lens.analysis.feature_interference import (InterferenceDesign,audit_interference,
    error_decomposition,make_factorial,sign_sensitivity)


def fixture(n=6):
    """Return complete paired truth with offsets that expose composition bias."""
    design=InterferenceDesign(n_bundles=n,seed_base=2001000,offset=(1.,1.))
    corpus,metadata=make_factorial(design)
    truth=corpus['component_future'].sum(1)+np.array([m['offset'] for m in metadata])[:,None]
    return design,corpus,metadata,truth


def audit(design,corpus,metadata,forecasts):
    """Use small resampling counts; these tests check planted answers, not power."""
    return audit_interference(corpus['context'],corpus['component_future'],forecasts,metadata,
                              corpus['row_ids'],design,n_boot=19,n_permutations=19)


def test_independent_oracle_and_constant_bias_sum_counterexample():
    """A pure-component forecast sum doubles constant bias rather than fixing it."""
    design,corpus,metadata,truth=fixture()
    forecasts={'perfect':dict(prediction=truth,row_ids=corpus['row_ids']),
               'biased':dict(prediction=truth+.3,row_ids=corpus['row_ids'])}
    result,_=audit(design,corpus,metadata,forecasts)
    for target in ('phi','psi'):
        assert result['models']['perfect']['changed_target'][target]['untouched_coefficient_change_normalized']['value']<1e-25
    native=result['models']['biased']['scaled_mse']['mixed']['value']
    assert result['models']['biased']['scaled_mse']['pure_phi']['value']==pytest.approx(native)
    assert result['models']['biased']['scaled_mse']['pure_psi']['value']==pytest.approx(native)
    composed=result['models']['biased']['oracle_component_sum']['mse_common_mixvariance']['value']
    assert composed==pytest.approx(4*native)


def test_planted_one_direction_coupling_does_not_invent_reverse_effect():
    """Changing φ rotates ψ, while changing ψ preserves the φ component."""
    design,corpus,metadata,truth=fixture()
    prediction=truth.copy();t=np.arange(design.context,design.context+design.horizon)
    for i,m in enumerate(metadata):
        if m['condition']=='mixed':
            new=m['b']*np.sin(2*np.pi*t/design.periods[1]+m['psi']+.5*np.sin(m['phi']))
            prediction[i]+=new-corpus['component_future'][i,1]
    result,_=audit(design,corpus,metadata,{'coupled':dict(prediction=prediction,row_ids=corpus['row_ids'])})
    targets=result['models']['coupled']['changed_target']
    assert targets['phi']['untouched_phase_change_radians']['value']>.1
    assert targets['psi']['untouched_phase_change_radians']['value']<1e-12


def test_row_permutation_and_missing_ids_are_refused():
    """The same-shaped prediction permutation is a deliberate identity decoy."""
    design,corpus,metadata,truth=fixture()
    with pytest.raises(ValueError,match='row IDs'):
        audit(design,corpus,metadata,{'wrong':dict(prediction=truth[::-1],row_ids=corpus['row_ids'][::-1])})
    with pytest.raises(ValueError,match='row IDs'):
        audit(design,corpus,metadata,{'missing':dict(prediction=truth)})


def test_common_scale_and_undefined_phase_keep_complete_family():
    """A dead predicted component is untestable, never perfectly invariant phase."""
    design,corpus,metadata,truth=fixture()
    result,_=audit(design,corpus,metadata,{'good':dict(prediction=truth,row_ids=corpus['row_ids']),
        'dead':dict(prediction=np.zeros_like(truth),row_ids=corpus['row_ids'])})
    assert result['family_size']==4
    for key,value in result['primary'].items():
        if 'untouched_phase' in key:
            assert value['p']==1 and value['n_seeds']==0 and 'untestable' in value
    json.dumps(result,allow_nan=False)
    broken=[dict(m) for m in metadata]
    for m in broken:
        if m['condition']=='pure_phi':m['scale']=m['a']/np.sqrt(2)
    with pytest.raises(ValueError,match='SAME mixed-energy'):
        audit(design,corpus,broken,{'good':dict(prediction=truth,row_ids=corpus['row_ids'])})


def test_orthogonal_decomposition_retains_offset_and_private_residual():
    """An undeclared Nyquist residual and DC error must not disappear in fitting."""
    design,corpus,metadata,truth=fixture()
    prediction=truth+.3+.2*np.tile([-1,1],16)
    channels=error_decomposition(prediction,truth,design)
    np.testing.assert_allclose(channels['offset'],.09,atol=1e-12)
    np.testing.assert_allclose(channels['residual'],.04,atol=1e-12)
    total=channels['offset']+channels['residual']+channels['phi_coefficient']+channels['psi_coefficient']
    np.testing.assert_allclose(total,((prediction-truth)**2).mean(1),atol=1e-12)


def test_alias_guard_tiling_and_sign_ties():
    """Identical/aliased frequencies and non-tiled horizons are not a factorial."""
    with pytest.raises(ValueError,match='below-Nyquist'):InterferenceDesign(periods=(32,2))
    with pytest.raises(ValueError,match='distinct'):InterferenceDesign(periods=(16,16))
    with pytest.raises(ValueError,match='tile context AND horizon'):InterferenceDesign(periods=(24,16))
    sensitivity=sign_sensitivity(np.array([1,2,-1,0,np.nan]))
    assert sensitivity['positive']==2 and sensitivity['negative']==1 and sensitivity['ties']==1
    assert sensitivity['fraction_positive_all']==.5
    assert sensitivity['fraction_positive_nonzero']==pytest.approx(2/3)


def test_configurable_periods_horizon_and_three_model_order():
    """A nonlegacy48-step horizon and nonsorted names preserve the full family."""
    design=InterferenceDesign(context=144,horizon=48,periods=(24,12),n_bundles=3)
    corpus,metadata=make_factorial(design)
    assert corpus['component_future'].shape==(24,2,48)
    truth=corpus['component_future'].sum(1)+np.array([m['offset'] for m in metadata])[:,None]
    predictions={name:dict(prediction=truth+bias,row_ids=corpus['row_ids']) for name,bias in [('b',.1),('a',.2),('c',.3)]}
    result,_=audit(design,corpus,metadata,predictions)
    assert result['family_size']==12
    assert result['primary']['a-minus-b/mixed_scaled_mse']['value']>0
    flipped,_=audit_interference(corpus['context'],corpus['component_future'],predictions,metadata,corpus['row_ids'],
                                 design,n_boot=19,n_permutations=19,model_order=['a','b','c'])
    assert flipped['primary']['b-minus-a/mixed_scaled_mse']['value']<0


def test_complete_family_inference_resolution_is_visible():
    """Twelve claims with nine permutations cannot reject even at the p floor."""
    design,corpus,metadata,truth=fixture()
    forecasts={name:dict(prediction=truth+bias,row_ids=corpus['row_ids']) for name,bias in [('a',.1),('b',.2),('c',.3)]}
    result,_=audit_interference(corpus['context'],corpus['component_future'],forecasts,metadata,corpus['row_ids'],
                               design,n_boot=19,n_permutations=9)
    resolution=result['inference_resolution']
    assert resolution['family_size']==12
    assert resolution['plus_one_p_floor']==.1
    assert resolution['smallest_possible_holm_p']==1.
    assert not resolution['can_reject_at_0_05']
    assert all(value['p_holm']==1. for value in result['primary'].values())
    assert any('cannot reject' in warning for warning in result['warnings'])


def test_prepared_json_uses_same_strict_configuration_validator():
    """A plausible hand-made model configuration cannot bypass artifact naming."""
    from run_feature_interference import validate_config
    malformed=dict(models=[dict(name='../wrong',adapter='mock_patch',checkpoint='')])
    with pytest.raises(ValueError,match='artifact-safe'):
        validate_config(malformed)
    malformed['models'][0]['name']='valid'
    malformed['inference']={'n_permutations':0}
    with pytest.raises(ValueError,match='positive integer resampling'):
        validate_config(malformed)


def test_forecast_artifact_binds_context_not_only_reused_factorial_ids(tmp_path):
    """Equal IDs/N/H are a decoy when physical input contexts differ."""
    from run_feature_interference import file_sha256, load_forecast_artifact
    short=InterferenceDesign(n_bundles=3,context=128,seed_base=2012000)
    long=InterferenceDesign(n_bundles=3,context=256,seed_base=2012000)
    a,meta=make_factorial(short);b,_=make_factorial(long)
    np.testing.assert_equal(a['row_ids'],b['row_ids'])
    assert a['component_future'].shape==b['component_future'].shape
    first=tmp_path/'short.npz';second=tmp_path/'long.npz'
    np.savez_compressed(first,**a);np.savez_compressed(second,**b)
    truth=a['component_future'].sum(1)+np.array([m['offset'] for m in meta])[:,None]
    forecast=tmp_path/'prediction.npz'
    np.savez_compressed(forecast,prediction=truth,row_ids=a['row_ids'],corpus_sha256=np.asarray(file_sha256(first)))
    np.testing.assert_equal(load_forecast_artifact(forecast,file_sha256(first))['prediction'],truth)
    with pytest.raises(ValueError,match='corpus hash mismatch'):
        load_forecast_artifact(forecast,file_sha256(second))
    np.savez_compressed(forecast,prediction=truth,row_ids=a['row_ids'])
    with pytest.raises(ValueError,match='corpus_sha256'):
        load_forecast_artifact(forecast,file_sha256(first))


def test_native_background_subtraction_preserves_affine_bias_and_legacy_inputs():
    """Actual background subtraction avoids doubling gain/offset forecast bias."""
    from dataclasses import replace
    design=InterferenceDesign(n_bundles=6,seed_base=2024000,offset=(1.,1.),native_background=True)
    corpus,metadata=make_factorial(design)
    assert len(metadata)==54
    background=np.array([m['condition']=='background' for m in metadata])
    assert background.sum()==6
    np.testing.assert_equal(corpus['component_future'][background],0)
    truth=corpus['component_future'].sum(1)+np.array([m['offset'] for m in metadata])[:,None]
    prediction=1.2*truth+.3
    result,_=audit(design,corpus,metadata,{'affine':dict(prediction=prediction,row_ids=corpus['row_ids'])})
    measured=result['models']['affine']
    assert result['component_sum_mode']=='native_background_forecast'
    assert measured['component_sum']['subtraction_mode']=='native_background_forecast'
    assert measured['component_sum']['mse_common_mixvariance']['value']==pytest.approx(measured['scaled_mse']['mixed']['value'])
    assert measured['component_sum']['composition_defect_common_mixvariance']['value']<1e-25
    assert measured['native_background_forecast']['mean_bias']['value']==pytest.approx(.5)
    assert measured['native_background_forecast']['raw_mse']['value']==pytest.approx(.25)
    assert 'oracle_component_sum' not in measured
    legacy,legacy_metadata=make_factorial(replace(design,native_background=False))
    position={row_id:i for i,row_id in enumerate(corpus['row_ids'])}
    ordered=[position[row_id] for row_id in legacy['row_ids']]
    np.testing.assert_equal(legacy['context'],corpus['context'][ordered])
    np.testing.assert_equal(legacy['component_future'],corpus['component_future'][ordered])
    assert len(legacy_metadata)==48
    broken=dict(corpus);broken['component_future']=corpus['component_future'].copy()
    broken['component_future'][np.flatnonzero(background)[0],0,0]=1e-12
    with pytest.raises(ValueError,match='background component futures must be exactly zero'):
        audit(design,broken,metadata,{'affine':dict(prediction=prediction,row_ids=corpus['row_ids'])})
    keep=~background
    incomplete={key:values[keep] for key,values in corpus.items()}
    with pytest.raises(ValueError,match='complete 9-cell factorial'):
        audit(design,incomplete,[m for m in metadata if m['condition']!='background'],
              {'affine':dict(prediction=prediction[keep],row_ids=corpus['row_ids'][keep])})


def test_seed_helpers_reject_repeated_row_matrices_and_infinities():
    """Repeated window rows cannot become extra seeds, and missingness is explicit."""
    from tsfm_lens.analysis.feature_interference import paired_seed_effect,_interval,sign_comparison
    settings=dict(n_boot=19,n_permutations=19,seed=113,ci=.95)
    functions=[lambda values:paired_seed_effect(values,**settings),sign_sensitivity,sign_comparison,
               lambda values:_interval(values,settings)]
    for function in functions:
        with pytest.raises(ValueError,match='one-dimensional'):
            function(np.tile([1.,-1.],(4,1)))
        with pytest.raises(ValueError,match='infinite seed metrics'):
            function(np.array([1.,2.,np.inf,np.nan]))
    first=np.array([.1,.2,.3,.9,np.nan]);second=np.array([np.nan,.1,.2,.8,.9])
    assert np.nanmean(second)-np.nanmean(first)==pytest.approx(.125)
    paired=paired_seed_effect(second-first,**settings)
    assert paired['value']==pytest.approx(-.1)
    assert paired['n_seeds']==3 and paired['excluded_seeds']==2


def test_exact_sign_probability_tails_intervals_ties_and_reversal():
    """Known binomial tails distinguish a two-sided test from the old sensitivity."""
    from tsfm_lens.analysis.feature_interference import sign_comparison
    positive=sign_comparison(np.ones(8))
    negative=sign_comparison(-np.ones(8))
    assert positive['p']==pytest.approx(2/256)
    assert negative['p']==positive['p']
    assert positive['lo']==pytest.approx(.025**(1/8))
    assert positive['hi']==1.
    assert negative['value']==0. and negative['lo']==0.
    assert negative['hi']==pytest.approx(1-positive['lo'])
    partial=sign_comparison(np.r_[np.ones(13),-np.ones(3)])
    assert partial['p']==pytest.approx(1394/65536)
    undefined=sign_comparison(np.array([0.,0.,np.nan]))
    assert undefined['value'] is None and undefined['lo'] is None and undefined['p']==1.
    assert undefined['ties']==2 and undefined['excluded_seeds']==1 and undefined['n_nonzero']==0
    with_ties=sign_comparison(np.array([1.,1.,-1.,0.,np.nan]))
    assert with_ties['value']==pytest.approx(2/3)
    assert with_ties['n_nonzero']==3 and with_ties['n_seeds']==4
    assert with_ties['ties']==1 and with_ties['excluded_seeds']==1


def test_only_predeclared_endpoint_controls_status_and_flags_mean_majority_disagreement():
    """Mean significance cannot rescue q=.5, nor can strong signs rescue the mean."""
    from tsfm_lens.analysis.feature_interference import paired_seed_effect,sign_comparison,_select_primary_family
    from tsfm_lens.analysis.stats import holm
    def families(values):
        means={str(i):paired_seed_effect(values,n_boot=199,n_permutations=199) for i in range(4)}
        signs={str(i):sign_comparison(values) for i in range(4)}
        for family in (means,signs):
            adjusted=holm({key:value['p'] for key,value in family.items()})
            for key in family:family[key]['p_holm']=adjusted[key]
        return means,signs
    means,signs=families(np.r_[np.full(32,100.),np.full(32,-1.)])
    assert means['0']['p_holm']<.05 and signs['0']['p_holm']==1.
    selected=_select_primary_family(means,signs,'sign_probability')
    assert selected['0']['statistical_status']=='not_detected'
    means,signs=families(np.r_[np.ones(63),-1000.])
    assert means['0']['p_holm']==1. and signs['0']['p_holm']<.05
    assert signs['0']['descriptive_signed_mean']<0 and signs['0']['descriptive_signed_median']==1.
    assert signs['0']['mean_vs_majority_disagreement']
    selected=_select_primary_family(means,signs,'mean')
    assert selected['0']['statistical_status']=='not_detected'


def test_selected_sign_resolution_ignores_permutation_budget_and_enum_is_strict():
    """Eight nonzero signs can be testable even when a nine-draw mean family is not."""
    design,corpus,metadata,truth=fixture(n=8)
    forecasts={'first':dict(prediction=truth,row_ids=corpus['row_ids']),
               'second':dict(prediction=truth+.3,row_ids=corpus['row_ids'])}
    def selected(permutations):
        return audit_interference(corpus['context'],corpus['component_future'],forecasts,metadata,corpus['row_ids'],
            design,n_boot=19,n_permutations=permutations,primary_estimand='sign_probability')[0]
    low=selected(9);high=selected(199)
    assert low['selected_primary']==high['selected_primary']
    assert low['sign_inference_resolution']==high['sign_inference_resolution']
    assert low['inference_resolution']['can_reject_at_0_05']
    assert not low['mean_inference_resolution']['can_reject_at_0_05']
    assert low['selected_primary']['second-minus-first/mixed_scaled_mse']['statistical_status']=='detected'
    with pytest.raises(ValueError,match='primary_estimand'):
        audit_interference(corpus['context'],corpus['component_future'],forecasts,metadata,corpus['row_ids'],
            design,primary_estimand='sign')
    from run_feature_interference import validate_config
    with pytest.raises(ValueError,match='primary_estimand'):
        validate_config(dict(models=[dict(name='a',adapter='mock_patch',checkpoint='')],
                             inference=dict(primary_estimand='median')))


def test_report_uses_selected_sign_and_exposes_magnitude_disagreement(tmp_path):
    """A majority of small losses can coexist with a large negative mean gap."""
    from html.parser import HTMLParser
    from tsfm_lens.analysis.feature_interference import render_interference_report
    design,corpus,metadata,truth=fixture(n=64)
    differences=np.r_[np.ones(63),-1000.]
    energy=np.array([m['scale']**2 for m in metadata])
    groups=np.array([m['seed'] for m in metadata])
    forecasts={
        'first':dict(prediction=truth+np.sqrt(energy*1001)[:,None],row_ids=corpus['row_ids']),
        'second':dict(prediction=truth+np.sqrt(energy*(1001+differences[groups]))[:,None],row_ids=corpus['row_ids']),
    }
    report,_=audit_interference(corpus['context'],corpus['component_future'],forecasts,metadata,corpus['row_ids'],
        design,n_boot=19,n_permutations=9,primary_estimand='sign_probability')
    path=tmp_path/'report.html'
    render_interference_report(report,path,{}, {},corpus['context'],corpus['component_future'],forecasts,metadata)
    class Rows(HTMLParser):
        """Read actual table cells, ignoring embedded Plotly script text."""
        def __init__(self):
            super().__init__();self.rows=[];self.row=[];self.cell=None
        def handle_starttag(self,tag,attributes):
            if tag=='tr':self.row=[]
            if tag in ('td','th'):self.cell=''
        def handle_data(self,data):
            if self.cell is not None:self.cell+=data
        def handle_endtag(self,tag):
            if tag in ('td','th') and self.cell is not None:
                self.row.append(self.cell);self.cell=None
            if tag=='tr':self.rows.append(self.row)
    parser=Rows();text=path.read_text(encoding='utf-8');parser.feed(text)
    row=next(row for row in parser.rows if row and row[0]=='second-minus-first/mixed_scaled_mse')
    assert float(row[1])==pytest.approx(63/64)
    assert float(row[6])<0 and float(row[7])==pytest.approx(1.)
    assert row[9]=='opposite directions'
    assert 'Clopper-Pearson' in text and 'not a generic model-winner label' in text


def test_producer_provenance_binds_protocol_and_forecast_hashes(tmp_path):
    """Portable replay retains its original plan and refuses swapped provenance."""
    from pathlib import Path
    from run_feature_interference import analyze_files, validate_config
    source = Path(__file__).resolve().parents[2] / 'examples' / 'feature_interference'
    _, cfg = validate_config(json.loads((source / 'protocol.json').read_text()))
    producer = json.loads((source / 'producer_analysis_provenance.json').read_text())
    protocol = tmp_path / 'protocol.json'
    protocol.write_bytes((source / 'protocol.json').read_bytes())
    provenance = tmp_path / 'producer.json'
    provenance.write_text(json.dumps(producer))
    predictions = {n: source / f'{n}_predictions.npz' for n in ('bolt', 'timesfm')}
    def run(name):
        out = tmp_path / name
        out.mkdir()
        (out / 'protocol.json').write_text(json.dumps(cfg))
        analyze_files(cfg, source / 'corpus.npz', source / 'metadata.json', predictions,
                      out, producer_provenance=provenance)
        return json.loads((out / 'analysis_provenance.json').read_text())
    original = run('original')['producer_inference']
    assert original['registered_plan_preserved'] is True
    cfg['inference'].update(n_boot=9, n_permutations=9)
    recorded = run('valid')['producer_inference']
    assert recorded['original_primary_estimand'] == 'mean'
    assert not recorded['registered_plan_preserved']  # Resampling plan changed explicitly.
    legal_decoy = json.loads(protocol.read_text())
    legal_decoy.setdefault('inference', {})['n_boot'] = 11
    protocol.write_text(json.dumps(legal_decoy))
    with pytest.raises(ValueError, match='protocol bytes'):
        run('wrong_protocol')
    protocol.write_bytes((source / 'protocol.json').read_bytes())
    producer['forecasts']['bolt']['sha256'] = '0' * 64
    provenance.write_text(json.dumps(producer))
    with pytest.raises(ValueError, match='forecast hash mismatch'):
        run('wrong_forecast')
