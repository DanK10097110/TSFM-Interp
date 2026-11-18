"""Oracle component-sum diagnostics, including an offset-bias counterexample."""
import importlib.util
from pathlib import Path
import numpy as np
from experiments.feature_search.mixture_replication import make_corpus,load_protocol

path=Path(__file__).resolve().parents[1]/'experiments/feature_search/mixture_composition_audit.py'
spec=importlib.util.spec_from_file_location('mixture_composition_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    """Use exact pure forecasts and suppressed mixed amplitudes."""
    protocol=load_protocol();protocol['n_bundles']=12
    _,truth,rows=make_corpus(protocol);prediction=truth.copy();t=np.arange(128,160)
    for i,row in enumerate(rows):
        if row['condition']=='mixed':
            prediction[i]=row['offset']+.4*row['a']*np.sin(2*np.pi*t/32+row['phi'])+.6*row['b']*np.sin(2*np.pi*t/16+row['psi'])
    return truth,prediction,rows


def test_oracle_component_sum_recovers_suppressed_mixture_and_shared_units():
    """Removing mixture suppression restores exact truth without fitted parameters."""
    truth,prediction,rows=fixture()
    result,composed=audit.composition_diagnostics(prediction,truth,rows,n_boot=99)
    mixed=np.array([row['condition']=='mixed' for row in rows])
    np.testing.assert_allclose(composed,truth[mixed],atol=1e-12)
    assert result['metrics']['native_error_mse_common_mixvariance']['value']>.1
    assert result['metrics']['component_sum_error_mse_common_mixvariance']['value']<1e-20
    np.testing.assert_allclose(result['metrics']['phi_amplitude_suppression_relative_to_true']['value'],.6,atol=1e-12)
    np.testing.assert_allclose(result['metrics']['psi_amplitude_suppression_relative_to_true']['value'],.4,atol=1e-12)
    np.testing.assert_allclose(result['fraction_native_error_removed_by_oracle_component_forecast_sum']['common_mixvariance_mse']['value'],1,atol=1e-12)
    assert result['fitted_parameters']==0


def test_oracle_offset_is_not_model_constant_forecast_and_can_worsen_error():
    """A constant forecast bias makes oracle-offset composition worse, not better."""
    truth,_,rows=fixture()
    exact,_=audit.composition_diagnostics(truth,truth,rows,n_boot=99)
    assert exact['metrics']['composition_defect_mse_raw']['value']<1e-20
    assert 'untestable' in exact['fraction_native_error_removed_by_oracle_component_forecast_sum']['raw_mse']
    result,_=audit.composition_diagnostics(truth+.2,truth,rows,n_boot=99)
    np.testing.assert_allclose(result['metrics']['native_error_mse_raw']['value'],.04,atol=1e-12)
    np.testing.assert_allclose(result['metrics']['component_sum_error_mse_raw']['value'],.16,atol=1e-12)
    np.testing.assert_allclose(result['fraction_native_error_removed_by_oracle_component_forecast_sum']['raw_mse']['value'],-3,atol=1e-12)
    assert 'oracle_exact_offset' in result['baseline']
    assert 'NOT f(offset)' in result['baseline']
