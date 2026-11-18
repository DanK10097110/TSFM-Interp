"""Trend/seasonality known answers, full Gram checks and signed paired families."""
import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / 'experiments/feature_search/trend_seasonality.py'
spec = importlib.util.spec_from_file_location('trend_seasonality', path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture():
    """Plant different native and actual-component-sum errors for three models."""
    protocol = study.load_protocol()
    protocol['n_bundles'] = 16
    protocol['inference'].update(n_boot=99, n_permutations=999)
    context, truth, metadata = study.make_corpus(protocol)
    scale = np.sqrt([row['variance_scale'] for row in metadata])[:, None]
    mixed = np.array([row['condition'] == 'mixed' for row in metadata])
    timesfm = truth + np.where(mixed[:, None], .8, 0.) * scale
    bolt = truth + np.where(mixed[:, None], .4, .1) * scale
    chronos2 = truth + .05 * scale
    return protocol, context, truth, metadata, dict(timesfm=timesfm, bolt=bolt, chronos2=chronos2)


def test_exact_input_control_common_future_variance_and_full_gram():
    """Physical time and nonorthogonal trend/seasonal terms cannot be discarded."""
    protocol, context, truth, metadata, _ = fixture()
    for period in protocol['periods']:
        keep = np.array([row['period'] == period for row in metadata])
        baseline = study.input_forecast_baseline(context[keep], period, 48)
        np.testing.assert_allclose(baseline, truth[keep], atol=1e-6)
        prediction = truth[keep] + .4 + .2 * np.arange(48)[None, :] / 48
        result = study.error_decomposition(prediction, truth[keep], period, 48)
        np.testing.assert_allclose(result['reconstructed_mse'], result['total_mse'], atol=1e-12)
        assert np.abs(result['gram_cross_terms']).max() > .01
    validation = study.validate_corpus(context, truth, metadata, protocol)
    assert validation['max_input_ols_forecast_mse_raw'] < 1e-10
    for seed in range(16):
        bundle = [row for row in metadata if row['seed'] == seed]
        assert len(bundle) == 18
        np.testing.assert_allclose([row['variance_scale'] for row in bundle], bundle[0]['variance_scale'])
    realized = np.var(truth, axis=1)
    declared = np.array([row['variance_scale'] for row in metadata])
    mixed = np.array([row['condition'] == 'mixed' for row in metadata])
    assert np.max(np.abs(realized[mixed] - declared[mixed])) > .01


def test_actual_background_affine_composition_and_native_48_step_path():
    """Actual background subtraction cancels affine bias; native calls preserve H=48."""
    _, context, _, metadata, _ = fixture()
    prediction = np.repeat((.7 * context[:, -1].astype(float) + .3 * context.mean(axis=1, dtype=float) + .2)[:, None], 48, axis=1)
    for period in (24, 48):
        rows = study.matched_rows(metadata, period)
        np.testing.assert_allclose(study.component_sum(prediction, rows), prediction[rows['mixed']], atol=1e-6)
    class FakeAdapter:
        def __init__(self):
            self.calls = []
        def predict(self, values, horizon, quantiles):
            self.calls.append((len(values), horizon))
            return dict(point=np.repeat(values[:, -1, None], horizon, axis=1))
    adapter = FakeAdapter()
    output = study.native_predict(adapter, context[:19], 48)
    assert output.shape == (19, 48)
    assert adapter.calls == [(8, 48), (8, 48), (3, 48)]


def test_signed_later_minus_earlier_fixed_holm_twelve_and_phase_undefined():
    """Negative discoveries remain valid and undefined phases cannot shrink the family."""
    protocol, _, truth, metadata, predictions = fixture()
    result = study.analyze(predictions, truth, metadata, protocol)
    assert result['family_size'] == 12
    assert len(result['primary']) == 12
    assert result['n_seeds'] == 16
    gap = result['primary']['native_gap_bolt_minus_timesfm_period24']
    np.testing.assert_allclose(gap['value'], -.48, atol=1e-12)
    interaction = result['primary']['component_sum_gain_chronos2_minus_timesfm_period48']
    np.testing.assert_allclose(interaction['value'], -.64, atol=1e-12)
    assert all(value['passed'] and value['direction'] == 'negative' for value in result['primary'].values())
    reverse_protocol = dict(protocol, models=list(reversed(protocol['models'])))
    reverse = study.analyze(predictions, truth, metadata, reverse_protocol)
    np.testing.assert_allclose(reverse['primary']['native_gap_timesfm_minus_bolt_period24']['value'], .48, atol=1e-12)
    flat = study.analyze({name: np.zeros_like(truth) for name in predictions}, truth, metadata, protocol)
    assert len(flat['primary']) == 12
    assert flat['descriptive']['24']['timesfm']['behavior']['mixed']['seasonal_phase_error_radians'] is None
