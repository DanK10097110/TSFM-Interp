"""Known-answer fixtures include the confusable curved linear path and large DC offsets."""
import numpy as np
from experiments.feature_search.response_surface_metrics import (
    difference, known_response, local_response, interaction, summarize)
from experiments.feature_search.response_surface_study import make_design, tone, CENTERS
from experiments.feature_search.response_surface_analysis import decoder_cv


def test_curved_input_path_does_not_imply_nonlinear_model():
    """A linear map inherits frequency-path bend, but has zero chord curvature."""
    t = np.arange(-128, 0); f = CENTERS[0]; step = .06
    xm, x0, xp = [tone(f+d, 2., .4, t) for d in [-step, 0, step]]
    midpoint = (xm+xp)/2
    matrix = np.random.default_rng(7).normal(size=(128, 9))
    result = local_response(xm@matrix, x0@matrix, xp@matrix, midpoint@matrix, xm, x0, xp, step, 1.)
    assert result['path_second_rms'] > .01
    assert result['chord_relative'] < 1e-12
    assert result['input_path_bend'] > .001


def test_centered_derivative_distance_and_quadratic_curvature():
    """A quadratic's derivatives and mixed differences have exact nonzero answers."""
    x0 = np.array([2., 2.]); d = np.array([.01, .01])
    r = local_response((x0-d)**2, x0**2, (x0+d)**2, x0**2, x0-d, x0, x0+d, .01, 2.)
    np.testing.assert_allclose(r['derivative_rms'], 4.)
    np.testing.assert_allclose(r['distance_sensitivity'], 8.)
    np.testing.assert_allclose(r['parameter_second_rms'], 2.)
    np.testing.assert_allclose(r['chord_curvature_per_distance2'], 8.)


def test_four_point_interaction_uses_offset_and_individual_effects():
    """Large offsets must cancel and a quadratic retains exactly2ab."""
    x, a, b = np.array([2., 2.]), np.array([.1, .1]), np.array([.3, .3])
    affine = lambda q: 7*q+1000
    linear = interaction(affine(x), affine(x+a), affine(x+b), affine(x+a+b), a, b, 2.)
    assert linear['interaction_rms'] < 1e-12
    quadratic = lambda q: q*q+1000
    nonlinear = interaction(quadratic(x), quadratic(x+a), quadratic(x+b), quadratic(x+a+b), a, b, 2.)
    np.testing.assert_allclose(nonlinear['interaction_rms'], .06)
    np.testing.assert_allclose(nonlinear['relative_sum'], .06/(.41+1.29))
    np.testing.assert_allclose(nonlinear['per_input_product'], 8.)
    dead = interaction(x, x, x, x, a, b, 2.)
    assert dead['relative_sum'] is None


def test_phase_distance_and_chord_design_are_bound_to_same_background():
    """No phase resampling or history replacement is permitted along a surface."""
    x = np.random.default_rng(11).normal(size=128)
    inputs, rows, paths, pairs, signals = make_design([(dict(source='s', domain='synthetic'), x)])
    assert len(set(r['phase'] for r in rows)) == 1
    for s in signals:
        expected = inputs[s['base']].astype(float)+tone(s['frequency'], s['amplitude']*s['scale'], s['phase'], np.arange(-128, 0))
        np.testing.assert_array_equal(inputs[s['index']], expected.astype(np.float32))
    assert len(signals) == 42 and len(paths) == 30 and len(pairs) == 9
    assert all(abs(s['frequency']-round(s['frequency'])) > .01 for s in signals)
    for path in paths:
        midpoint = (inputs[path['minus']].astype(float)+inputs[path['plus']])/2
        np.testing.assert_array_equal(inputs[path['midpoint']], midpoint.astype(np.float32))
    for pair in pairs:
        closure = inputs[pair['ab']].astype(float)-inputs[pair['a']]-inputs[pair['b']]+inputs[pair['base']]
        assert np.max(abs(closure)) < 5e-7


def test_saved_arithmetic_promotes_before_subtraction_and_semantics_are_separate():
    """FP32 subtraction loses the small operand next to a large state."""
    a = np.array([2**24], np.float32); b = np.array([-1], np.float32)
    assert difference(a, b)[0] == 2**24+1
    signal = np.sin(np.arange(32))
    assert known_response(signal, signal)['nmse'] == 0
    assert known_response(np.zeros(32), signal)['nmse'] == 1
    assert known_response(-signal, signal)['gain'] == -1


def test_grouped_summary_does_not_count_conditions_as_independent_sources():
    """Uneven duplicated conditions must not change equal-source weighting."""
    rows = [dict(source='a', model='m', value=2.)]*10+[dict(source='b', model='m', value=10.)]
    result = summarize(rows, ['model'], ['value'])[0]
    assert result['n_sources'] == 2 and result['value']['mean'] == 6.


def test_decoder_withholds_sources_and_training_scaler_and_beats_shuffled_floor():
    """Planted accessible Fourier coefficients survive source-held-out linear decoding."""
    rng = np.random.default_rng(8); groups = np.repeat(np.arange(8), 9)
    labels = rng.normal(size=(72, 4)); matrix = rng.normal(size=(4, 16))
    features = labels@matrix
    result, pred, shuffled, mean = decoder_cv(features, labels, groups)
    assert result['nmse'] < .01 and result['shuffled_nmse'] > .5
    for fold in result['folds']:
        assert fold['source_id'] not in fold['train_sources'] and fold['test_rows'] == 9
    altered = features.copy(); altered[groups == 0] *= 1000
    changed, _, _, _ = decoder_cv(altered, labels, groups)
    assert changed['folds'][0]['feature_scale'] == result['folds'][0]['feature_scale']
    assert pred.shape == shuffled.shape == mean.shape == labels.shape
