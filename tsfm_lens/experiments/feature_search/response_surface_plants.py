"""Run focused tests against in-memory source mutations without editing frozen files."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path


def run():
    """Require each deliberately broken implementation to raise an assertion failure."""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('response_surface_tests', root/'tests/test_response_surface_pilot.py')
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    plants = [
        ('curved_path_confounded', 'response_surface_metrics', 'local_response',
         'chord_second = rms(yp+ym-2*yc)', 'chord_second = rms(yp+ym-2*y0)',
         'test_curved_input_path_does_not_imply_nonlinear_model'),
        ('parameter_step_instead_of_input_distance', 'response_surface_metrics', 'local_response',
         'distance_sensitivity=ratio(secant, radius)', 'distance_sensitivity=ratio(secant, step)',
         'test_centered_derivative_distance_and_quadratic_curvature'),
        ('four_point_missing_offset', 'response_surface_metrics', 'interaction',
         'delta = yab-ya-yb+y0', 'delta = yab-ya-yb',
         'test_four_point_interaction_uses_offset_and_individual_effects'),
        ('frequency_changes_phase', 'response_surface_study', 'make_design',
         'return tone(f, a*scale, phase, np.arange(-128, 0))', 'return tone(f, a*scale, phase+f, np.arange(-128, 0))',
         'test_phase_distance_and_chord_design_are_bound_to_same_background'),
        ('fp32_before_subtraction', 'response_surface_metrics', 'difference',
         'return np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)',
         'return (np.asarray(a, dtype=np.float32)-np.asarray(b, dtype=np.float32)).astype(np.float64)',
         'test_saved_arithmetic_promotes_before_subtraction_and_semantics_are_separate'),
        ('conditions_count_as_sources', 'response_surface_metrics', 'summarize',
         "item['n_sources'] = len(sources)", "item['n_sources'] = sum(len(records) for records in sources.values())",
         'test_grouped_summary_does_not_count_conditions_as_independent_sources'),
        ('decoder_test_source_leak', 'response_surface_analysis', 'decoder_cv',
         'test = groups == group; train = ~test', 'test = groups == group; train = np.ones(len(groups), dtype=bool)',
         'test_decoder_withholds_sources_and_training_scaler_and_beats_shuffled_floor'),
    ]
    results = []
    for name, module, symbol, old, new, test in plants:
        path = root/'experiments/feature_search'/f'{module}.py'
        original = path.read_text(encoding='utf-8')
        if original.count(old) != 1: raise ValueError(f'Plant not applied exactly once: {name}')
        namespace = dict(__name__='response_surface_mutant', __file__=str(path))
        exec(compile(original.replace(old, new), str(path), 'exec'), namespace)
        correct = getattr(tests, symbol); setattr(tests, symbol, namespace[symbol])
        try:
            try: getattr(tests, test)()
            except AssertionError: results.append(dict(plant=name, test=test, detected=True))
            else: raise AssertionError(f'Plant survived: {name}')
        finally: setattr(tests, symbol, correct)
    print(json.dumps(results, indent=2))


if __name__ == '__main__': run()
