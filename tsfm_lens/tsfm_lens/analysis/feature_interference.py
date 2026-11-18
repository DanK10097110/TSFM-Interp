"""Controlled two-feature forecasting interference, without hidden-state claims.

Eight pure/mixed inputs share each generating seed's amplitudes, offset and
mixed-energy denominator. An opt-in ninth cell supplies a native background
forecast. Whole generating bundles are the statistical units. Component sums
remain controlled composition diagnostics, not generic deployment rules.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np
from scipy.stats import binomtest
from .stats import holm, mean_ci

@dataclass(frozen=True)
class InterferenceDesign:
    """A frozen periodic factorial with identifiable, orthogonal frequencies."""
    context: int = 128
    horizon: int = 32
    periods: tuple = (32, 16)
    phase_shift: float = np.pi / 2
    n_bundles: int = 128
    seed_base: int = 1500000
    amplitudes: tuple = (0.5, 2.0)
    offset: tuple = (-2.0, 2.0)
    phase_floor: float = 1e-06
    native_background: bool = False

    def __post_init__(self):
        """Refuse aliases, untiled extents and degenerate input interventions."""
        if not isinstance(self.native_background, bool):
            raise ValueError('native_background must be true or false')
        integers = (self.context, self.horizon, self.n_bundles, self.seed_base, *self.periods)
        if any((isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in integers)):
            raise ValueError('context/horizon/periods/count/seed must be integers')
        if len(self.periods) != 2 or len(set(self.periods)) != 2 or min(self.periods) <= 2:
            raise ValueError('need two distinct below-Nyquist frequencies: periods must exceed 2')
        if self.n_bundles < 3 or self.seed_base < 0 or any((self.context % p or self.horizon % p for p in self.periods)):
            raise ValueError('need at least 3 bundles and both periods must tile context AND horizon')
        if self.context <= 0 or self.horizon <= 0 or (not np.isfinite(self.phase_shift)) or (abs(np.sin(self.phase_shift / 2)) < 1e-08):
            raise ValueError('need positive extents and a nonzero circular phase intervention')
        if len(self.amplitudes) != 2 or len(self.offset) != 2 or min(self.amplitudes) <= 0 or (self.amplitudes[0] > self.amplitudes[1]) or (self.offset[0] > self.offset[1]) or (not np.isfinite([*self.amplitudes,
                    *self.offset, self.phase_floor]).all()) or (self.phase_floor <= 0):
            raise ValueError('need finite amplitude/offset ranges and positive phase floor')
        harmonic_design(self.context, self.horizon, self.periods)

    @property
    def rows_per_bundle(self):
        """Retain the historical eight cells unless native background is enabled."""
        return 9 if self.native_background else 8


def _factorial_cells(design):
    """Preserve the original eight-cell order and optionally append background."""
    cells = [('mixed', p, q) for p in (0, 1) for q in (0, 1)]
    cells += [('pure_phi', p, 0) for p in (0, 1)]
    cells += [('pure_psi', 0, q) for q in (0, 1)]
    if design.native_background:
        cells.append(('background', 0, 0))
    return cells

def harmonic_design(start, length, periods):
    """Return an exactly tiled two-frequency basis with explicit rank checks."""
    t = np.arange(start, start + length)
    matrix = np.column_stack([f(2 * np.pi * t / p) for p in periods for f in (np.sin,
                np.cos)] + [np.ones(length)])
    condition = float(np.linalg.cond(matrix))
    if np.linalg.matrix_rank(matrix) != 5 or condition > 1000 or (not np.allclose(matrix.T @ matrix / length,
            np.diag([0.5, 0.5, 0.5, 0.5, 1]), atol=1e-10)):
        raise ValueError('two-frequency harmonic basis must be full-rank, conditioned and orthogonal')
    return matrix

def canonical_row_ids(metadata):
    """Name physical factorial rows; forecast artifacts must carry these IDs."""
    return np.asarray([f"{m['generator_seed']}:{m['condition']}:{m['cell_phi']}:{m['cell_psi']}" for m in metadata])

def make_factorial(design):
    """Generate clean pure/mixed inputs with identical nuisance draws per bundle."""
    t = np.arange(design.context + design.horizon)
    contexts = []
    components = []
    metadata = []
    cells = _factorial_cells(design)
    for index in range(design.n_bundles):
        rng = np.random.default_rng(design.seed_base + index)
        a, b = rng.uniform(*design.amplitudes, 2)
        offset = rng.uniform(*design.offset)
        phi, psi = rng.uniform(-np.pi, np.pi, 2)
        for condition, p, q in cells:
            angles = (phi + p * design.phase_shift, psi + q * design.phase_shift)
            atoms = np.array([a * np.sin(2 * np.pi * t / design.periods[0] + angles[0]),
                    b * np.sin(2 * np.pi * t / design.periods[1] + angles[1])])
            if condition == 'pure_phi':
                atoms[1] = 0
            if condition == 'pure_psi':
                atoms[0] = 0
            if condition == 'background':
                atoms[:] = 0
            contexts.append(atoms.sum(0)[:design.context] + offset)
            components.append(atoms[:, design.context:].copy())
            metadata.append(dict(seed=index, generator_seed=design.seed_base + index,
                    condition=condition, cell_phi=p, cell_psi=q, phi=angles[0], psi=angles[1], a=a, b=b,
                    offset=offset, scale=float(np.sqrt((a * a + b * b) / 2))))
    return (dict(context=np.asarray(contexts, dtype=np.float32), component_future=np.asarray(components),
            row_ids=canonical_row_ids(metadata)), metadata)

def validate_factorial(contexts, components, metadata, row_ids, design):
    """Prove exact input pairing, component truth and artifact row identity."""
    contexts = np.asarray(contexts)
    components = np.asarray(components)
    n = len(metadata)
    if contexts.shape != (n, design.context) or components.shape != (n, 2, design.horizon) or (not np.isfinite(contexts).all()) or (not np.isfinite(components).all()):
        raise ValueError('need finite contexts and true component futures aligned to metadata')
    canonical = canonical_row_ids(metadata)
    if len(set(canonical)) != n or not np.array_equal(np.asarray(row_ids), canonical):
        raise ValueError('corpus row IDs must uniquely match declared factorial metadata IN ORDER')
    lookup = {}
    expected = set(_factorial_cells(design))
    t = np.arange(design.context + design.horizon)
    for i, m in enumerate(metadata):
        key = (m['seed'], m['condition'], m['cell_phi'], m['cell_psi'])
        if key in lookup or key[1:] not in expected:
            raise ValueError('duplicate or unknown factorial condition cell')
        lookup[key] = i
        if not np.isfinite([m[k] for k in ('a', 'b', 'phi', 'psi', 'offset', 'scale')]).all() or min(m['a'],
            m['b'], m['scale']) <= 0:
            raise ValueError('need finite matched nuisances and positive true component amplitudes')
        if any((not design.amplitudes[0] <= m[k] <= design.amplitudes[1] for k in ('a',
                    'b'))) or not design.offset[0] <= m['offset'] <= design.offset[1]:
            raise ValueError('metadata amplitudes/offset must agree with the frozen configured ranges')
        if not np.isclose(m['scale'] ** 2, (m['a'] ** 2 + m['b'] ** 2) / 2, rtol=1e-10):
            raise ValueError('all pure/mixed conditions require the SAME mixed-energy scale')
        atoms = np.array([m['a'] * np.sin(2 * np.pi * t / design.periods[0] + m['phi']),
                m['b'] * np.sin(2 * np.pi * t / design.periods[1] + m['psi'])])
        if m['condition'] == 'pure_phi':
            atoms[1] = 0
        if m['condition'] == 'pure_psi':
            atoms[0] = 0
        if m['condition'] == 'background':
            atoms[:] = 0
            if np.any(components[i] != 0):
                raise ValueError('background component futures must be exactly zero')
        if not np.allclose(contexts[i], atoms.sum(0)[:design.context] + m['offset'], atol=2e-06, rtol=2e-06):
            raise ValueError('observed context disagrees with declared controlled components')
        if not np.allclose(components[i], atoms[:, design.context:], atol=1e-09, rtol=1e-08):
            raise ValueError('true component future disagrees with physical intervention metadata')
    seeds = np.unique([m['seed'] for m in metadata])
    if len(seeds) != design.n_bundles:
        raise ValueError('bundle count does not match frozen design')
    if not np.array_equal(seeds, np.arange(design.n_bundles)):
        raise ValueError('metadata seed indices must match the frozen 0..n_bundles-1 design')
    for seed in seeds:
        bundle = [m for m in metadata if m['seed'] == seed]
        if {(m['condition'], m['cell_phi'], m['cell_psi']) for m in bundle} != expected or len(bundle) != design.rows_per_bundle:
            raise ValueError(f'every generating seed needs the complete {design.rows_per_bundle}-cell factorial')
        for key in ('a', 'b', 'offset', 'scale', 'generator_seed'):
            if not np.allclose([m[key] for m in bundle], bundle[0][key], atol=1e-12, rtol=1e-12):
                raise ValueError('amplitudes/offset/scale/generator must match within seed')
        if bundle[0]['generator_seed'] != design.seed_base + int(seed):
            raise ValueError('generator seed does not match frozen seed_base and bundle index')
        for phase in ('phi', 'psi'):
            base = bundle[0][phase] - bundle[0]['cell_' + phase] * design.phase_shift
            differences = [np.angle(np.exp(1j * (m[phase] - base - m['cell_' + phase] * design.phase_shift))) for m in bundle]
            if not np.allclose(differences, 0, atol=1e-10):
                raise ValueError('only the declared component phase may change within a matched pair')
    return (seeds, lookup)

def harmonic_components(values, design):
    """Fit component coefficients while retaining out-of-basis forecast residuals."""
    matrix = harmonic_design(design.context, design.horizon, design.periods)
    coefficients = np.linalg.lstsq(matrix, np.asarray(values, dtype=float).T, rcond=None)[0].T
    return (coefficients, np.asarray(values) - coefficients @ matrix.T)

def error_decomposition(prediction, truth, design):
    """Retain physical MSE; zero-amplitude phase is assigned to amplitude error.

    This algebraic convention does not identify a zero-amplitude forecast phase.
    """
    pc, pr = harmonic_components(prediction, design)
    tc, tr = harmonic_components(truth, design)
    channels = {'offset': (pc[:, 4] - tc[:, 4]) ** 2, 'residual': ((pr - tr) ** 2).mean(1)}
    total = channels['offset'] + channels['residual']
    for index, name in enumerate(('phi', 'psi')):
        p = pc[:, 2 * index:2 * index + 2]
        t = tc[:, 2 * index:2 * index + 2]
        ap = np.linalg.norm(p, axis=1)
        at = np.linalg.norm(t, axis=1)
        denom = ap * at
        cosine = np.divide((p * t).sum(1), denom, out=np.ones(len(p)), where=denom > 1e-24)
        cosine = np.clip(cosine, -1, 1)
        component = 0.5 * ((p - t) ** 2).sum(1)
        channels[name + '_coefficient'] = component
        channels[name + '_amplitude'] = 0.5 * (ap - at) ** 2
        channels[name + '_phase_at_true_amplitude'] = at ** 2 * (1 - cosine)
        channels[name + '_amplitude_phase_interaction'] = at * (ap - at) * (1 - cosine)
        if not np.allclose(channels[name + '_amplitude'] + channels[name + '_phase_at_true_amplitude'] + channels[name + '_amplitude_phase_interaction'],
            component, atol=1e-10, rtol=1e-08):
            raise ValueError('amplitude/phase algebraic decomposition failed')
        total = total + component
    direct = ((np.asarray(prediction) - truth) ** 2).mean(1)
    if not np.allclose(total, direct, atol=1e-10, rtol=1e-08):
        raise ValueError('orthogonal components failed to reconstruct full forecast MSE')
    return channels

def _seed_vector(values):
    """Keep one observation per seed; NaN is explicit missingness, infinity is invalid."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError('seed metrics must be one-dimensional; rows must not be flattened into pseudo-seeds')
    if np.isinf(vector).any():
        raise ValueError('infinite seed metrics are invalid; only explicit NaN may mark undefined phase')
    return vector


def paired_seed_effect(differences, *, n_boot=2000, n_permutations=1999, seed=113, ci=0.95):
    """Mean seed contrast with bootstrap CI and two-sided sign-flip plus-one p.

    The sign-flip null assumes symmetric seed differences or exchangeable paired
    labels. The interval conditions on these fixed native model forecasts.
    """
    differences = _seed_vector(differences)
    valid = np.isfinite(differences)
    values = differences[valid]
    if len(values) < 3:
        return dict(p=1.0, untestable='fewer than 3 jointly identifiable seed bundles',
            n_seeds=len(values), excluded_seeds=int((~valid).sum()))
    if min(n_boot, n_permutations) < 1 or not 0 < ci < 1:
        raise ValueError('need positive resampling counts and valid CI level')
    result = mean_ci(values, n_boot=n_boot, seed=seed, ci=ci, unit='generating_seed_bundle')
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    count = sum((abs(np.mean(values * rng.choice((-1, 1), len(values)))) >= observed for _ in range(n_permutations)))
    result.update(p=float((1 + count) / (n_permutations + 1)), n_seeds=len(values),
        excluded_seeds=int((~valid).sum()), inference='fixed predictions; two-sided paired seed signflip; symmetry/exchangeability assumed')
    return result

def sign_sensitivity(values):
    """Exact positive-sign sensitivity; ties excluded from its distinct estimand."""
    values = _seed_vector(values)
    valid = np.isfinite(values)
    x = values[valid]
    positive = int((x > 0).sum())
    negative = int((x < 0).sum())
    nonzero = positive + negative
    return dict(p_sign=1.0 if nonzero == 0 else float(binomtest(positive, nonzero, 0.5,
                alternative='greater').pvalue), positive=positive, negative=negative, ties=int((x == 0).sum()),
        excluded_seeds=int((~valid).sum()), fraction_positive_all=None if not len(x) else positive / len(x),
        fraction_positive_nonzero=None if not nonzero else positive / nonzero, median=None if not len(x) else float(np.median(x)),
        estimand='P(positive seed contrast | nonzero)>.5; differs from mean contrast')


def sign_comparison(values, *, ci=0.95):
    """Exact two-sided comparison of positive-sign probability with one half.

    The endpoint is conditional on jointly identifiable, nonzero seed contrasts.
    Clopper-Pearson intervals are pointwise, not multiplicity-adjusted. Signed
    mean and median are descriptive summaries of all identifiable seeds, including
    ties, and can disagree with the majority sign.
    """
    vector = _seed_vector(values)
    if not np.isfinite(ci) or not 0 < ci < 1:
        raise ValueError('sign comparison requires a confidence level between 0 and 1')
    valid = np.isfinite(vector)
    observed = vector[valid]
    positive = int((observed > 0).sum())
    negative = int((observed < 0).sum())
    nonzero = positive + negative
    mean = None if not len(observed) else float(observed.mean())
    median = None if not len(observed) else float(np.median(observed))
    result = {
        'value': None, 'lo': None, 'hi': None, 'p': 1.0,
        'positive': positive, 'negative': negative,
        'ties': int((observed == 0).sum()), 'n_seeds': len(observed),
        'n_nonzero': nonzero, 'excluded_seeds': int((~valid).sum()),
        'descriptive_signed_mean': mean, 'descriptive_signed_median': median,
        'mean_vs_majority_disagreement': None,
        'minimum_two_sided_p': 1.0 if nonzero == 0 else min(1.0, 2.0 ** (1 - nonzero)),
        'ci': ci, 'interval': 'pointwise Clopper-Pearson exact binomial interval',
        'estimand': 'q=P(D>0 | D!=0, jointly identifiable); H0:q=0.5; exact two-sided binomial',
        'resample_unit': 'generating_seed_bundle',
    }
    if nonzero == 0:
        result['untestable'] = 'no nonzero jointly identifiable seed contrasts'
        return result
    test = binomtest(positive, nonzero, p=0.5, alternative='two-sided')
    interval = test.proportion_ci(confidence_level=ci, method='exact')
    probability = positive / nonzero
    result.update(
        value=probability, lo=float(interval.low), hi=float(interval.high),
        p=float(test.pvalue), signed_balance=2 * probability - 1,
        mean_vs_majority_disagreement=bool(
            mean != 0 and probability != 0.5 and np.sign(mean) != np.sign(probability - 0.5)
        ),
    )
    return result


def _select_primary_family(mean_results, sign_results, primary_estimand):
    """Select one predeclared family; secondary endpoints never rescue its status."""
    if primary_estimand not in ('mean', 'sign_probability'):
        raise ValueError('primary_estimand must be mean or sign_probability')
    source = mean_results if primary_estimand == 'mean' else sign_results
    return {
        key: dict(
            value, primary_estimand=primary_estimand,
            statistical_status=(
                'untestable' if 'untestable' in value else
                'detected' if value['p_holm'] <= 0.05 else 'not_detected'
            ),
        )
        for key, value in source.items()
    }

def _interval(values, settings):
    """Represent undefined phases explicitly without writing NaN into JSON."""
    values = _seed_vector(values)
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return dict(untestable='fewer than 3 identifiable bundles', n_seeds=int(valid.sum()),
            excluded_seeds=int((~valid).sum()))
    result = mean_ci(values[valid], n_boot=settings['n_boot'], seed=settings['seed'],
        ci=settings['ci'], unit='generating_seed_bundle')
    result.update(n_seeds=int(valid.sum()), excluded_seeds=int((~valid).sum()))
    return result

def audit_interference(contexts, component_futures, forecasts, metadata, row_ids, design,
    *, n_boot=2000, n_permutations=1999, seed=113, ci=0.95, model_order=None,
    primary_estimand='mean'):
    """Audit controlled feature coupling for every prospectively named model pair.

    ``forecasts`` maps names to ``{'prediction': [rows,horizon], 'row_ids': [...]}``.
    The complete family contains four contrasts per unordered model pair; signed
    contrasts are later-listed model minus earlier-listed model. Solo models retain
    descriptive diagnostics. Predictions with missing or reordered IDs fail.
    Array callers must establish forecast/input provenance themselves. The
    artifact CLI additionally requires a matching exact source-corpus hash.
    """
    settings = dict(n_boot=n_boot, n_permutations=n_permutations, seed=seed, ci=ci)
    if primary_estimand not in ('mean', 'sign_probability'):
        raise ValueError('primary_estimand must be mean or sign_probability')
    if any((isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in (n_boot,
                n_permutations))) or not np.isfinite(ci) or (not 0 < ci < 1):
        raise ValueError('need positive integer resampling counts and finite valid CI level')
    seeds, lookup = validate_factorial(contexts, component_futures, metadata, row_ids, design)
    if not forecasts:
        raise ValueError('need at least one explicitly named forecast artifact')
    order = list(forecasts) if model_order is None else list(model_order)
    if len(set(order)) != len(order) or set(order) != set(forecasts):
        raise ValueError('model order must name every forecast exactly once')
    if min(n_boot, n_permutations) < 1 or not 0 < ci < 1:
        raise ValueError('invalid resampling settings')
    truth = np.asarray(component_futures).sum(1) + np.array([m['offset'] for m in metadata])[:, None]
    scale = np.array([m['scale'] ** 2 for m in metadata])
    groups = np.array([m['seed'] for m in metadata])
    conditions = np.array([m['condition'] for m in metadata])
    condition_names = ('mixed', 'pure_phi', 'pure_psi')
    if design.native_background:
        condition_names += ('background',)
    mixed = np.flatnonzero(conditions == 'mixed')
    models = {}
    errors = {}
    coupling = {}
    seed_values = {}
    for model, artifact in sorted(forecasts.items()):
        prediction = np.asarray(artifact['prediction'], dtype=float)
        if not np.array_equal(np.asarray(artifact.get('row_ids', [])), np.asarray(row_ids)):
            raise ValueError(f'{model}: forecast row IDs missing, permuted or mismatched')
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError(f'{model}: forecast shape or finiteness mismatch')
        pc, residual = harmonic_components(prediction, design)
        loss = ((prediction - truth) ** 2).mean(1) / scale
        errors[model] = {
            condition: np.array([
                loss[(groups == unit) & (conditions == condition)].mean()
                for unit in seeds
            ])
            for condition in condition_names
        }
        target_results = {}
        for target, index in [('phi', 0), ('psi', 1)]:
            observations = {key: [] for key in ('untouched_phase_change_radians', 'untouched_amplitude_change_relative',
                    'untouched_coefficient_change_mse', 'untouched_coefficient_change_normalized', 'changed_coefficient_change_normalized',
                    'phase_unidentifiable_fraction')}
            for s in seeds:
                values = {k: [] for k in observations}
                for direction in (0, 1):
                    for nuisance in (0, 1):
                        p, q = (direction, nuisance) if index == 0 else (nuisance, direction)
                        dp, dq = (1 - direction, nuisance) if index == 0 else (nuisance, 1 - direction)
                        r = lookup[s, 'mixed', p, q]
                        d = lookup[s, 'mixed', dp, dq]
                        other = 1 - index
                        rr = pc[r, 2 * other:2 * other + 2]
                        dd = pc[d, 2 * other:2 * other + 2]
                        amplitude = metadata[r]['b' if other else 'a']
                        changed_amp = metadata[r]['a' if index == 0 else 'b']
                        ar, ad = (np.linalg.norm(rr), np.linalg.norm(dd))
                        valid = min(ar, ad) > design.phase_floor * amplitude
                        angle = abs(np.angle(np.exp(1j * (np.arctan2(dd[1], dd[0]) - np.arctan2(rr[1],
                                            rr[0]))))) if valid else np.nan
                        values['untouched_phase_change_radians'].append(angle)
                        values['untouched_amplitude_change_relative'].append(abs(ad - ar) / amplitude)
                        change = 0.5 * np.sum((dd - rr) ** 2)
                        values['untouched_coefficient_change_mse'].append(change)
                        values['untouched_coefficient_change_normalized'].append(change / amplitude ** 2)
                        changed = pc[d, 2 * index:2 * index + 2] - pc[r, 2 * index:2 * index + 2]
                        values['changed_coefficient_change_normalized'].append(0.5 * np.sum(changed ** 2) / changed_amp ** 2)
                        values['phase_unidentifiable_fraction'].append(float(not valid))
                for key in observations:
                    observations[key].append(float(np.mean(values[key])))
            coupling[model, target] = np.array(observations['untouched_phase_change_radians'])
            target_results[target] = {k: _interval(v, settings) for k, v in observations.items()}
        phi = np.array([lookup[metadata[i]['seed'], 'pure_phi', metadata[i]['cell_phi'], 0] for i in mixed])
        psi = np.array([lookup[metadata[i]['seed'], 'pure_psi', 0, metadata[i]['cell_psi']] for i in mixed])
        if design.native_background:
            background = np.array([
                lookup[metadata[i]['seed'], 'background', 0, 0] for i in mixed
            ])
            subtraction = prediction[background]
            subtraction_mode = 'native_background_forecast'
            label = ('f(pure_phi) + f(pure_psi) - f(background); '
                     'actual constant-background forecast; known component inputs, '
                     'not a generic deployment improvement')
        else:
            subtraction = np.array([metadata[i]['offset'] for i in mixed])[:, None]
            subtraction_mode = 'oracle_exact_offset'
            label = ('f(pure_phi) + f(pure_psi) - exact known offset; '
                     'not f(offset); diagnostic not deployment improvement')
        composed = prediction[phi] + prediction[psi] - subtraction
        component_loss = ((composed - truth[mixed]) ** 2).mean(1) / scale[mixed]
        defect = ((prediction[mixed] - composed) ** 2).mean(1) / scale[mixed]
        aggregate = lambda values: np.array([values[groups[mixed] == s].mean() for s in seeds])
        decomposition = error_decomposition(prediction, truth, design)
        component_sum = {
            'subtraction_mode': subtraction_mode,
            'mse_common_mixvariance': _interval(aggregate(component_loss), settings),
            'composition_defect_common_mixvariance': _interval(aggregate(defect), settings),
            'error_reduction_common_mixvariance': _interval(
                errors[model]['mixed'] - aggregate(component_loss), settings
            ),
            'label': label,
        }
        model_report = {
            'scaled_mse': {c: _interval(v, settings) for c, v in errors[model].items()},
            'changed_target': target_results,
            'forecast_residual_mse': float(np.mean(residual ** 2)),
            'physical_error_decomposition': {
                condition: {
                    channel: {
                        'raw_mse_units_squared': float(values[conditions == condition].mean()),
                        'common_mixvariance_mse': float((values / scale)[conditions == condition].mean()),
                    }
                    for channel, values in decomposition.items()
                }
                for condition in condition_names
            },
            'component_sum': component_sum,
        }
        if not design.native_background:
            model_report['oracle_component_sum'] = component_sum
        else:
            background_rows = np.array([lookup[unit, 'background', 0, 0] for unit in seeds])
            background_error = prediction[background_rows] - truth[background_rows]
            background_bias = background_error.mean(1)
            model_report['native_background_forecast'] = {
                'raw_mse': _interval((background_error ** 2).mean(1), settings),
                'common_mixvariance_mse': _interval(
                    (background_error ** 2).mean(1) / scale[background_rows], settings
                ),
                'mean_bias': _interval(background_bias, settings),
                'mean_bias_relative_mix_rms': _interval(
                    background_bias / np.sqrt(scale[background_rows]), settings
                ),
            }
        models[model] = model_report
    primary = {}
    sensitivity = {}
    sign_results = {}
    pair_descriptive = {}
    for first, second in combinations(order, 2):
        values = {f'untouched_phase_{target}': coupling[second, target] - coupling[first,
                target] for target in ('phi', 'psi')}
        mixed_gap = errors[second]['mixed'] - errors[first]['mixed']
        pure_gap = 0.5 * (errors[second]['pure_phi'] - errors[first]['pure_phi'] + (errors[second]['pure_psi'] - errors[first]['pure_psi']))
        mixed_interval = _interval(mixed_gap, settings)
        pure_interval = _interval(pure_gap, settings)
        flip = mixed_interval.get('lo', 0) > 0 and pure_interval.get('hi', 0) < 0 or (mixed_interval.get('hi',
                0) < 0 and pure_interval.get('lo', 0) > 0)
        pair_descriptive[f'{second}-minus-{first}'] = dict(mixed_gap=mixed_interval,
            equal_weight_pure_gap=pure_interval, ranking_flip_descriptive=bool(flip), intervals='pointwise descriptive; not additional registered claims')
        values.update(mixed_scaled_mse=mixed_gap, model_by_composition=mixed_gap - pure_gap)
        for metric, difference in values.items():
            key = f'{second}-minus-{first}/{metric}'
            seed_values[key] = difference
            primary[key] = paired_seed_effect(difference, **settings)
            sign_results[key] = sign_comparison(difference, ci=ci)
            valid = np.isfinite(difference)
            primary[key]['valid_seed_ids'] = seeds[valid].tolist()
            primary[key]['excluded_seed_ids'] = seeds[~valid].tolist()
            sign_results[key]['valid_seed_ids'] = seeds[valid].tolist()
            sign_results[key]['excluded_seed_ids'] = seeds[~valid].tolist()
            sign_results[key]['nonzero_seed_ids'] = seeds[valid & (difference != 0)].tolist()
            if metric.startswith('untouched_phase_'):
                target = metric.removeprefix('untouched_phase_')
                primary[key]['paired_subset_model_means'] = {
                    name: float(coupling[name, target][valid].mean()) if valid.any() else None
                    for name in (first, second)
                }
            sensitivity[key] = sign_sensitivity(difference)
    adjusted = holm({k: v['p'] for k, v in primary.items()}) if primary else {}
    sign_adjusted = holm({k: v['p_sign'] for k, v in sensitivity.items()}) if sensitivity else {}
    exact_sign_adjusted = holm({k: value['p'] for k, value in sign_results.items()}) if sign_results else {}
    for key in primary:
        primary[key]['p_holm'] = adjusted[key]
        sensitivity[key]['p_sign_holm'] = sign_adjusted[key]
        sign_results[key]['p_holm'] = exact_sign_adjusted[key]
    family_size = len(primary)
    mean_resolution = dict(family_size=family_size, plus_one_p_floor=1 / (n_permutations + 1),
        smallest_possible_holm_p=None if family_size == 0 else min(1.0, family_size / (n_permutations + 1)),
        can_reject_at_0_05=bool(family_size and family_size / (n_permutations + 1) <= 0.05),
        scope='Monte Carlo mean sign-flip family; exact sign sensitivity has a different resolution')
    sign_floor = min((value['minimum_two_sided_p'] for value in sign_results.values()), default=1.0)
    sign_resolution = {
        'family_size': family_size,
        'nonzero_seed_counts': {key: value['n_nonzero'] for key, value in sign_results.items()},
        'minimum_two_sided_p_by_comparison': {
            key: value['minimum_two_sided_p'] for key, value in sign_results.items()
        },
        'smallest_possible_holm_p': None if not family_size else min(1.0, family_size * sign_floor),
        'can_reject_at_0_05': bool(family_size and family_size * sign_floor <= 0.05),
        'scope': 'exact two-sided sign family; resolution depends on nonzero seed counts, not permutation budget',
    }
    resolution = mean_resolution if primary_estimand == 'mean' else sign_resolution
    selected_primary = _select_primary_family(primary, sign_results, primary_estimand)
    report = dict(design=asdict(design), n_seed_bundles=len(seeds), n_rows=len(metadata),
        component_sum_mode='native_background_forecast' if design.native_background else 'oracle_exact_offset',
        practical_relevance='statistical effect summaries only; no minimum practical threshold or discovery gate is implemented',
        models=models, primary=primary, mean_comparisons=primary, sign_comparisons=sign_results,
        selected_primary=selected_primary, primary_estimand=primary_estimand,
        sign_sensitivity=sensitivity, pair_descriptive=pair_descriptive,
        family_size=family_size, inference_resolution=resolution,
        mean_inference_resolution=mean_resolution, sign_inference_resolution=sign_resolution,
        settings=settings, evidence='behavioral controlled input intervention and predictive comparison; no internal causal mediation or hidden feature claim',
        model_order=order, contrasts='later-listed model minus earlier-listed model; four fixed contrasts per pair; global Holm family retains untestable p=1',
        normalization='same per-seed mixed energy (a²+b²)/2 for EVERY pure/mixed MSE', warnings=['positive composition interaction alone is not a ranking flip',
            'phase unidentifiable below relative amplitude floor; entire seed phase mean then untestable',
            'complete-case phase conditioning: any unidentifiable matched cell excludes its whole seed bundle for that contrast; model-dependent exclusions can change the estimand; undefined is not absence',
            'per-model phase intervals use each model\'s identifiable seeds; subtracting those marginal summaries is not a paired comparison; model-pair phase effects use only jointly identifiable bundles',
            'zero-amplitude error decomposition uses amplitude-only convention, not evidence of correct phase',
            'mean signflip assumes symmetry/exchangeability and is not distribution-free under arbitrary skew; sign probability tests a different estimand',
            'sign probability does not establish mean-MSE superiority or practical relevance; mean and majority sign may point in opposite directions',
            'only the predeclared selected family controls primary statistical status; no secondary-endpoint rescue or generic model-winner label',
            'coupling can arise from additive frequency mixing; this is not proof of hidden nonlinear feature interaction',
            'linear representation probes and hidden selectivity require independent validation'])
    if design.native_background:
        report['warnings'].append('native background subtraction uses an actual forecast, but the pure-component input decomposition is still known by the experiment; it is not a generic deployment rule')
    else:
        report['warnings'].append('oracle offset subtraction is not f(offset) and can worsen constant forecast bias')
    if not resolution['can_reject_at_0_05']:
        report['warnings'].append('selected primary family cannot reject at alpha 0.05 at its attainable resolution; results are descriptive (or no model-pair family exists)')
    return (report, dict(seeds=seeds, **seed_values))

def render_interference_report(report, path, protocol, provenance, contexts, components, forecasts, metadata):
    """Render saved measurements and fixed first-seed waveforms as offline HTML."""
    from html import escape
    from pathlib import Path
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def table(headers, rows):
        """Escape artifact-derived labels and numerical audit values."""

        def value(x):
            """Use compact numbers without concealing undefined measurements."""
            if x is None:
                return 'undefined'
            return f'{x:.9g}' if isinstance(x, float) else str(x)
        return '<table><tr>' + ''.join(('<th>' + escape(h) + '</th>' for h in headers)) + '</tr>' + ''.join(('<tr>' + ''.join(('<td>' + escape(value(v)) + '</td>' for v in row)) + '</tr>' for row in rows)) + '</table>'
    selected_estimand = report.get('primary_estimand', 'mean')
    selected = report.get('selected_primary', report['primary'])
    primary_rows = []
    endpoint_rows = []
    for name, result in selected.items():
        signs = report.get('sign_comparisons', {}).get(name, {})
        disagreement = signs.get('mean_vs_majority_disagreement')
        primary_rows.append([
            name, result.get('value'), result.get('lo'), result.get('hi'),
            result['p_holm'], result.get('statistical_status', 'not recorded'),
            signs.get('descriptive_signed_mean'), signs.get('descriptive_signed_median'),
            signs.get('value'), 'opposite directions' if disagreement else
            'not defined' if disagreement is None else 'no opposition',
            signs.get('positive'), signs.get('negative'), signs.get('ties'),
            result.get('excluded_seeds', 0),
        ])
    for estimand, family in (
        ('mean', report.get('mean_comparisons', report['primary'])),
        ('sign_probability', report.get('sign_comparisons', {})),
    ):
        for name, result in family.items():
            endpoint_rows.append([
                estimand, 'primary' if estimand == selected_estimand else 'secondary',
                name, result.get('value'), result.get('lo'), result.get('hi'),
                result['p'], result['p_holm'],
            ])
    model_rows = []
    coupling_rows = []
    component_rows = []
    background_rows = []
    for name, result in report['models'].items():
        for condition, interval in result['scaled_mse'].items():
            model_rows.append([name, condition, interval.get('value', 'undefined'),
                    interval.get('lo', 'undefined'), interval.get('hi', 'undefined')])
        for target, metrics in result['changed_target'].items():
            for metric, interval in metrics.items():
                coupling_rows.append([name, target, metric, interval.get('value', 'undefined'),
                        interval.get('lo', 'undefined'), interval.get('hi', 'undefined'), interval.get('excluded_seeds',
                            0)])
        component_summary = result.get('component_sum', result.get('oracle_component_sum', {}))
        for metric, interval in component_summary.items():
            if isinstance(interval, dict):
                component_rows.append([name, metric, interval.get('value', 'undefined'),
                        interval.get('lo', 'undefined'), interval.get('hi', 'undefined')])
        for metric, interval in result.get('native_background_forecast', {}).items():
            background_rows.append([
                name, metric, interval.get('value', 'undefined'),
                interval.get('lo', 'undefined'), interval.get('hi', 'undefined'),
            ])
    design = report['design']
    first = min((m['seed'] for m in metadata))
    exemplars = [i for i, m in enumerate(metadata) if m['seed'] == first and m['condition'] == 'mixed']
    figure = make_subplots(rows=2, cols=2, subplot_titles=[f"seed {metadata[i]['generator_seed']}, φ cell {metadata[i]['cell_phi']}, ψ cell {metadata[i]['cell_psi']}" for i in exemplars])
    context = design['context']
    horizon = design['horizon']
    start = max(0, context - 2 * max(design['periods']))
    future_time = np.arange(context, context + horizon)
    palette = ['#2563eb', '#ea580c', '#16a34a', '#9333ea', '#0891b2', '#be123c']
    model_colors = {name: palette[i % len(palette)] for i, name in enumerate(report['model_order'])}
    for slot, index in enumerate(exemplars):
        row = slot // 2 + 1
        column = slot % 2 + 1
        show = slot == 0
        figure.add_trace(go.Scatter(x=np.arange(start, context), y=np.asarray(contexts)[index,
                    start:], name='observed context', legendgroup='context', showlegend=show, line=dict(color='#999999')),
            row=row, col=column)
        truth = np.asarray(components)[index].sum(0) + metadata[index]['offset']
        figure.add_trace(go.Scatter(x=future_time, y=truth, name='true mixed future',
                legendgroup='truth', showlegend=show, line=dict(color='black', width=3)), row=row,
            col=column)
        for name in report['model_order']:
            figure.add_trace(go.Scatter(x=future_time, y=forecasts[name]['prediction'][index],
                    name=name, legendgroup=name, showlegend=show, line=dict(color=model_colors[name])),
                row=row, col=column)
        figure.update_xaxes(title='Physical time', row=row, col=column)
    figure.update_layout(template='plotly_white', height=650, title='Fixed exemplars: all four mixed cells of the first generating seed')
    chart = figure.to_html(full_html=False, include_plotlyjs=True)
    html = '<!doctype html><html><head><meta charset="utf-8"><title>Controlled feature interference</title><style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:auto;padding:24px;line-height:1.5}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin:18px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f1f5f9}pre{white-space:pre-wrap;background:#f8fafc;padding:14px}.notice{background:#fffbeb;border-left:4px solid #d97706;padding:12px}</style></head><body>'
    html += '<h1>Controlled feature interference</h1><p class="notice">Behavioral controlled input intervention and predictive comparison. This audit does not establish internal causal mediation, hidden feature selectivity, uniqueness, or monosemanticity. Linear representation probes require separate validation.</p>'
    html += '<details><summary>Frozen design and identity provenance</summary><pre>' + escape(json.dumps(dict(protocol=protocol,
                provenance=provenance, n_seed_bundles=report['n_seed_bundles'], n_rows=report['n_rows']),
            indent=2)) + '</pre></details>'
    if 'producer_inference' in provenance:
        html += '<p class="notice">' + escape(provenance['producer_inference']['interpretation']) + ' Producer inference source hashes are retained separately from the later analysis source hashes above.</p>'
    elif provenance.get('prepared_source') is None:
        html += '<p class="notice">This saved-forecast analysis does not verify prospective registration timing. A newly chosen endpoint is an alternate-estimand reanalysis, not prospective confirmation.</p>'
    html += '<p>Changed target φ denotes period ' + str(design['periods'][0]) + '; its untouched component ψ has period ' + str(design['periods'][1]) + '. Changing ψ reverses those roles. Each independent generating seed supplies four mixed and two matched cells for each pure composition. Both periods tile context and horizon; the harmonic design is orthogonal and full rank.</p>'
    if design.get('native_background', False):
        html += '<p>The ninth cell is the same seed\'s constant-offset background, with exactly zero true component futures. Its actual native forecast is used for subtraction; all nine cells and their source corpus hash remain paired.</p>'
    else:
        html += '<p>This is the original eight-cell design. No background forecast was collected; the component-sum diagnostic subtracts the exact known offset.</p>'
    html += '<h2>Selected primary model-pair endpoint: ' + escape(selected_estimand) + '</h2><p>' + escape(report['contrasts']) + '. Each endpoint type has a separate complete Holm family over all ' + str(report['family_size']) + ' comparisons, retaining untestable p=1. Only the predeclared selected family controls primary statistical status. Secondary evidence never rescues it, and detected status is not a generic model-winner label.</p>'
    if selected_estimand == 'sign_probability':
        html += '<p class="notice">The selected endpoint is q=P(D&gt;0 | D≠0, jointly identifiable), tested against 0.5 by an exact two-sided binomial test. Its Clopper-Pearson interval is pointwise, not multiplicity-adjusted. This sign probability does not establish expected-MSE superiority or practical relevance. Signed mean and median remain visible because magnitudes and majority signs can disagree.</p>'
    else:
        html += '<p class="notice">The selected endpoint is the mean seed contrast. Bootstrap intervals are approximate and pointwise, conditional on fixed native predictions. The mean sign-flip test assumes symmetry or exchangeable paired labels; it is not a distribution-free mean-zero test under arbitrary skew. Exact sign comparisons below are secondary and do not replace the registered mean endpoint.</p>'
    html += '<p>These are statistical effect summaries, with no minimum practical threshold or discovery gate. Statistical significance alone does not establish a discovery or practical relevance.</p>'
    html += '<p>Here <code>model_by_composition = mixed model gap − 0.5 × (pure φ model gap + pure ψ model gap)</code>, where each model gap is later-listed model minus earlier-listed model. It is not a difference between native-minus-component-sum gains.</p>'
    html += '<h3>Selected endpoint inference resolution</h3><pre>' + escape(json.dumps(report['inference_resolution'],
            indent=2)) + '</pre>'
    if not report['inference_resolution']['can_reject_at_0_05']:
        html += '<p class="notice">The selected family cannot reject at alpha 0.05 at its attainable resolution. Treat this run as descriptive. Sign resolution depends on nonzero seed counts, not the permutation budget.</p>'
    html += table([
        'Explicit signed contrast', 'Selected estimate', 'Pointwise CI lower',
        'Pointwise CI upper', 'Selected p Holm', 'Selected statistical status',
        'Signed mean D', 'Signed median D', 'q positive/nonzero',
        'Mean versus majority', 'Positive', 'Negative', 'Ties', 'Excluded',
    ], primary_rows)
    html += '<h3>Both endpoint families, without secondary rescue</h3><p>Mean and exact sign tests answer different questions and have separate Holm corrections. The legacy one-sided sign_sensitivity JSON remains a secondary compatibility diagnostic; it is not the new exact two-sided endpoint.</p>'
    html += table([
        'Endpoint', 'Role', 'Comparison', 'Estimate', 'Pointwise CI lower',
        'Pointwise CI upper', 'Raw p', 'Own-family p Holm',
    ], endpoint_rows)
    html += '<details><summary>Resolution of each endpoint family</summary><pre>' + escape(json.dumps({
        'mean': report.get('mean_inference_resolution', {}),
        'sign_probability': report.get('sign_inference_resolution', {}),
    }, indent=2)) + '</pre></details>'
    paired_cases = {
        key: {field: value.get(field) for field in (
            'valid_seed_ids', 'excluded_seed_ids', 'paired_subset_model_means'
        )}
        for key, value in report['primary'].items() if 'untouched_phase_' in key
    }
    html += '<h3>Paired complete-case phase comparisons</h3><p>Per-model phase intervals below use each model\'s identifiable seeds. Subtracting those marginal summaries is not a paired comparison. Every model-pair phase effect uses only bundles where all four matched observations are identifiable for both models, and describes that selected subset. Model-dependent exclusions can change the estimand; undefined phase does not imply absent coupling.</p><pre>' + escape(json.dumps(paired_cases, indent=2)) + '</pre>'
    html += '<h2>Native forecast error by composition</h2><p>Every condition, including background when present, uses the same per-seed mixed energy (a²+b²)/2. Pure errors are not divided by their own component energy. Error differences are later-listed model minus earlier-listed model.</p>'
    html += table(['Model', 'Condition', 'Mixed-energy-scaled MSE', 'CI lower', 'CI upper'], model_rows)
    html += '<h2>Untouched and changed component responses</h2><p>Each seed averages both intervention directions and both untouched phase cells. Phase is undefined below ' + str(design['phase_floor']) + ' times its true amplitude; coefficient changes remain available. Response coefficient MSE is half the squared sin/cos coefficient change; its normalized response metric divides by the untouched true amplitude squared. Relative amplitude changes divide by that true amplitude. These response diagnostics differ from forecast MSE, which uses the common mixed-energy denominator. Intervals here are descriptive pointwise intervals.</p>'
    html += table(['Model', 'Changed target', 'Metric', 'Mean', 'CI lower', 'CI upper',
            'Excluded seeds'], coupling_rows)
    if design.get('native_background', False):
        html += '<h2>Native background component-sum diagnostic</h2><p class="notice">The sum is f(pure φ) + f(pure ψ) − f(background), using the actual native forecast of the matched constant background. It does not substitute the known offset for f(background). Pure-component inputs are still supplied by the controlled experiment, so this is a composition diagnostic, not a generic deployment improvement.</p>'
    else:
        html += '<h2>Oracle component-sum diagnostic</h2><p class="notice">f(pure φ) + f(pure ψ) − exact known offset is an oracle decomposition diagnostic. It is not f(offset), a generic forecasting improvement, or a deployable rule. Constant model bias is doubled by this sum and can make MSE four times worse.</p>'
    html += table(['Model', 'Diagnostic', 'Mean', 'CI lower', 'CI upper'], component_rows)
    if background_rows:
        html += '<h3>Native background forecast error and bias</h3><p>These are descriptive intervals for one actual constant-background forecast per generating seed. Raw error is in original units squared; bias is signed.</p>'
        html += table(['Model', 'Background metric', 'Mean', 'CI lower', 'CI upper'], background_rows)
    html += '<h2>Composition ranking and physical error channels</h2><p>A positive model-by-composition interaction alone does not establish a ranking flip. Opposite-signed mixed and pure gap intervals are reported separately. Coefficient, amplitude-only, phase-at-true-amplitude, signed amplitude×phase, offset and residual channels retain original units² and the common mixed-energy scale. The algebraic amplitude×phase term is distinct from cross-frequency input interference.</p><pre>' + escape(json.dumps(dict(pair_descriptive=report['pair_descriptive'],
                physical_error_decomposition={m: r['physical_error_decomposition'] for m, r in report['models'].items()}),
            indent=2)) + '</pre>'
    html += '<h2>Fixed waveform examples</h2>' + chart + '<p>Examples are the first seed in numerical order, all four mixed factorial cells; they were not selected by model outcome. Context and target future are shown on their physical time axis.</p>'
    html += '<h2>Interpretation limits</h2><ul>' + ''.join(('<li>' + escape(x) + '</li>' for x in report['warnings'])) + '</ul></body></html>'
    Path(path).write_text(html, encoding='utf-8')
