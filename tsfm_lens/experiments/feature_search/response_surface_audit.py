"""Independent saved-arithmetic replay and compact descriptive orchestration handoff.

This post-inference audit does not modify the frozen native or analysis sources.
It checks every reported metric against raw arrays before summarizing contrasts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha


def audit(out):
    """Replay all source-level metrics and expose shrinkage without inferring a population."""
    inputs = np.load(out/'inputs.npy', mmap_mode='r')
    responses = read(out/'response_rows.json'); locals_ = read(out/'local_rows.json')
    pairs = read(out/'interaction_rows.json'); rows = read(out/'rows.json')
    arrays = {}; max_error = 0.; checks = 0

    def values(row):
        """Read raw endpoint by its model, policy and cut, with explicit promotion."""
        cut = row.get('cut', 'forecast'); key = (row['model'], row['policy'], cut)
        if key not in arrays:
            suffix = 'forecast' if cut == 'forecast' else f'hidden_{cut}'
            arrays[key] = np.load(out/f'{key[0]}_{key[1]}_{suffix}.npy', mmap_mode='r')
        return arrays[key]

    def norm(x):
        """Independent RMS formula."""
        return float(np.linalg.norm(np.ravel(x))/np.sqrt(np.size(x)))

    def check(actual, expected):
        """Check serialization and arithmetic with FP64-relative tolerance."""
        nonlocal checks, max_error
        if actual is None or expected is None:
            assert actual is expected
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
            max_error = max(max_error, abs(actual-expected))
        checks += 1

    for r in responses:
        value = values(r); effect = value[r['index']].astype(float)-value[r['base']].astype(float)
        y = r['amplitude']*r['scale']*np.sin(2*np.pi*r['frequency']*np.arange(32)/128+r['phase'])
        check(r['nmse'], float(np.dot(effect-y, effect-y)/np.dot(y, y)))
        check(r['gain'], float(np.dot(effect, y)/np.dot(y, y)))
    for r in locals_:
        value = values(r)
        minus, zero, plus, mid = [value[r[k]].astype(float) for k in ('minus', 'zero', 'plus', 'midpoint')]
        xm, xp = inputs[r['minus']].astype(float), inputs[r['plus']].astype(float)
        dx = norm(xp-xm)/(2*r['scale'])
        chord = norm(plus+minus-2*mid); effects = norm(plus-mid)+norm(minus-mid)
        check(r['input_radius'], dx)
        check(r['distance_sensitivity'], norm(plus-minus)/(2*dx))
        check(r['chord_relative'], chord/effects if effects else None)
        check(r['parameter_second_rms'], norm(plus+minus-2*zero)/r['step']**2)
    for r in pairs:
        value = values(r); base, a, b, ab = [value[r[k]].astype(float) for k in ('base', 'a', 'b', 'ab')]
        mixed = norm(ab-a-b+base); ea, eb = norm(a-base), norm(b-base)
        check(r['interaction_rms'], mixed)
        check(r['relative_sum'], mixed/(ea+eb) if ea+eb else None)
    for model in ('bolt', 'timesfm'):
        for policy in ('native', 'locked'):
            forecast = arrays[(model, policy, 'forecast')]
            saved = np.load(out/f'{model}_{policy}_arithmetic.npz')
            for i, r in enumerate(read(out/'paths.json')):
                expected = forecast[r['plus']].astype(float)+forecast[r['minus']].astype(float)-2*forecast[r['midpoint']].astype(float)
                np.testing.assert_array_equal(saved['local_chord_second'][i], expected); checks += 1
            for i, r in enumerate(read(out/'interactions.json')):
                expected = forecast[r['ab']].astype(float)-forecast[r['a']].astype(float)-forecast[r['b']].astype(float)+forecast[r['base']].astype(float)
                np.testing.assert_array_equal(saved['interaction'][i], expected); checks += 1
    shrinkage = []
    group = {}
    for r in locals_:
        key = tuple(r[k] for k in ('model', 'policy', 'cut', 'axis', 'source', 'center', 'amplitude'))
        group.setdefault(key, []).append(r)
    for key, items in group.items():
        small, large = sorted(items, key=lambda r:r['step'])[0], sorted(items, key=lambda r:r['step'])[-1]
        shrinkage.append(dict(zip(('model', 'policy', 'cut', 'axis', 'source', 'center', 'amplitude'), key),
            small_radius=small['input_radius'], large_radius=large['input_radius'],
            small_curvature=small['chord_relative'], large_curvature=large['chord_relative'],
            curvature_shrinks=small['chord_relative'] < large['chord_relative'],
            sensitivity_ratio=small['distance_sensitivity']/large['distance_sensitivity']))
    compact = []
    for key in sorted(set(tuple(r[k] for k in ('model', 'policy', 'cut', 'axis')) for r in shrinkage)):
        subset = [r for r in shrinkage if tuple(r[k] for k in ('model', 'policy', 'cut', 'axis')) == key]
        compact.append(dict(zip(('model', 'policy', 'cut', 'axis'), key),
            dependent_paths=len(subset), sources=len(set(r['source'] for r in subset)),
            shrink_count=sum(r['curvature_shrinks'] for r in subset),
            small_curvature_median=float(np.median([r['small_curvature'] for r in subset])),
            large_curvature_median=float(np.median([r['large_curvature'] for r in subset])),
            sensitivity_ratio_median=float(np.median([r['sensitivity_ratio'] for r in subset])),
            small_curvature_max=float(max(r['small_curvature'] for r in subset))))
    centers = [r for r in responses if r['offset'] == 0]
    paired = []
    for r in centers:
        if r['policy'] != 'native': continue
        locked = next(s for s in centers if s['model'] == r['model'] and s['policy'] == 'locked' and s['index'] == r['index'])
        paired.append(dict(model=r['model'], source=r['source'], domain=r['domain'], period=128/r['center'],
            amplitude=r['amplitude'], native_nmse=r['nmse'], locked_nmse=locked['nmse'],
            locked_minus_native=locked['nmse']-r['nmse'], native_gain=r['gain'], locked_gain=locked['gain']))
    result = dict(audit=dict(checks=checks, maximum_absolute_replay_difference=max_error, passed=True,
                            audit_source_sha256=sha(Path(__file__))),
        source_count=8, source_units='Four independent synthetic seeds and one real FIT source per domain; repeated paths are dependent',
        raw_state_shapes={str(k):list(v.shape) for k,v in arrays.items()},
        curvature=compact, center_response_paired=paired,
        suggested_expansions=[
            'Cross the same phase grid with all backgrounds, then add fresh sources; current phase and context are confounded.',
            'Use three smaller measured input radii and half-step convergence at the worst source/cut. Compare raw path curvature with chord-corrected curvature and FP32 floors.',
            'Bridge the24h-to19.2h frequency gap continuously, matching RMS history distances across amplitudes, with pure-tone and balanced/dominant-background controls.',
            'Interpolate nearby backgrounds at fixed injected tone and phase to test smooth context dependence directly, then test local Jacobian transport.',
            'Separate background-locked from locally locked normalization; test whether high-amplitude off-manifold lock artifacts explain changes.',
            'Decode on fresh source-phase combinations against raw-input and random-initialization baselines before attributing accessibility to learned semantics.'
        ],
        stopping='Pilot only. No expansion or new native inference launched. Failures do not prove uninterpretable; smoothness does not establish semantic correctness.')
    write(out/'shrinkage_rows.json', shrinkage); write(out/'handoff.json', result)
    print(dict(checks=checks, maximum_absolute_replay_difference=max_error, passed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
