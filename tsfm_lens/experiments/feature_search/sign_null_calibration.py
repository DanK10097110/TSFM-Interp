"""Exact sign-null calibration with deliberately asymmetric, nonzero-mean data.

Independent fair signs establish q=.5. Sign-dependent exponential magnitudes
make the population mean -1, so this is not a mean-null calibration. Only the
actual production exact sign helper and global Holm are used for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import binomtest
from tsfm_lens.analysis.stats import holm

PROTOCOL_PATH = Path(__file__).with_name('sign_null_calibration_protocol.json')


def load_protocol():
    """Read the prospective exact-sign calibration before generating data."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def actual_sign_comparison():
    """Import production only after its owner declares the implementation stable."""
    from tsfm_lens.analysis.feature_interference import sign_comparison
    return sign_comparison


def run_family(differences):
    """Use finite 1-D seed columns and keep the complete exact-sign Holm4 family."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or differences.shape[1] != 4 or not np.isfinite(differences).all():
        raise ValueError('Need finite seed rows and four hypothesis columns')
    helper = actual_sign_comparison()
    effects = {f'h{j}': helper(differences[:, j]) for j in range(4)}
    adjusted = holm({name: effect['p'] for name, effect in effects.items()})
    for name, effect in effects.items():
        effect['p_holm'] = adjusted[name]
    return dict(effects=effects,
                raw_rejections=[bool(effects[f'h{j}']['p'] < .05) for j in range(4)],
                adjusted_rejections=[bool(adjusted[f'h{j}'] < .05) for j in range(4)])


def controls():
    """Check two-sided reversal, conditional ties and a raw-only Holm decoy."""
    helper = actual_sign_comparison()
    positive, negative = helper(np.ones(8)), helper(-np.ones(8))
    if positive['p'] != 2 / 256 or negative['p'] != positive['p']:
        raise AssertionError('Exact two-sided direction reversal known answer failed')
    if positive['value'] != 1 or negative['value'] != 0:
        raise AssertionError('Direction reversal did not complement q')
    ties = helper(np.array([1., 1., -1., -1., 0., 0., 0., 0., np.nan]))
    if ties['n_nonzero'] != 4 or ties['ties'] != 4 or ties['excluded_seeds'] != 1 or ties['value'] != .5:
        raise AssertionError('Conditional nonzero and missing-seed denominators failed')
    all_ties = helper(np.zeros(64))
    if all_ties['p'] != 1 or all_ties.get('value') is not None:
        raise AssertionError('All-tied data must retain p=1 with undefined q')
    moderate = np.zeros((16, 4))
    moderate[:, 0] = np.r_[np.ones(13), -3 * np.ones(3)]
    family = run_family(moderate)
    if not family['raw_rejections'][0] or any(family['adjusted_rejections']):
        raise AssertionError('Complete Holm4 correction failed the raw-only sign decoy')
    return dict(positive=positive, negative=negative, ties=ties, all_ties=all_ties, Holm_decoy=family)


def generate_family(protocol, replicate, phase):
    """Correlate exponential magnitudes while keeping all seed/hypothesis signs fair."""
    sequence = np.random.SeedSequence([protocol['master_seed'], replicate, phase])
    magnitude_sequence, sign_sequence = sequence.spawn(2)
    magnitude_rng = np.random.default_rng(magnitude_sequence)
    sign_rng = np.random.default_rng(sign_sequence)
    n, m = protocol['n_seed_bundles'], protocol['hypotheses_per_family']
    gaussian = (magnitude_rng.normal(size=(n, 1)) + magnitude_rng.normal(size=(n, m))) / np.sqrt(2.)
    uniform = np.clip(ndtr(gaussian), np.finfo(float).tiny, 1 - np.finfo(float).eps)
    exponential = -np.log1p(-uniform)
    positive = sign_rng.integers(0, 2, size=(n, m)).astype(bool)
    return np.where(positive, exponential, -3 * exponential)


def binomial_interval(count, total):
    """Treat independent replicate families, not correlated endpoints, as trials."""
    if total == 0:
        return dict(count=0, total=0, untestable='no completed families')
    interval = binomtest(count, total).proportion_ci(confidence_level=.95, method='exact')
    return dict(count=int(count), total=total, rate=float(count / total),
                lo=float(interval.low), hi=float(interval.high), interval='95% exact Clopper-Pearson')


def summarize(records):
    """Describe valid sign-null rejection frequencies and the deliberately negative mean."""
    if not records:
        return dict(completed_replicates=0)
    adjusted = np.array([row['adjusted_rejections'] for row in records])
    raw = np.array([row['raw_rejections'] for row in records])
    total = len(records)
    return dict(completed_replicates=total,
        Holm_family_rejection=binomial_interval(int(adjusted.any(axis=1).sum()), total),
        raw_family_rejection=binomial_interval(int(raw.any(axis=1).sum()), total),
        per_hypothesis={f'h{j}': dict(Holm_rejection=binomial_interval(int(adjusted[:, j].sum()), total),
            raw_rejection=binomial_interval(int(raw[:, j].sum()), total),
            observed_mean=float(np.mean([row['sample_means'][j] for row in records])),
            observed_positive_fraction=float(np.mean([row['effects'][f'h{j}']['value'] for row in records]))) for j in range(4)})


def run(out):
    """Run a short independent pilot and at most two CPU minutes of exact tests."""
    protocol = load_protocol()
    out.mkdir(parents=True, exist_ok=False)
    raw_protocol = PROTOCOL_PATH.read_bytes()
    (out / 'protocol.json').write_bytes(raw_protocol)
    helper = actual_sign_comparison()
    source = inspect.getsource(helper)
    (out / 'production_sign_function_snapshot.py').write_text(source, encoding='utf-8')
    provenance = dict(protocol_sha256=hashlib.sha256(raw_protocol).hexdigest(),
        helper_module=helper.__module__, helper_source_path=inspect.getsourcefile(helper),
        helper_source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
        helper_compiled_sha256=hashlib.sha256(marshal.dumps(helper.__code__)).hexdigest(),
        worker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution='actual imported production helper; snapshot retained for provenance')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    started = time.perf_counter()
    known = controls()
    (out / 'known_controls.json').write_text(json.dumps(known, indent=2), encoding='utf-8')
    pilot_seconds = []
    for replicate in range(protocol['pilot_replicates']):
        data = generate_family(protocol, replicate, 0)
        block = time.perf_counter()
        run_family(data)
        pilot_seconds.append(time.perf_counter() - block)
    projected = max(pilot_seconds) * protocol['main_replicates']
    (out / 'pilot.json').write_text(json.dumps(dict(family_seconds=pilot_seconds,
        projected_main_seconds=projected, main_replicates=protocol['main_replicates']), indent=2), encoding='utf-8')
    print(json.dumps(dict(projected_main_seconds=projected, main_replicates=128)), flush=True)
    if projected > 100:
        (out / 'paused.json').write_text(json.dumps(dict(reason='pilot exceeds100-second main budget')), encoding='utf-8')
        return
    records = []
    for replicate in range(protocol['main_replicates']):
        if time.perf_counter() - started > protocol['runtime']['cap_seconds']:
            break
        data = generate_family(protocol, replicate, 1)
        result = run_family(data)
        records.append(dict(replicate=replicate, seed_sequence=[protocol['master_seed'], replicate, 1],
            sample_means=data.mean(axis=0).tolist(), **result))
    summary = summarize(records)
    report = dict(summary=summary, elapsed_seconds=time.perf_counter() - started,
        complete=len(records) == protocol['main_replicates'], population_positive_probability=.5,
        population_mean=-1., population_median=0.,
        scope='Valid exact sign null with asymmetric sign-dependent magnitudes;NOT a mean-zero null',
        caution='Coarse128-family check,not precise5%certification;does not remove phase complete-case selection or establish expected-error superiority')
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'replicate_results.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.out)
