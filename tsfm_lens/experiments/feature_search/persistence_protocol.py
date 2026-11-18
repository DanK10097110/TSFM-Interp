"""Freeze a synthetic persistence pilot before any native inference.

Identical histories identify an expected-loss tradeoff, never future-regime
detection. Cued histories supply evidence before the forecast boundary. Each
generating seed is one source, kept intact across every split and contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.feature_search.selective_tree import decompose, continuation

REGIMES = ('persistent', 'amplitude_decay', 'frequency_drift', 'abrupt_cessation')
METHODS = ('native', 'tones_native_residual', 'tones_zero_residual', 'seasonal')
SEEDS = tuple(range(4951000, 4951008))
ENV = dict(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')


def sha(path):
    """Hash bytes without loading a checkpoint into memory."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    """Read UTF-8 JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Reject nonfinite JSON before writing any artifact."""
    text = json.dumps(value, indent=2, allow_nan=False)
    Path(path).write_text(text, encoding='utf-8')


def finite(value):
    """Reject nonfinite numeric arrays at artifact and prediction boundaries."""
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite array')
    return value


def exact(a, b, label):
    """Identity controls are exact, without tolerance-based escape hatches."""
    finite(a); finite(b)
    if not np.array_equal(a, b):
        raise ValueError('Exact identity failed: ' + label)


def signal(seed, regime, onset):
    """Change only the dominant tone; background, phase and noise stay paired.

    t=0 is the first future sample. Decay half-life is16; drift increases
    instantaneous frequency by50% per32 steps, continuously in phase. Abrupt
    cessation removes the dominant tone at onset. All parameters are fixed.
    """
    rng = np.random.default_rng(seed)
    phi, psi = rng.uniform(-np.pi, np.pi, 2)
    t = np.arange(-128, 32, dtype=float)
    elapsed = np.maximum(t-onset, 0)
    phase = 2*np.pi*t/19.2+phi
    amplitude = np.full(len(t), 8.)
    if regime == 'amplitude_decay':
        amplitude *= 2**(-elapsed/16)
    elif regime == 'frequency_drift':
        phase += 2*np.pi/19.2*.5*elapsed**2/(2*32)
    elif regime == 'abrupt_cessation':
        amplitude[t >= onset] = 0
    elif regime != 'persistent':
        raise ValueError(regime)
    background = .3+.5*np.sin(2*np.pi*17.35*t/128+psi)+rng.normal(0, .05, len(t))
    return finite(background+amplitude*np.sin(phase))


def corpus():
    """Make64 paired rows, verifying identical no-cue histories and source split."""
    histories, futures, rows = [], [], []
    for j, seed in enumerate(SEEDS):
        reference = signal(seed, 'persistent', 0)
        scale = float(np.var(reference[:128]))
        for arm, onset in [('identical', 0), ('cued', -32)]:
            for regime in REGIMES:
                value = signal(seed, regime, onset)
                if arm == 'identical' or regime == 'persistent':
                    exact(value[:128], reference[:128], 'counterfactual history')
                else:
                    exact(value[:96], reference[:96], 'pre-cue history')
                    if np.array_equal(value[96:128], reference[96:128]):
                        raise ValueError('Cue absent')
                rows.append(dict(row=len(rows), seed=seed, source=f'synthetic:{seed}',
                                 split='fit' if j < 4 else 'test', arm=arm, regime=regime,
                                 scale=scale))
                histories.append(value[:128]); futures.append(value[128:])
    return np.asarray(histories, np.float32), np.asarray(futures), rows


def plan(history):
    """Freeze depth1 extraction and a history-estimated seasonal-naive baseline.

    Selective thresholds inherit selective_tree unchanged. With no accepted
    tone, hybrid equals native and tones-only is exactly zero (including DC).
    Seasonal lag minimizes mean squared lagged difference over lags4..64;
    ties use the smaller lag. No generator frequency enters either method.
    """
    history = finite(history).astype(float)
    levels, detail = decompose(history, depth=1)
    tone = continuation(detail, len(detail))
    residual = levels[-1][-1]
    np.testing.assert_allclose(history-residual, levels[-1][0] if detail else 0,
                               atol=1e-12, rtol=1e-12)
    lags = np.arange(4, 65)
    losses = [np.mean((history[lag:]-history[:-lag])**2) for lag in lags]
    lag = int(lags[np.argmin(losses)])
    seasonal = history[-lag:][np.arange(32) % lag]
    return dict(accepted=len(detail), details=detail, lag=lag), residual, tone, seasonal


def source_files():
    """Record all experiment sources and reused native/helper sources."""
    here = Path(__file__).resolve().parent
    names = ['persistence_protocol.py', 'persistence_study.py', 'persistence_analysis.py',
             'persistence_audit.py', 'selective_tree.py', 'selective_study.py',
             'real_background_poc.py', 'homogeneous_normalizer.py', 'coherence_study.py']
    paths = [here/name for name in names]
    paths.append(here.parents[1]/'tsfm_lens/extraction/hooks.py')
    return {str(path): sha(path) for path in paths}


def prepare(out, prior):
    """Create fresh worker/evaluation trees and seal plans, targets and provenance."""
    out.mkdir(parents=True, exist_ok=False)
    inputs_dir = out/'inference'; inputs_dir.mkdir()
    evaluation = out/'evaluation'; evaluation.mkdir()
    x, y, rows = corpus()
    lookup, bank, residual, tones, seasonal, plans = {}, [], [], [], [], []
    for i, history in enumerate(x):
        key = history.tobytes()
        if key not in lookup:
            lookup[key] = len(bank)
            info, r, t, s = plan(history)
            bank.append(history); residual.append(r); tones.append(t); seasonal.append(s)
            plans.append(info)
        rows[i]['query'] = lookup[key]
    np.savez_compressed(inputs_dir/'inputs.npz', history=finite(bank).astype(np.float32),
                        residual=finite(residual).astype(np.float32), tones=finite(tones),
                        seasonal=finite(seasonal))
    np.savez_compressed(evaluation/'targets.npz', future=finite(y))
    write(inputs_dir/'plans.json', plans); write(evaluation/'rows.json', rows)
    prior_protocol = read(prior/'protocol.json')
    checkpoints = prior_protocol['checkpoints']
    for item in checkpoints.values():
        for name, digest in item['weights'].items():
            if sha(Path(item['path'])/name) != digest:
                raise ValueError('Cached checkpoint mismatch')
    worker = dict(version=1, checkpoints=checkpoints, sources=source_files(),
                  hashes={name: sha(inputs_dir/name) for name in ['inputs.npz', 'plans.json']},
                  max_seconds=600, max_rss_bytes=3*2**30, cpu_threads=1,
                  query_count=len(bank), layers='existing blocks(): first,quarter,half',
                  methods=METHODS, environment=ENV)
    write(inputs_dir/'protocol.json', worker)
    protocol = dict(version=1, evidence='exploratory behavioral pilot; no detection or causal claim',
                    seeds=SEEDS, source_split='4951000..4951003 FIT;4951004..4951007 TEST',
                    rows=len(rows), unique_histories=len(bank), context=128, horizon=32,
                    generator=dict(dominant_amplitude=8, dominant_period=19.2,
                                   weak_amplitude=.5, weak_frequency_per128=17.35,
                                   offset=.3, noise_sd=.05, decay_half_life=16,
                                   drift_fraction_per32=.5, cue_onset=-32),
                    regimes=REGIMES, methods=METHODS,
                    extraction='fixed depth1; inherited .65 stability/.15 energy/8 prominence; no tuning',
                    seasonal='past-only argmin mean squared lagged difference, lags4..64',
                    loss='MSE / variance of the same seed persistent history, shared across arms and worlds',
                    priors=[0., .25, .5, .75, 1.],
                    expected_loss='p persistent; (1-p)/3 each other future; hypothetical deployment priors',
                    ambiguity='identical histories imply identical forecasts; expected loss only, no regime detection',
                    cue_contrast='ignore-cue minus observed-cue loss on the SAME cued target; positive uses evidence beneficially',
                    inference='FIT and TEST separately; all8 descriptive; seed-level means, no significance claim or selector',
                    uncertainty='4096 paired source bootstrap draws; percentile95%; tiny-n descriptive only',
                    expansion='recommend after pilot; do not execute without coordination',
                    prior_protocol_sha256=sha(prior/'protocol.json'),
                    sealed={str(p.relative_to(out)): sha(p) for p in
                            [inputs_dir/'protocol.json', evaluation/'targets.npz', evaluation/'rows.json']})
    write(out/'protocol.json', protocol)
    write(out/'seal.json', dict(protocol_sha256=sha(out/'protocol.json')))
    print(json.dumps(dict(rows=len(rows), queries=len(bank), accepted=sum(p['accepted'] for p in plans))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prior', type=Path, default=Path('tsfm_lens/runs/selective_replication4942000'))
    args = parser.parse_args()
    prepare(args.out, args.prior)
