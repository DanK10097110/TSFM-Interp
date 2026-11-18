"""Prospective64-source persistence replication; preparation never runs a model.

Preserves the pilot's signal family and calibrated singleton convention. Five
cue lead times and two fixed periods test a past-only veto and a fixed recent
refit without tuning on TEST. Each seed remains one dependent source cluster.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from experiments.feature_search.persistence_protocol import ENV, REGIMES, exact, finite, read, sha, write
from experiments.feature_search.selective_tree import candidate, continuation, decompose

SEEDS = tuple(range(4953000, 4953064))
PERIODS = (19.2, 24.)
LEADS = (0, 8, 16, 32, 64)
THRESHOLDS = (-1., 0., .1, .25, .5, 1., 2., 4., None)
METHODS = ('native', 'hybrid', 'tones_zero', 'seasonal', 'recent_hybrid', 'recent_zero')


def signal(seed, period, regime, lead):
    """Same paired noise/phase as pilot; dominant tone changes at t=-lead."""
    rng = np.random.default_rng(seed)
    phi, psi = rng.uniform(-np.pi, np.pi, 2)
    t = np.arange(-128, 32, dtype=float)
    elapsed = np.maximum(t+lead, 0)
    phase = 2*np.pi*t/period+phi
    amplitude = np.full(len(t), 8.)
    if regime == 'amplitude_decay':
        amplitude *= 2**(-elapsed/16)
    elif regime == 'frequency_drift':
        phase += 2*np.pi/period*.5*elapsed**2/(2*32)
    elif regime == 'abrupt_cessation':
        amplitude[t >= -lead] = 0
    elif regime != 'persistent':
        raise ValueError(regime)
    return finite(.3+.5*np.sin(2*np.pi*17.35*t/128+psi)+rng.normal(0, .05, len(t))+
                  amplitude*np.sin(phase))


def split_for(seed):
    """Keep all dependent conditions of a generating seed in one partition."""
    if seed not in SEEDS:
        raise ValueError('Unregistered seed')
    return 'fit' if seed < 4953032 else 'test'


def corpus(seeds=SEEDS):
    """Generate2,560 rows with exact hidden-future and pre-cue identity gates."""
    histories, targets, rows = [], [], []
    for seed in seeds:
        for period in PERIODS:
            reference = signal(seed, period, 'persistent', 0)
            for lead in LEADS:
                for regime in REGIMES:
                    value = signal(seed, period, regime, lead)
                    prefix = 128-lead
                    exact(value[:prefix], reference[:prefix], 'unchanged pre-cue prefix')
                    if lead == 0 or regime == 'persistent':
                        exact(value[:128], reference[:128], 'indistinguishable history')
                    elif np.array_equal(value[:128], reference[:128]):
                        raise ValueError('Cue absent')
                    rows.append(dict(row=len(rows), seed=seed, source=f'synthetic:{seed}',
                                     split=split_for(seed), period=period, lead=lead, regime=regime,
                                     scale=float(np.var(reference[:128]))))
                    histories.append(value[:128]); targets.append(value[128:])
    return np.asarray(histories, np.float32), finite(targets), rows


def waveform(detail, t):
    """Evaluate one fitted tone at the shared physical origin, preserving DC outside it."""
    a, b = detail['coefficients']
    angle = 2*np.pi*detail['frequency']*np.asarray(t)/128
    return a*np.sin(angle)+b*np.cos(angle)


def recent_error(history, detail):
    """Past-only coefficient change: latest32 versus previous32 and whole history.

    Fits use the accepted whole-history frequency and absolute t, including an
    intercept. Error is normalized by whole-history tone amplitude. It exposes
    amplitude and phase changes without using regime, lead, period or targets.
    """
    t = np.arange(-128, 0)
    angle = 2*np.pi*detail['frequency']*t/128
    basis = np.column_stack([np.ones(128), np.sin(angle), np.cos(angle)])
    earlier = np.linalg.lstsq(basis[-64:-32], history[-64:-32], rcond=None)[0][1:]
    recent = np.linalg.lstsq(basis[-32:], history[-32:], rcond=None)[0][1:]
    full = np.asarray(detail['coefficients'])
    return float(max(np.linalg.norm(recent-full), np.linalg.norm(recent-earlier))/
                 max(np.linalg.norm(full), 1e-12))


def make_plan(history):
    """Full-depth1 continuation and recent32 refit conditional on the SAME acceptance.

    Recent frequency/amplitude/phase are re-estimated with the existing continuous
    candidate fitter, without applying a new recent stability threshold. Its tone
    is projected over all128 observed steps, then subtracted to form an intact
    native residual. Zero-residual controls isolate the continuation fit itself.
    """
    x = finite(history).astype(float)
    levels, details = decompose(x, depth=1)
    tone = continuation(details, len(details))
    residual = levels[-1][-1]
    recent_tone, recent_residual, recent_detail, error = np.zeros(32), x.copy(), None, 0.
    if details:
        _, recent_detail = candidate(x[-32:])
        recent_tone = waveform(recent_detail, np.arange(32))
        recent_residual = x-waveform(recent_detail, np.arange(-128, 0))
        error = recent_error(x, details[0])
    lags = np.arange(4, 65)
    lag = int(lags[np.argmin([np.mean((x[k:]-x[:-k])**2) for k in lags])])
    info = dict(accepted=len(details), full=details, recent=recent_detail,
                recent_error=error, lag=lag)
    return info, residual, recent_residual, tone, recent_tone, x[-lag:][np.arange(32) % lag]


def use_hybrid(error, accepted, threshold):
    """A veto never adds an extraction; -1 always native, None always accepts eligible."""
    return bool(accepted and (threshold is None or error <= threshold))


def binding(out):
    """Validate only inference manifests, source and inputs; do not read targets."""
    p = read(out/'inference/protocol.json')
    if sha(out/'inference/protocol.json') != read(out/'inference/seal.json')['sha256']:
        raise ValueError('Inference protocol changed')
    for name, digest in p['sources'].items():
        if sha(name) != digest:
            raise ValueError('Frozen source changed: '+name)
    for name, digest in p['hashes'].items():
        if sha(out/'inference'/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    return p


def prepare(out, pilot):
    """Freeze all data, instrument sentinels, hypotheses and selection rules."""
    out.mkdir(parents=True, exist_ok=False)
    (out/'inference').mkdir()
    for split in ['fit', 'test']:
        (out/'evaluation'/split).mkdir(parents=True)
    x, targets, rows = corpus()
    history_ids, histories, bank_ids, bank, plans, tones, recent, seasonal = {}, [], {}, [], [], [], [], []
    def add_bank(value):
        arr = finite(value).astype(np.float32)
        key = arr.tobytes()
        if key not in bank_ids:
            bank_ids[key] = len(bank); bank.append(arr)
        return bank_ids[key]
    for i, history in enumerate(x):
        key = history.tobytes()
        if key not in history_ids:
            history_ids[key] = len(histories); histories.append(history)
            info, r, rr, tone, rt, s = make_plan(history)
            info.update(native=add_bank(history), residual=add_bank(r), recent_residual=add_bank(rr))
            plans.append(info); tones.append(tone); recent.append(rt); seasonal.append(s)
        rows[i]['query'] = history_ids[key]
    np.savez_compressed(out/'inference/inputs.npz', bank=finite(bank), tones=finite(tones),
                        recent_tones=finite(recent), seasonal=finite(seasonal))
    write(out/'inference/plans.json', plans)
    for split in ['fit', 'test']:
        selected = [r for r in rows if r['split'] == split]
        write(out/'evaluation'/split/'rows.json', selected)
        np.savez_compressed(out/'evaluation'/split/'targets.npz', future=targets[[r['row'] for r in selected]])
    pilot_worker = read(pilot/'inference/protocol.json')
    checkpoints = pilot_worker['checkpoints']
    for checkpoint in checkpoints.values():
        for name, digest in checkpoint['weights'].items():
            if sha(Path(checkpoint['path'])/name) != digest:
                raise ValueError('Cached weight mismatch')
    here = Path(__file__).resolve().parent
    sources = dict(pilot_worker['sources'])
    files = sorted(here.glob('persistence_replication*.py'))
    files += sorted((here.parents[1]/'tests').glob('test_persistence_replication*.py'))
    sources.update({str(p): sha(p) for p in files})
    sentinel_rows = [next(r for r in rows if r['seed'] == seed and r['period'] == period and
                         r['lead'] == lead and r['regime'] == regime)
                     for seed in [4953000, 4953032] for period in PERIODS
                     for lead, regime in [(0, 'persistent'), (8, 'amplitude_decay'),
                                          (16, 'frequency_drift'), (32, 'abrupt_cessation'),
                                          (64, 'frequency_drift')]]
    sentinels = sorted({r['query'] for r in sentinel_rows})
    worker = dict(checkpoints=checkpoints, sources=sources,
                  hashes={f: sha(out/'inference'/f) for f in ['inputs.npz', 'plans.json']},
                  environment=ENV, max_seconds=900, max_rss_bytes=3*2**30,
                  cpu_threads=1, batch_size=1, queries=len(histories), native_inputs=len(bank),
                  sentinels=sentinels, methods=METHODS,
                  gates='Every input finite/shape;20 preregistered sentinel histories: exact repeat, capture/plain, three self patches, duplicate singleton calls; both residual repeats. Every abstention exact.',
                  hidden='FP32 first/quarter/half full states on20 sentinels only; no cross-model alignment claim')
    write(out/'inference/protocol.json', worker)
    write(out/'inference/seal.json', dict(sha256=sha(out/'inference/protocol.json')))
    protocol = dict(version=1, stage='fresh replication', seeds=SEEDS, periods=PERIODS, leads=LEADS,
                    regimes=REGIMES, context=128, horizon=32, rows=len(rows), source_split='FIT4953000..31;TEST4953032..63',
                    generator='pilot exact family, dominant amplitude8, weak .5 at17.35/128,offset.3,noise.05,decay half-life16,drift+.5 frequency/32 steps',
                    methods=METHODS, refit='last32 only; unthresholded candidate conditional on same full depth1 acceptance; no TEST tuning; residual=x-refitted tone over128',
                    veto=dict(feature='max(norm(latest32-full),norm(latest32-previous32))/norm(full coefficients), fixed full frequency, intercept, absolute phase',
                              thresholds=THRESHOLDS, fit='Per model minimize mean normalized loss on all FIT seed/cell means; one threshold shared across both periods and all leads/regimes; deterministic ties first grid candidate, native first',
                              test='Freeze selection.json and its hash before opening TEST targets; no TEST feature/outcome tuning'),
                    loss='MSE / persistent-history variance shared by seed and period across all conditions',
                    expected_priors=[0., .25, .5, .75, 1.],
                    primary_hypotheses=['H1_Bolt/H1_TimesFM: FIT-only veto improves TEST loss over always native',
                                        'H2_Bolt/H2_TimesFM: FIT-only veto improves TEST loss over always hybrid',
                                        'H3_19.2/H3_24: recent-zero improves full-tones-zero on visible changing futures (lead>0,regime!=persistent)'],
                    primary_scope='Six one-sided sign tests on32 TEST seed-level paired gains; Holm family6; require mean gain>.01 AND adjusted p<=.05. No window/condition pseudoreplication. Native/selector endpoints average all40 balanced cells per seed.',
                    diagnostics=['Risk curves per period on lead0 only: no detection claim',
                                 'Cue gain compares lead0 and cued predictions on the SAME cued future',
                                 'Recent_hybrid versus hybrid and recent_zero versus tones_zero separates continuation fit from native residual effects',
                                 'Veto acceptance/benefit per lead/regime/period, reported without TEST threshold selection',
                                 'Fixed future-mixture probabilities are hypothetical deployment weights, not estimates of pretraining priors'],
                    uncertainty='4096 paired source bootstrap percentile95%, seed4953999; intervals descriptive, primary inference via6-test Holm family',
                    execution='HOLD until explicit coordinator slot grant; at most1 worker for this track;<=900s/model,CPU1,3GiB,cached/offline,kill on cap; partial artifacts invalid, no subset scoring',
                    pilot_protocol_sha256=sha(pilot/'protocol.json'),
                    sealed={str(p.relative_to(out)): sha(p) for p in
                            [out/'inference/protocol.json', out/'inference/seal.json']+
                            [out/'evaluation'/s/f for s in ['fit', 'test'] for f in ['rows.json', 'targets.npz']]})
    write(out/'protocol.json', protocol); write(out/'seal.json', dict(sha256=sha(out/'protocol.json')))
    write(out/'launch_hold.json', dict(status='awaiting explicit coordinator slot grant', native_launched=False))
    write(out/'knowledge.json', dict(question='Can visible-history stability predict when selective continuation helps, and does recent refitting remedy old-history oversmoothing?',
          design=protocol, exact_findings=dict(status='prepared; no native inference or behavioral findings',rows=len(rows),unique_histories=len(histories),native_inputs=len(bank),sentinels=len(sentinels)),
          controls=worker['gates'], limitations=['Prospective synthetic replication; no results yet','Recent refit conditional on full-history acceptance; does not test salvaging abstained cases'],
          evidence_class='prospective protocol and preparation audit',next_decision='Complete tests/plants; HOLD native launch until explicit coordinator slot grant'))
    print(dict(rows=len(rows), histories=len(histories), native_inputs=len(bank), sentinels=len(sentinels)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('tsfm_lens/runs/persistence_pilot4951000/singleton_geometry'))
    args = parser.parse_args(); prepare(args.out, args.pilot)
