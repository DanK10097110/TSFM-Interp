"""Freeze the bounded weather pilot without native inference or selection.

Thirty complete fixed windows form eight dependent station groups. The existing
selective_tree is used unchanged; fitting its per-history tone is not training
or tuning the rule. Chronology establishes new observations, not new locations.
"""
from pathlib import Path
import argparse
import importlib.metadata
import numpy as np
from experiments.feature_search.persistence_protocol import ENV, exact, finite, read, sha, write
from experiments.feature_search.postrelease_weather_prepare import utcnow
from experiments.feature_search.selective_tree import decompose, continuation

METHODS = ('native', 'hybrid', 'seasonal24', 'tones_zero')
STATIONS = ('KSEA', 'KPHX', 'KDEN', 'KMSP', 'KDFW', 'KJFK', 'KATL', 'KMIA')


def plan_history(history):
    """Apply frozen depth1 defaults and verify additive reconstruction before cast."""
    x = finite(history).astype(np.float64)
    if x.shape != (128,):
        raise ValueError('Expected128 history hours')
    levels, details = decompose(x, depth=1)
    residual = finite(levels[-1][-1])
    reconstructed = np.sum(levels[-1], axis=0)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-12, atol=1e-12)
    tones = finite(continuation(details, len(details)))
    seasonal = x[-24:][np.arange(32) % 24]
    if not details:
        exact(residual, x, 'abstention input'); exact(tones, np.zeros(32), 'abstention tones')
    return residual, tones, seasonal, dict(accepted=bool(details), details=details,
            reconstruction_max_abs=float(np.max(abs(reconstructed-x))))


def binding(out):
    """Validate frozen inference inputs and sources without reading outcomes."""
    if sha(out/'protocol.json') != read(out/'seal.json')['sha256']:
        raise ValueError('Protocol seal changed')
    protocol = read(out/'protocol.json')
    if sha(out/'inference/protocol.json') != protocol['inference_sha256']:
        raise ValueError('Inference protocol changed')
    p = read(out/'inference/protocol.json')
    for path, digest in p['sources'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: '+path)
    for name, digest in p['hashes'].items():
        if sha(out/'inference'/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    return p


def prepare(out, weather):
    """Seal exact complete-window membership, recipes, accounting and analysis."""
    preflight = out.with_name(out.name+'_preflight')
    audit = read(preflight/'knowledge.json')['exact_findings']
    assert audit['tests_passed'] == 5 and audit['plants_caught'] == 11
    for name, digest in read(preflight/'manifest.json').items():
        assert sha(preflight/name) == digest
    prior = weather/'ghcnh_oct2026/prepared'
    seal = read(prior/'seal.json')
    for name, digest in seal['files'].items():
        if sha(prior/name) != digest:
            raise ValueError('Prepared data changed: '+name)
    oldrows = read(prior/'windows.json')
    selected = [i for i, r in enumerate(oldrows) if r['history_eligible'] and r['future_complete']]
    assert len(selected) == 30
    x0 = np.load(prior/'histories.npz')['history']
    future = np.load(prior/'targets.npz')
    assert future['observed'][selected].all()
    x = finite(x0[[oldrows[i]['input_index'] for i in selected]])
    y = finite(future['target'][selected])
    counts = [sum(oldrows[i]['icao'] == s for i in selected) for s in STATIONS]
    assert counts == [4,4,4,4,4,2,4,4]
    scale = np.var(x, axis=1)
    if not (scale > 0).all():
        raise ValueError('Zero history variance; do not silently choose a loss floor')
    out.mkdir(parents=True, exist_ok=False)
    (out/'inference').mkdir(); (out/'evaluation').mkdir()
    bank, bank_ids, plans, tones, seasonal, rows = [], {}, [], [], [], []
    def add(value):
        value = finite(value).astype(np.float32)
        finite(value)
        key = value.tobytes()
        if key not in bank_ids:
            bank_ids[key] = len(bank); bank.append(value)
        return bank_ids[key]
    for i, (history, oldindex) in enumerate(zip(x, selected)):
        residual, tone, season, info = plan_history(history)
        plans.append(dict(native=add(history), residual=add(residual), **info))
        tones.append(tone); seasonal.append(season)
        rows.append(dict(row=i, original_window=oldindex, station=oldrows[oldindex]['icao'],
                         origin=oldrows[oldindex]['origin'], variance=float(scale[i])))
    constant = np.full(128, 7., dtype=np.float64)
    residual, tone, _, info = plan_history(constant)
    assert not info['accepted']; exact(residual, constant, 'constant abstention')
    sentinel = add(constant)
    np.savez(out/'inference/inputs.npz', history=x, bank=finite(bank), tones=finite(tones),
             seasonal24=finite(seasonal), scales=scale, sentinel_tone=tone)
    write(out/'inference/plans.json', plans)
    write(out/'inference/rows.json', rows)
    np.savez(out/'evaluation/targets.npz', future=y)
    metadata = read(weather/'metadata_verification.json')
    checks = read(weather/'ghcnh_oct2026/protocol.json')['checkpoints']
    for model, cp in checks.items():
        assert metadata['models'][model]['weights_match'] and metadata['models'][model]['predates_observations']
        for name, digest in cp['weights'].items():
            assert sha(Path(cp['path'])/name) == digest
    here = Path(__file__).resolve().parent
    files = list(here.glob('postrelease_weather_*.py'))
    files += list((here.parents[1]/'tests').glob('test_postrelease_weather*.py'))
    files += [here/n for n in ['persistence_protocol.py', 'persistence_study.py', 'persistence_singleton.py',
                               'persistence_execution.py', 'selective_tree.py', 'real_background_poc.py',
                               'homogeneous_normalizer.py', 'coherence_study.py']]
    files += [here.parents[1]/'tsfm_lens/extraction/hooks.py']
    versions = {}
    for package in ['numpy','scipy','torch','transformers','chronos-forecasting','timesfm','safetensors']:
        distribution = importlib.metadata.distribution(package)
        versions[package] = distribution.version
        if package in ['chronos-forecasting','timesfm']:
            files += [Path(distribution.locate_file(f)) for f in distribution.files or [] if str(f).endswith('.py')]
        elif package == 'transformers':
            files += [Path(distribution.locate_file('transformers/models/t5/modeling_t5.py'))]
    for cp in checks.values():
        files += list(Path(cp['path']).glob('*.json'))
    inf = dict(checkpoints=checks, sources={str(f.resolve()):sha(f) for f in files},
               hashes={n:sha(out/'inference'/n) for n in ['inputs.npz','plans.json','rows.json']},
               environment=ENV, versions=versions, max_seconds=300, max_rss_bytes=3*2**30, batch_size=1,
               native_inputs=len(bank), calls_per_model=2*len(bank), sentinel=sentinel,
               forbidden_data_roots=[str((out/'evaluation').resolve()), str(weather.resolve())])
    write(out/'inference/protocol.json', inf)
    protocol = dict(created_utc=utcnow(), question='Does fixed depth1 tone continuation improve chronological-holdout airport temperature forecasts over native median and seasonal24?',
            cohort='Same fixed8 airports and October08/15/22/29 UTC origins; two incomplete JFK futures excluded prospectively before native inference.',
            stations=STATIONS, windows_by_station=counts, windows=30, context=128, horizon=32,
            methods=METHODS, primary='hybrid versus native, separately for each model',
            secondary='native and hybrid versus seasonal24; tones_zero versus hybrid descriptive diagnostic',
            tones_zero='Predeclared analytic control: accepted oscillatory tones only with zero residual, including zero DC; if abstained predicts zero. Diagnostic, not a competitive temperature baseline.',
            rule='Unmodified selective_tree.decompose(depth=1), stability=.65 energy=.15 prominence=8, frequency2..48/128. No training, new tuning, veto, refit, depth selection or forecast selection.',
            numeric='Decompose original float64 Celsius history; verify reconstruction rtol=atol=1e-12. Native inputs cast once to float32; singleton geometry. Native repeat and abstention gates are exact, never tolerance-based.',
            gates='All finite; every bank input predicted twice in separate batch1 calls (second pass reversed); exact repeat; additive reconstruction; exact abstention identity for all abstentions and a planted constant-history instrument. Deny all target/raw-data reads in worker.',
            loss='Per window: mean32 squared errors / original float64 population history variance (ddof0). No epsilon: fail if variance<=0. Then mean windows within each station, then equal mean8 stations.',
            uncertainty=dict(kind='descriptive paired station bootstrap percentile95%', draws=4096, seed=4954100,
                             resample='Same8 station indices with replacement shared across all methods/models; preserve all windows within station.',
                             caution='Shared weather creates dependence across stations; intervals are descriptive, not calibrated population confidence or significance. No sign-test/population-mean claim.'),
            hypothesis='Forecast grounding on chronologically unseen observations; geography, climate and temperature distribution can have been seen. Native outputs are medians, not means; squared-error findings do not identify training prior.',
            budget='<=300seconds/model including loading and gates; CPU1, RSS<=3GiB; cached offline weights. At most1 native worker in this track; explicit later slot2 grant required; no partial scoring.',
            budget_estimate='<=122 singleton calls/model (2*(30+30+1)); previous TF3968 calls plus gates560.7s, conservative estimate<90s/model including load. Hard300s supervisor bound.',
            preparation_sha256=sha(prior/'seal.json'), checkpoint_chronology=metadata['models'],
            preflight_manifest_sha256=sha(preflight/'manifest.json'),
            target_sha256=sha(out/'evaluation/targets.npz'), inference_sha256=sha(out/'inference/protocol.json'))
    write(out/'protocol.json', protocol); write(out/'seal.json', dict(sha256=sha(out/'protocol.json')))
    findings=dict(status='frozen; native HOLD', windows=30, stations=8, windows_by_station=counts,
                  accepted_histories=sum(p['accepted'] for p in plans), bank_inputs=len(bank),
                  singleton_calls_per_model=2*len(bank), native_launched=False)
    write(out/'knowledge.json', dict(question=protocol['question'], design=protocol, exact_findings=findings,
          controls=protocol['gates'], limitations=[protocol['uncertainty']['caution'],protocol['hypothesis'],
          'Quality flags null means unflagged, not evidence that QC completed; archived data are not a real-time vintage.'],
          evidence_class='prospective forecasting pilot preparation; no forecast results',
          next_decision='Complete model-free audit, then HOLD pending explicit native slot2 grant.'))
    write(out/'preparation_knowledge.json', read(out/'knowledge.json'))
    print(findings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weather', type=Path, default=Path('tsfm_lens/runs/postrelease_weather_202610'))
    args = parser.parse_args(); prepare(args.out, args.weather)
