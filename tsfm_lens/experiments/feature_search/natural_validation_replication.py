"""Frozen depth-one natural replication; preparation never launches native models.

Ranks72..135 in electricity, M4 and traffic are all fresh evaluation sources.
The original cached helper, normalizer and selective extraction remain intact.
Only M4 retains depth3 residual inference for the frozen pilot secondary rule.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import resource
import subprocess
import sys
import time
import types

import numpy as np

from experiments.feature_search import coherence_study as cached
from experiments.feature_search import natural_validation_study as base
from experiments.feature_search.selective_tree import decompose, continuation
from experiments.feature_search.coherence_study import read, write, sha, Bank, load_bank

DOMAINS = ('monash_electricity_hourly', 'm4_hourly', 'monash_traffic')
MODELS = base.MODELS
ORIGINS = base.ORIGINS
METHODS = ('native', 'hybrid1', 'seasonal_naive24', 'tone_last1', 'pilot_candidate')
OFFSET, COUNT = 72, 64
SECONDS, MEMORY = 900, 3*2**30


def knowledge(out, phase, status, findings, next_decision):
    """Append successful or failed experiments without losing prior evidence."""
    path = out/'knowledge.json'
    events = read(path).get('events', []) if path.exists() else []
    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), phase=phase, status=status,
        question='Does fixed depth1 improve fresh unmodified real forecasts, and does two-hindcast benefit predict main benefit?',
        design=dict(domains=list(DOMAINS), ranks=list(range(OFFSET, OFFSET+COUNT)), sources=192,
            split='All fresh sources TEST; rules inherited from pilot FIT, no new fitting',
            origins=list(ORIGINS), history=128, horizon=32, primary='hybrid1 versus native, six domain/model cells',
            native_point_forecast='Median for both: Bolt quantile .5, TimesFM aridx5; evaluated by MSE',
            secondary='Frozen depth1 selector, seasonal24, tone_last1 and original pilot FIT domain rule'),
        exact_findings=findings,
        controls=['Past-only availability audit', 'Independent normalization at each origin',
            'Target-blind native worker', 'Fresh source IDs', 'Frozen strict two-hindcast mean loss, ties native',
            'Paired source uncertainty adjusted across six primary comparisons',
            'Finite, exact-repeat, reconstruction, hashes and resource caps'],
        limitations=['Conditional source resampling; cross-source calendar dependence may remain',
            'Overlapping historical hindcasts are selector features, not independent prospective tests',
            'Hindcast/main association is descriptive and can vary over time; no causal or universal predictability claim',
            'No TEST threshold tuning, depth selection or source replacement', 'Real pretraining overlap unknown'],
        evidence_class='behavioral replication; diagnostic association descriptive' if phase == 'analysis' and status == 'complete' else 'instrumentation',
        next_decision=next_decision)
    events.append(entry)
    write(path, dict(schema_version=3, **entry, events=events))


def eligible_past(past, stamps):
    """Match cached eligibility using 256 observed values and known timestamps."""
    past = np.asarray(past, float)
    if past.shape != (256,):
        raise ValueError('Availability must receive mainpast256 only')
    return bool(np.isfinite(past).all() and past[128:256].std() > 1e-8
                and np.all(np.diff(stamps) == np.timedelta64(1, 'h')))


def audit_table(table, domain):
    """Never materialize the 32 future target values while auditing membership."""
    ids = []
    for index in np.random.default_rng(492306).permutation(len(table)):
        values = table['target'][int(index)].values
        if len(values) < 288:
            continue
        past = values.slice(len(values)-288, 256).to_numpy(zero_copy_only=False)
        timestamps = table['timestamp'][int(index)].values
        stamps = timestamps.slice(len(timestamps)-288, 288).to_numpy(zero_copy_only=False)
        if eligible_past(past, stamps):
            ids.append(domain+':'+table['id'][int(index)].as_py())
        if len(ids) == OFFSET+COUNT:
            break
    return dict(domain=domain, sufficient=len(ids) >= OFFSET+COUNT,
        eligible_count_lower_bound=len(ids), count_exact_if_insufficient=len(ids) if len(ids) < OFFSET+COUNT else None,
        selected_source_ids=ids[OFFSET:OFFSET+COUNT], ranks=list(range(OFFSET, min(len(ids), OFFSET+COUNT))))


def availability(cache):
    """Audit all domains before any selected future values are materialized."""
    import pyarrow as pa
    result = {}
    for domain in DOMAINS:
        paths = sorted((cache/'datasets/autogluon___chronos_datasets'/domain).rglob('*train.arrow'))
        if len(paths) != 1:
            raise ValueError(f'Need one cached hourly source file: {domain}')
        with pa.memory_map(str(paths[0]), 'r') as handle:
            table = pa.ipc.open_stream(handle).read_all()
            result[domain] = dict(audit_table(table, domain), source_file=str(paths[0]), source_file_sha256=sha(paths[0]))
    return result


def source_data(cache, audit):
    """Use the exact original helper after a complete outcome-blind availability gate."""
    if not all(audit[d]['sufficient'] for d in DOMAINS):
        raise ValueError('Insufficient fresh availability; no reduced cohort or source substitution authorized')
    result = []
    for domain in DOMAINS:
        original = cached.real_data
        helper = types.FunctionType(original.__code__, dict(original.__globals__, DOMAINS=(domain,)),
                                    original.__name__, original.__defaults__, original.__closure__)
        values = helper(cache, OFFSET, COUNT)
        if [m['source'] for m, _ in values] != audit[domain]['selected_source_ids']:
            raise ValueError('Availability audit and cached helper disagree')
        for meta, value in values:
            result.append((dict(meta, helper_split=meta['split'], split='test'), value))
    return result


def build(sources):
    """Bank only roots and needed residuals; tone continuations need no model call."""
    bank, rows, plans, hind, future = Bank(), [], [], [], []
    for meta, value in sources:
        value = np.asarray(value, float)
        if value.shape != (288,) or not np.isfinite(value).all():
            raise ValueError('Selected data invalid; never replace on outcomes')
        row = {k: meta[k] for k in ('source', 'domain', 'rank')}
        row.update(index=len(rows), split='TEST')
        rows.append(row)
        pp, targets = [], []
        max_depth = 3 if meta['domain'] == 'm4_hourly' else 1
        for origin in ORIGINS:
            x, center, scale, valid = base.normalize_history(value[origin-128:origin])
            levels, details = decompose(x, depth=max_depth)
            closure64 = max(float(abs(np.sum(level, axis=0)-x).max()) for level in levels)
            closure32 = max(float(abs(np.asarray(level, np.float32).sum(axis=0)-x.astype(np.float32)).max())
                            /max(1., float(abs(x).max())) for level in levels)
            if closure64 > 1e-12 or closure32 > 2e-6:
                raise ValueError('Reconstruction gate failed')
            d1, ds = min(1, len(details)), min(max_depth, len(details))
            pp.append(dict(origin=origin, history=[origin-128, origin], target=[origin, origin+32],
                center=center, scale=scale, selector_usable=valid, root=bank.add(x),
                residual1=bank.add(levels[d1][-1]), secondary_residual=bank.add(levels[ds][-1]),
                primary_depth=d1, secondary_depth=ds, details=details, abstained=d1 == 0,
                closure_fp64=closure64, closure_fp32=closure32,
                amplitudes=[float(np.linalg.norm(d['coefficients'])) for d in details]))
            targets.append((value[origin:origin+32]-center)/scale)
        plans.append(pp)
        hind.append(targets[:2])
        future.append(targets[2])
    return bank, rows, plans, np.asarray(hind), np.asarray(future)


def budget(nodes, pilot_nodes, execution):
    """Conservative linear scaling of complete pilot runtime, plus 2x allowance."""
    result = {}
    for model in MODELS:
        nominal = execution[model]['seconds']*max(1., nodes/pilot_nodes)
        result[model] = dict(pilot_seconds=execution[model]['seconds'], pilot_nodes=pilot_nodes,
            bank_nodes=nodes, estimated_seconds=nominal, allowance_seconds=2*nominal,
            within_900_seconds=2*nominal <= SECONDS,
            caveat='Estimate includes scaled loading overhead; not a measured runtime guarantee. Same batch4 bounds peak memory.')
    return result


def prepare(out, cache):
    """Register strict membership and statistics before any selected future loading."""
    if out.exists():
        raise FileExistsError('Fresh run required')
    out.mkdir(parents=True, exist_ok=False)
    declaration = dict(version=3, hold_native=True, fresh_slot_required=True, domains=list(DOMAINS),
        ranks=list(range(OFFSET, OFFSET+COUNT)), sources=192, excluded='pedestrian exhausted',
        split='All 192 evaluation sources; no new FIT rules', primary_depth=1,
        native_point_forecast='Median for both: Bolt quantile .5, TimesFM aridx5; MSE compares these chosen point forecasts, not predictive means',
        primary_family=[dict(model=m, domain=d, contrast='hybrid1-native') for m in MODELS for d in DOMAINS],
        primary_uncertainty='23999 paired source bootstrap draws, two-sided Bonferroni family6 intervals at alpha=.05/6; approximate bootstrap coverage',
        winrate='Strict wins/all sources, ties separately; non-tie exact sign tests Holm6 as supportive directional endpoint, not a mean-effect test',
        conditional='Prespecified main extraction eligible and all-three-origins extraction eligible; conditional summaries descriptive, not extra primary tests',
        continuous_diagnostic='Spearman correlation of mean two-hindcast benefit with main benefit; 9999 source-pairing permutations, positive one-sided p, Holm6 diagnostic family separate from primary. Pearson and per-origin variability descriptive.',
        selector='hybrid1 iff mean two own-history-normalized hindcast MSEs strictly less than native, both histories usable; ties native',
        secondary='Original pilot FIT-selected domain rule, unchanged and hash-bound; compute extra depth3 residuals only for M4',
        no_future_threshold_fit=True, no_outcome_replacement=True,
        budgets=dict(cpu_threads=1, worker_bytes=MEMORY, seconds_per_model=SECONDS, models_sequential=True))
    write(out/'declaration.json', declaration)
    try:
        audit = availability(cache)
        write(out/'availability.json', audit)
        knowledge(out, 'availability', 'complete' if all(v['sufficient'] for v in audit.values()) else 'failed',
                  audit, 'Only prepare the full192 if every domain supplies ranks72..135; native remains on HOLD.')
        sources = source_data(cache, audit)
        previous, audited = base.prior_sources(out.parent)
        ids = [m['source'] for m, _ in sources]
        if len(ids) != 192 or len(set(ids)) != 192 or previous.intersection(ids):
            raise ValueError('Need 192 fresh distinct source IDs')
        bank, rows, plans, hind, future = build(sources)
        pilot = out.parent/'natural_validation_pilot4951001'
        pilot_protocol, rules = read(pilot/'protocol.json'), read(pilot/'fit_rules.json')['rules']
        frozen = {m: {d: rules[m][d] for d in DOMAINS} for m in MODELS}
        if frozen != {'bolt': {DOMAINS[0]: 'hybrid1', DOMAINS[1]: 'past_selector3', DOMAINS[2]: 'native'},
                      'timesfm': {DOMAINS[0]: 'hybrid1', DOMAINS[1]: 'hybrid3', DOMAINS[2]: 'native'}}:
            raise ValueError('Pilot rule changed; explicit amendment required')
        estimate = budget(len(bank.values), pilot_protocol['nodes'], read(pilot/'execution.json'))
        if not all(v['within_900_seconds'] for v in estimate.values()):
            raise ValueError('Conservative expected budget exceeds15min/model')
        bank.save(out/'bank.npz')
        write(out/'rows.json', rows)
        write(out/'plans.json', plans)
        np.savez_compressed(out/'hindcast_targets.npz', targets=hind)
        np.savez_compressed(out/'main_targets.npz', targets=future, source_ids=np.asarray(ids))
        write(out/'provenance.json', dict(sources=[m for m, _ in sources], freshness=dict(overlap=[], audited_artifacts=audited)))
        write(out/'budget.json', estimate)
        from experiments.feature_search import selective_tree, real_background_poc, homogeneous_normalizer, natural_validation_replication_analysis
        modules = (base, cached, selective_tree, real_background_poc, homogeneous_normalizer, natural_validation_replication_analysis)
        paths = [Path(__file__)]+[Path(m.__file__) for m in modules]
        artifacts = ['declaration.json', 'availability.json', 'bank.npz', 'rows.json', 'plans.json',
                     'hindcast_targets.npz', 'main_targets.npz', 'provenance.json', 'budget.json']
        protocol = dict(declaration, run=out.name, rows=len(rows), nodes=len(bank.values),
            source_ids=ids, models=list(MODELS), origins=list(ORIGINS), methods=list(METHODS),
            frozen_pilot_rules=frozen, pilot_bindings={str(pilot/name): sha(pilot/name)
                for name in ('protocol.json', 'fit_rules.json', 'execution.json', 'knowledge.json')},
            checkpoints=pilot_protocol['checkpoints'], worker_cap=MEMORY, seconds_per_model=SECONDS,
            gates=dict(closure_fp64=1e-12, closure_fp32=2e-6, repeat_exact=True, finite=True),
            worker_reads=['protocol.json', 'bank.npz', 'rows.json', 'plans.json'],
            sources={str(p.resolve()): sha(p) for p in paths}, hashes={name: sha(out/name) for name in artifacts})
        write(out/'protocol.json', protocol)
    except Exception as exc:
        knowledge(out, 'prepare', 'failed', dict(error_type=type(exc).__name__, error=str(exc), native_launched=False),
                  'Resolve the recorded failure; no native launch and no outcome-based replacement.')
        raise
    knowledge(out, 'prepare', 'complete', dict(sources=192, availability=audit, nodes=len(bank.values),
        budget=estimate, frozen_pilot_rules=frozen, protocol_sha256=sha(out/'protocol.json'), native_launched=False,
        main_future_losses_read=False), 'HOLD native. Await a new coordinator slot grant; slot1 currently dictionary.')
    print(dict(sources=192, nodes=len(bank.values), budget=estimate, native='HOLD'), flush=True)


def assemble(bank, plans, forecasts, rows, model, rules):
    """Use classical tones plus native residual; secondary candidate follows pilot."""
    result = np.zeros((len(rows), 3, len(METHODS), 32))
    for i, source in enumerate(plans):
        rule = rules[model][rows[i]['domain']]
        for j, plan in enumerate(source):
            x = bank[plan['root']]
            native = forecasts[plan['root']]
            hybrid = continuation(plan['details'], plan['primary_depth'])+forecasts[plan['residual1']]
            secondary = native if rule == 'native' else hybrid
            if rule in ('hybrid3', 'past_selector3'):
                secondary = continuation(plan['details'], plan['secondary_depth'])+forecasts[plan['secondary_residual']]
            result[i, j] = [native, hybrid, np.resize(x[-24:], 32),
                continuation(plan['details'], plan['primary_depth'])+bank[plan['residual1']][-1], secondary]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite composed forecasts')
    return result


def worker(out, model, native_factory=None):
    """Target-blind inference; only the monitored dispatcher may invoke this."""
    if os.environ.get('NATURAL_REPLICATION_MONITORED') != '1':
        raise RuntimeError('Use run after a fresh slot grant under the resource monitor')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in base.THREAD_VARS):
        raise RuntimeError('Requires CPU and one compute thread')
    started = time.monotonic()
    p = read(out/'protocol.json')
    for path, digest in p['sources'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed')
    for name in ('bank.npz', 'rows.json', 'plans.json'):
        if sha(out/name) != p['hashes'][name]:
            raise ValueError('Frozen worker input changed')
    if (out/f'{model}_forecasts.npz').exists():
        raise FileExistsError('Refuse forecast overwrite')
    checkpoint = p['checkpoints'][model]
    for name, digest in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/name) != digest:
            raise ValueError('Weights changed')
    if native_factory is None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from experiments.feature_search.real_background_poc import Native
        native_factory = Native
    native = native_factory(model, checkpoint['path'])
    values, plans, rows = load_bank(out/'bank.npz'), read(out/'plans.json'), read(out/'rows.json')
    forecasts = np.zeros((len(values), 32), np.float32)
    for start in range(0, len(values), 4):
        forecasts[start:start+4] = native.predict(np.stack(values[start:start+4]))[0]
        if time.monotonic()-started > SECONDS or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MEMORY:
            raise RuntimeError('Resource cap exceeded')
    if not np.array_equal(native.predict(np.stack(values[:4]))[0], forecasts[:4]) or not np.isfinite(forecasts).all():
        raise ValueError('Finite/repeat gate failed')
    result = assemble(values, plans, forecasts, rows, model, p['frozen_pilot_rules'])
    np.savez_compressed(out/f'{model}_forecasts.npz', forecasts=result, bank_forecasts=forecasts,
        source_ids=np.asarray(p['source_ids']), methods=np.asarray(METHODS), origins=np.asarray(ORIGINS))
    write(out/f'{model}_worker.json', dict(protocol_sha256=sha(out/'protocol.json'),
        sha256=sha(out/f'{model}_forecasts.npz'), repeat_exact=True, finite=True,
        closure_fp32=max(q['closure_fp32'] for source in plans for q in source),
        seconds=time.monotonic()-started, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def run(out, slot_granted=False):
    """Explicit new grant required; sequential monitored workers with 15min caps."""
    if not slot_granted:
        raise RuntimeError('HOLD native: fresh coordinator slot grant required')
    import fcntl
    if (out/'execution.json').exists():
        raise FileExistsError('Refuse rerun of frozen experiment')
    execution = {}
    try:
        with (out/'native.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for model in MODELS:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1', NATURAL_REPLICATION_MONITORED='1',
                    PYTHONPATH=str(Path(__file__).resolve().parents[2]), **{k:'1' for k in base.THREAD_VARS})
                started, peak, reason = time.monotonic(), 0, None
                with (out/f'{model}.log').open('x') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.natural_validation_replication',
                        'worker', '--out', str(out), '--model', model], env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while proc.poll() is None:
                            try:
                                lines = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                                peak = max(peak, int(next(l.split()[1] for l in lines if l.startswith('VmRSS:')))*1024)
                            except (FileNotFoundError, StopIteration):
                                pass
                            if peak > MEMORY or time.monotonic()-started > SECONDS:
                                reason = 'Resource cap exceeded'
                                break
                            time.sleep(.1)
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                record = read(out/f'{model}_worker.json') if (out/f'{model}_worker.json').exists() else {}
                valid = proc.returncode == 0 and reason is None and bool(record)
                valid = valid and record.get('peak_rss', float('inf')) <= MEMORY and record.get('seconds', float('inf')) <= SECONDS
                execution[model] = dict(valid=valid, returncode=proc.returncode, reason=reason,
                    seconds=time.monotonic()-started, monitored_peak=peak)
                write(out/'execution.json', execution)
                if not valid:
                    raise RuntimeError(f'{model} invalid; stopping sequential run')
    except Exception as exc:
        knowledge(out, 'native_execution', 'failed', dict(error=str(exc), execution=execution),
                  'Inspect failed instrumentation; invalid outputs cannot support inference.')
        raise
    knowledge(out, 'native_execution', 'complete', execution, 'Freeze past-only choices before opening main targets.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'analyze'])
    parser.add_argument('--out', type=Path, default=Path('runs/natural_validation_replication4955000'))
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--slot-granted', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.cache)
    elif args.command == 'run':
        run(args.out, args.slot_granted)
    elif args.command == 'worker':
        if args.model is None:
            parser.error('worker requires --model')
        worker(args.out, args.model)
    else:
        from experiments.feature_search.natural_validation_replication_analysis import analyze
        analyze(args.out)
