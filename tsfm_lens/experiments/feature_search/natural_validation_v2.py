"""Prospective availability amendment to the failed 4951000 natural pilot.

Preserves the original modules and run. Uses the identical cached real_data
function with a private single-domain globals mapping, never changing its code
or shared globals. Only source availability may reduce a domain's cohort.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import re
import types

import numpy as np

from experiments.feature_search import natural_validation_study as base
from experiments.feature_search import coherence_study
from experiments.feature_search.coherence_study import read, write, sha, DOMAINS

PEDESTRIAN = 'monash_pedestrian_counts'


def domain_data(cache, domain, count):
    """Call the unchanged cached helper for one domain without global mutation."""
    original = coherence_study.real_data
    namespace = dict(original.__globals__, DOMAINS=(domain,))
    helper = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    return helper(cache, 64, count)


def available_sources(cache):
    """Reduce counts only on the helper's exact insufficient-availability error."""
    sources, availability = [], {}
    for domain in DOMAINS:
        requested = 2 if domain == PEDESTRIAN else 8
        count, error = requested, None
        try:
            selected = domain_data(cache, domain, count)
        except ValueError as exc:
            match = re.fullmatch(r'Only (\d+) eligible sources in '+re.escape(domain), str(exc))
            if match is None:
                raise
            error = str(exc)
            count = min(requested, int(match.group(1))-64)
            if count < 2:
                raise ValueError(f'{domain}: only {max(count, 0)} remaining sources; need >=2') from exc
            selected = domain_data(cache, domain, count)
        if [m['rank'] for m, _ in selected] != list(range(64, 64+count)):
            raise ValueError('Availability selection changed rank ordering')
        if any(m['domain'] != domain or m['split'] != ('fit' if j < count//2 else 'test')
               for j, (m, _) in enumerate(selected)):
            raise ValueError('Domain/source split contract changed')
        availability[domain] = dict(count=count, ranks=list(range(64, 64+count)),
            FIT=count//2, TEST=count-count//2, source_limited=count < 8,
            evidence='source-limited descriptive' if count < 8 else 'exploratory behavioral',
            availability_error=error)
        sources.extend(selected)
    return sources, availability


def record_knowledge(out, phase, status, findings, next_decision):
    """Distill this amended design and exact results while retaining all events."""
    out.mkdir(parents=True, exist_ok=True)
    path = out/'knowledge.json'
    events = read(path).get('events', []) if path.exists() else []
    protocol = read(out/'protocol.json') if (out/'protocol.json').exists() else {}
    amendment = read(out/'amendment.json') if (out/'amendment.json').exists() else {}
    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), phase=phase, status=status,
        question='Do selective hybrids improve unmodified real forecasts, and can two past hindcasts identify continuation failure?',
        design=dict(run=out.name, amendment=amendment,
            availability=protocol.get('availability'), sources=protocol.get('rows'),
            origins=list(base.ORIGINS), history=128, horizon=32,
            selector='Strict lower mean of two own-history-normalized hindcast losses; ties native',
            methods=list(base.METHODS), native_budget='Sequential CPU, one compute thread, 3GiB, 600 seconds/model'),
        exact_findings=findings,
        controls=['Origin-specific normalization', 'Future poisoning / scale invariance',
                  'Target-blind worker', 'Native ties', 'Source freshness and FIT/TEST separation',
                  'Finite, repeat, closure, hash and resource gates', 'No interval at n=1'],
        limitations=['Outcome-blind availability amendment; source-limited domains are descriptive only',
                     'n=1 split has no confidence interval or significance claim',
                     'Overlapping hindcasts are past selector features, not independent prospective tests',
                     'No TEST depth deployment; stability and abstention descriptive; pretraining overlap unknown'],
        evidence_class='exploratory behavioral/descriptive' if phase == 'analysis' and status == 'complete' else 'instrumentation',
        next_decision=next_decision)
    events.append(entry)
    write(path, dict(schema_version=2, **entry, events=events))


def prepare(out, cache):
    """Freeze an explicit amendment before accessing any new cohort arrays."""
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    previous = out.parent/'natural_validation_pilot4951000/knowledge.json'
    previous_hash = sha(previous)
    out.mkdir(parents=True, exist_ok=False)
    amendment = dict(version=2, previous_failed_run='natural_validation_pilot4951000',
        previous_knowledge_sha256=previous_hash,
        prior_negative_availability='Only 66 eligible sources in monash_pedestrian_counts; original 8/domain design impossible',
        authorization='User prospectively authorized availability-only reduction and slot1 serial execution',
        domain_policy={d: dict(requested_ranks=list(range(64, 66 if d == PEDESTRIAN else 72))) for d in DOMAINS},
        fallback='For another domain with <8 available, use all remaining ranks starting64, only if >=2; never use old sources or replace on outcomes',
        unchanged='Histories, two-hindcast selector, ties-native, extraction thresholds, models, FIT rules and budgets',
        limited_domain='Mark counts<8 source-limited descriptive; n=1 split has no CI or significance claim',
        outcome_blind=True, future_losses_read_before_amendment=False)
    write(out/'amendment.json', amendment)
    record_knowledge(out, 'availability_amendment', 'registered', dict(
        prior_negative_availability=amendment['prior_negative_availability'],
        previous_knowledge_sha256=previous_hash, native_workers_launched=False),
        'Apply the registered availability rule; freeze source IDs and protocol before native inference.')
    try:
        _prepare(out, cache)
    except Exception as exc:
        record_knowledge(out, 'prepare', 'failed', dict(error_type=type(exc).__name__, error=str(exc)),
                         'Inspect the recorded failure; do not substitute based on outcomes.')
        raise
    if sha(previous) != previous_hash:
        raise ValueError('Original failed pilot changed')
    p = read(out/'protocol.json')
    record_knowledge(out, 'prepare', 'complete', dict(sources=p['rows'], nodes=p['nodes'],
        availability=p['availability'], protocol_sha256=sha(out/'protocol.json'), native_workers_launched=False),
        'Slot1 granted: run the two native models sequentially after tests pass.')
    print(dict(sources=p['rows'], nodes=p['nodes'], availability=p['availability']), flush=True)


def _prepare(out, cache):
    """Create sealed inputs/targets using fixed original extraction and windows."""
    sources, availability = available_sources(cache)
    ids = [m['source'] for m, _ in sources]
    previous, audited = base.prior_sources(out.parent)
    if len(set(ids)) != len(ids) or previous.intersection(ids):
        raise ValueError('Duplicate/reused source IDs')
    bank, rows, plans, hind, future = base.build(sources)
    bank.save(out/'bank.npz')
    write(out/'rows.json', rows)
    write(out/'plans.json', plans)
    np.savez_compressed(out/'hindcast_targets.npz', targets=hind)
    for split in ('FIT', 'TEST'):
        ix = np.asarray([i for i, row in enumerate(rows) if row['split'] == split])
        np.savez_compressed(out/f'{split}_targets.npz', indices=ix, targets=future[ix])
    write(out/'provenance.json', dict(sources=[m for m, _ in sources], availability=availability,
        freshness=dict(overlap=[], audited_artifacts=audited,
                       scope='Recorded sibling rows/metadata; pretraining overlap unknown')))
    from experiments.feature_search import selective_tree, real_background_poc, homogeneous_normalizer, natural_validation_analysis_v2
    paths = [Path(__file__)] + [Path(m.__file__) for m in (base, coherence_study, selective_tree,
        real_background_poc, homogeneous_normalizer, natural_validation_analysis_v2)]
    files = ['amendment.json', 'bank.npz', 'rows.json', 'plans.json', 'hindcast_targets.npz',
             'FIT_targets.npz', 'TEST_targets.npz', 'provenance.json']
    protocol = dict(version=2, run=out.name, source_ids=ids, availability=availability,
        domains=list(DOMAINS), rows=len(rows), nodes=len(bank.values),
        origins=list(base.ORIGINS), methods=list(base.METHODS), models=list(base.MODELS),
        checkpoints=read(out.parent/'fourier_poc4929000/protocol.json')['checkpoints'],
        worker_cap=3*2**30, seconds_per_model=600, cpu_threads=1, gpu=False,
        eligibility='Unmodified coherence_study.real_data, scoped per-domain: past256 finite, main history nonconstant, hourly; no outcome replacement',
        normalization='Each origin independently centers/scales its own 128-point history after helper normalization',
        tree='Unchanged: stability .65, explained .15, prominence 8, max depth3',
        selector='At each fixed depth, hybrid iff mean of TWO own-history-standardized hindcast MSEs < native; ties and unusable histories native',
        fit_rule='Per domain/model choose native, fixed hybrid1/2/3 or past-selector1/2/3 by FIT future MSE; native first on ties; freeze before TEST',
        gates=dict(closure_fp64=1e-12, closure_fp32=2e-6, repeat_exact=True, finite=True,
                   fresh_source_ids=True, source_split_disjoint=True),
        interpretation='Exploratory unmodified-real behavioral pilot. Availability-only amendment. Source-limited domains descriptive; no CI/significance for n=1. Overlapping hindcasts are past selector features, not prospective tests. No best TEST depth deployed. Stability/abstention descriptive; no causal claim.',
        effect_gate='Descriptive paired mean gains; no significance or automatic promotion. Source-limited domains cannot qualify for generalization.',
        worker_reads=['protocol.json', 'bank.npz', 'rows.json', 'plans.json'],
        sources={str(p.resolve()): sha(p) for p in paths}, hashes={f: sha(out/f) for f in files})
    write(out/'protocol.json', protocol)


def run(out, slot_granted=False):
    """Use the unchanged sequential native monitor and record each outcome."""
    if not slot_granted:
        raise RuntimeError('Coordinator slot required')
    try:
        base._run(out, slot_granted=True)
    except Exception as exc:
        record_knowledge(out, 'native_execution', 'failed', dict(error_type=type(exc).__name__, error=str(exc),
            execution=read(out/'execution.json') if (out/'execution.json').exists() else {}),
            'Inspect failed instrumentation; invalid forecasts cannot support inference.')
        raise
    record_knowledge(out, 'native_execution', 'complete', read(out/'execution.json'),
                     'Freeze hindcast choices and FIT rules, then evaluate TEST targets.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--out', type=Path, default=Path('runs/natural_validation_pilot4951001'))
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--slot-granted', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.cache)
    elif args.command == 'run':
        run(args.out, args.slot_granted)
    else:
        from experiments.feature_search.natural_validation_analysis_v2 import analyze
        analyze(args.out)
