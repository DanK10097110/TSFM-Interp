"""Posthoc validity and saved-only paired-response audits; no fitting or native calls.

Raw pilot outputs remain immutable. A zero forecast effect is not a zero hidden
edit. Independently sampled +/- null edits are not a calibrated smooth-map null.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def read(path):
    """Read a preserved JSON artifact."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Write finite audit output without changing the source experiment."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def paired_score(h, rebuilt):
    """FP64 central differences remove shared offsets without dividing by full state."""
    import numpy as np
    h, rebuilt = np.asarray(h, dtype=np.float64), np.asarray(rebuilt, dtype=np.float64)
    desired = h[0]-h[1]
    actual = rebuilt[0]-rebuilt[1]
    energy = float(np.mean(desired**2))
    error = float(np.mean((actual-desired)**2))
    return dict(desired_energy=energy, error_energy=error,
                relative_error=None if energy <= 1e-20 else error/energy)


def validity_cell(row, direction, budget):
    """Separate numerical control availability from causal/semantic conclusions."""
    case = row['causal_selectivity'][direction]
    value = case['budgets'][str(budget)]
    selected = value['response_change_nmse']
    random = value['random_response_change_nmse']
    reasons = []
    if not row['exact_gates']:
        reasons.append('exact_native_gate_failed')
    if not case['reach']:
        reasons.append('natural_input_response_has_no_reach')
    if not value['random_valid']:
        reasons.append('random_geometry_invalid_fallback_not_a_null')
    if selected is None or selected == 0:
        reasons.append('selected_response_effect_missing_or_zero_NOT_an_activation_edit_claim')
    if random is None or random == 0:
        reasons.append('random_response_effect_missing_or_zero')
    numerical = not reasons
    reasons.append('independent_plus_minus_random_edits_NOT_a_calibrated_smooth_response_null')
    return dict(direction=direction, budget=budget, raw_random_geometry_valid=value['random_valid'],
                natural_input_reach=case['reach'], raw_selected_response_change_nmse=selected,
                raw_random_response_change_nmse=random,
                selected_response_change_nonzero=selected is not None and selected > 0,
                random_response_change_nonzero=random is not None and random > 0,
                descriptive_comparison_numerically_available=numerical,
                calibrated_response_null=False, causal_selectivity_claim_scorable=False,
                actual_perturbed_activation_edit_norm='not recorded; cached pairs can only provide a proxy',
                reasons=reasons)


def validity(root, dest):
    """Count every completed native method/model/seed/budget/direction cell."""
    dest.mkdir(exist_ok=False)
    cells, groups, missing = [], [], []
    for path in sorted(root.glob('native_*/results.json')):
        if not (path.parent/'execution.json').exists():
            missing.append(path.parent.name)
            continue
        if read(path.parent/'execution.json')['returncode']:
            missing.append(path.parent.name)
            continue
        fit = root/path.parent.name.replace('native_', 'fit_', 1)/'results.json'
        fitted = read(fit)
        meta = {key: fitted[key] for key in ('model', 'method', 'cut', 'k', 'seed')}
        rows = read(path)
        for direction in ('sinusoid', 'noise'):
            for budget in (1, 4):
                values = [dict(meta, row=row['row'], source=row['source'], domain=row['domain'],
                               **validity_cell(row, direction, budget)) for row in rows]
                cells.extend(values)
                groups.append(dict(meta, direction=direction, budget=budget, receivers=len(values),
                    random_geometry_valid=sum(v['raw_random_geometry_valid'] for v in values),
                    selected_response_nonzero=sum(v['selected_response_change_nonzero'] for v in values),
                    random_response_nonzero=sum(v['random_response_change_nonzero'] for v in values),
                    numerical_comparison_available=sum(v['descriptive_comparison_numerically_available'] for v in values),
                    calibrated_null_available=0, causal_selectivity_claim_scorable=0))
    write(dest/'cells.json', cells)
    write(dest/'groups.json', groups)
    write(dest/'knowledge.json', dict(question='Which causal response comparisons are actually scorable?',
        exact_results=dict(cells=len(cells), groups=len(groups), group_table='groups.json', cell_table='cells.json',
                           missing=missing, invalid_geometry=sum(not v['raw_random_geometry_valid'] for v in cells),
                           zero_selected_response=sum(not v['selected_response_change_nonzero'] for v in cells)),
        controls=['Raw fallback values retained but excluded as null evidence',
                  'Natural-input reach is distinct from selected-edit reach',
                  'Zero response effect is distinct from zero activation edit'],
        limitations=['Independently sampled +/- nulls are not calibrated smooth response maps; no semantic selectivity claim is supported by this battery',
                     'Actual perturbed edit norms were not saved; separate cached-state audit reports proxies'],
        next_step='Use saved-difference audit for response reconstruction; any new causal test needs reviewed matched smooth controls.',
        status='posthoc_validity_correction', supersedes='conservative-null wording and interpretations of zero effect as inactive/harmless'))
    print(json.dumps(dict(cells=len(cells), groups=len(groups))))


def response(root, dest, model):
    """Frozen dictionaries on saved +/- states; sourcewise FP64 difference errors."""
    import numpy as np
    import torch
    from experiments.feature_search import dictionary_core as core
    from experiments.feature_search.dictionary_pilot import verify, sha
    torch.set_num_threads(1)
    started = time.monotonic()
    dest.mkdir(exist_ok=False)
    _, source = verify(root, model)
    rows = read(source/'rows.json')
    fit, test = core.split_indices(rows)
    histories = np.load(source/'inputs.npz')['history']
    cuts = read(source/f'{model}_worker.json')['cuts']
    summaries, base_edits, paired_edits = [], [], []
    with np.load(source/f'{model}_states.npz') as archive:
        for path in sorted(root.glob(f'fit_{model}_*/dictionary.npz')):
            meta = read(path.parent/'results.json')
            if not (path.parent/'execution.json').exists() or read(path.parent/'execution.json')['returncode']:
                continue
            if sha(path) != meta['dictionary_sha256'] or meta['protocol_sha256'] != sha(root/'protocol.json'):
                raise ValueError('Frozen dictionary/protocol binding changed')
            artifact = dict(np.load(path))
            scale = float(artifact['scale'])
            c = cuts.index(meta['cut'])
            ids = {b: meta['perturbations'][str(b)]['selected'] for b in (1, 4)}
            identity = {key: meta[key] for key in ('model', 'method', 'cut', 'k', 'seed')}
            native_path = root/path.parent.name.replace('fit_', 'native_', 1)/'results.json'
            native_rows = {r['row']: r for r in read(native_path)} if native_path.exists() else {}
            for direction in ('sinusoid', 'noise'):
                metrics = []
                for i in test:
                    h = archive[f'{i}_{direction}_0.1_locked'][c]
                    code = core.encode((h-artifact['center'])/scale, artifact)
                    rebuilt = core.reconstruct(code, artifact).astype(np.float64)*scale
                    metrics.append(dict(row=int(i), source=rows[i]['source'], domain=rows[i]['domain'],
                                        mean_l0=float(np.mean(np.count_nonzero(code, axis=-1))),
                                        all_tokens_at_declared_k=bool(np.all(np.count_nonzero(code, axis=-1)==meta['k'])),
                                        **paired_score(h, rebuilt)))
                    if int(i) in native_rows:
                        for budget, selected in ids.items():
                            delta = -(code[..., selected] @ core.decoder(artifact)[selected])*scale
                            injected = (h+delta).astype(np.float32).astype(np.float64)-h.astype(np.float64)
                            paired_edits.append(dict(identity, row=int(i), direction=direction, budget=budget,
                                selected=selected, cached_active_tokens_per_sign=np.count_nonzero(np.any(code[..., selected] != 0, axis=-1), axis=-1).tolist(),
                                cached_injected_edit_rms_per_sign=np.sqrt(np.mean(injected**2, axis=(-2, -1))).tolist(),
                                status='CACHED_BATCH_PROXY_NOT_ACTUAL_FRESH_SINGLETON_EDIT'))
                sourcewise = []
                for source_id in sorted({r['source'] for r in metrics}):
                    group = [r for r in metrics if r['source']==source_id]
                    energy = float(np.mean([r['desired_energy'] for r in group]))
                    error = float(np.mean([r['error_energy'] for r in group]))
                    sourcewise.append(dict(source=source_id, domain=group[0]['domain'], dependent_rows=len(group),
                        desired_energy=energy, error_energy=error,
                        mean_l0=float(np.mean([r['mean_l0'] for r in group])),
                        all_tokens_at_declared_k=all(r['all_tokens_at_declared_k'] for r in group),
                        response_nmse=None if energy <= 1e-20 else error/energy))
                summaries.append(dict(identity, direction=direction, radius=.1, normalization='background_locked',
                    sourcewise=sourcewise, domains={domain: dict(n_sources=len(values),
                        mean_l0=float(np.mean([r['mean_l0'] for r in values])),
                        all_tokens_at_declared_k=all(r['all_tokens_at_declared_k'] for r in values),
                        mean_source_nmse=float(np.mean([r['response_nmse'] for r in values])),
                        ratio_of_mean_energies=float(np.mean([r['error_energy'] for r in values])/np.mean([r['desired_energy'] for r in values])))
                        for domain in sorted({r['domain'] for r in sourcewise})
                        if (values := [r for r in sourcewise if r['domain']==domain and r['response_nmse'] is not None])}))
            for i, native_record in native_rows.items():
                h = archive[f'{i}_full'][c:c+1]
                code = core.encode((h-artifact['center'])/scale, artifact)
                for budget, selected in ids.items():
                    delta = -(code[..., selected] @ core.decoder(artifact)[selected])*scale
                    injected = (h+delta).astype(np.float32).astype(np.float64)-h.astype(np.float64)
                    base_edits.append(dict(identity, row=i, budget=budget, selected=selected,
                        active_selected_entries=int(np.count_nonzero(code[..., selected])),
                        proposed_edit_l2=float(np.linalg.norm(delta.astype(np.float64))),
                        injected_edit_l2=float(np.linalg.norm(injected)),
                        injected_edit_rms=float(np.sqrt(np.mean(injected**2))),
                        injected_edit_nonzero=bool(np.any(injected != 0)),
                        native_forecast_edit_mse=native_record['measurements'][f'ablate{budget}']['mse'],
                        exact_saved_state_gate=native_record['exact_gates'],
                        status='BASE_EDIT_REPLAY_FROM_EXACT_SAVED_STATE_NOT_PERTURBED_EDIT'))
            if time.monotonic()-started > 580 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 3*2**30:
                raise RuntimeError('Saved-only audit resource cap')
    write(dest/'paired_response.json', summaries)
    write(dest/'base_edit_replay.json', base_edits)
    write(dest/'cached_pair_edit_proxies.json', paired_edits)
    write(dest/'knowledge.json', dict(question='Does reconstruction preserve held-out controlled hidden-state differences?',
        exact_results=dict(paired_response='paired_response.json', base_edits='base_edit_replay.json',
                           cached_proxies='cached_pair_edit_proxies.json', method_direction_rows=len(summaries),
                           seconds=time.monotonic()-started, peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        provenance=dict(protocol_sha256=sha(root/'protocol.json'), audit_code_sha256=sha(__file__),
                        source_state_sha256=read(root/'protocol.json')['hashes'][f'{model}_states.npz']),
        controls=['Frozen FIT-fitted dictionaries; TEST sourcewise aggregation; sinusoid AND noise at equal input RMS radius',
                  'Compare reconstructed plus-minus difference directly; decoder/center static offsets cancel',
                  'Exact base-state replay distinguished from cached-batch perturbed-state proxies'],
        limitations=['Descriptive response reconstruction, not true-signal semantics or forecast response preservation',
                     'Existing perturbation phase fixed; no fresh crossed-phase generalization',
                     'Cached +/- states were captured in a different batch geometry from native singleton intervention states'],
        next_step='Review joint full-state, paired-response, forecast and validity tables before any expansion.', status='saved_only_audit_complete'))
    print(json.dumps(dict(model=model, method_direction_rows=len(summaries), seconds=time.monotonic()-started)))


def monitored_response(args):
    """Bound saved-only analysis externally, including imports and matrix operations."""
    if not args.slot_granted:
        raise ValueError('Saved-only audit requires the granted slot')
    if args.dest.exists():
        raise FileExistsError(args.dest)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    command = [sys.executable, '-m', 'experiments.feature_search.dictionary_audit', '_response',
               '--root', str(args.root), '--dest', str(args.dest), '--model', args.model]
    start, peak, reason = time.monotonic(), 0, None
    process = subprocess.Popen(command, env=env, start_new_session=True)
    while process.poll() is None:
        try:
            lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
            rss = int(next((l.split()[1] for l in lines if l.startswith('VmRSS:')), '0'))*1024
            peak = max(peak, rss+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        except FileNotFoundError:
            pass
        if peak > 3*2**30 or time.monotonic()-start > 600:
            reason = 'resource cap'
            os.killpg(process.pid, 9)
            process.wait()
            break
        time.sleep(.1)
    args.dest.mkdir(exist_ok=True)
    execution = dict(returncode=process.returncode, reason=reason, peak_rss=peak, seconds=time.monotonic()-start)
    write(args.dest/'execution.json', execution)
    if process.returncode:
        write(args.dest/'knowledge.json', dict(question='Saved-only response audit', exact_results=execution,
              controls=['external resource caps'], limitations=['failed; no completed result'],
              next_step='Inspect failure before retry', status='failed'))
        raise RuntimeError(execution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validity', 'response', '_response'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--model', choices=('bolt', 'timesfm'))
    parser.add_argument('--slot-granted', action='store_true')
    args = parser.parse_args()
    if args.command == 'validity':
        validity(args.root, args.dest)
    elif args.command == 'response':
        monitored_response(args)
    else:
        response(args.root, args.dest, args.model)
