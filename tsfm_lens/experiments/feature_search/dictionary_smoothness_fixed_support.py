"""Bounded saved-only child study isolating support switches with identical weights."""
from pathlib import Path
from datetime import datetime, timezone
import os
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha
from experiments.feature_search import dictionary_smoothness_math as geometry
from experiments.feature_search import dictionary_smoothness_fixed_support_math as controls
from experiments.feature_search import dictionary_smoothness_fixed_support_audit as independent

OUT = Path('runs/dictionary_smoothness_fixed_support4965100')
PARENT = Path('runs/dictionary_smoothness4965000')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
MODES = ('hard', 'fixed_support', 'fixed_affine')


def knowledge(out, phase, status, findings):
    """Persist compact child knowledge, including any failed instrumentation."""
    file = out / 'knowledge.json'
    previous = read(file) if file.exists() else dict(events=[])
    event = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), phase=phase, status=status,
        question='Does freezing actual-midpoint support remove dictionary roughness, and what changes in state/hidden-response fidelity?',
        design='Same frozen weights, locked normalization, k16, seed4951000;16 synthetic sources,768 matched-radius paths; hard versus frozen TopK with live ReLU versus frozen TopK and central ReLU gate.',
        exact_findings=findings,
        controls=['Same central selected indices, weights and coefficients across gate contrasts',
                  'Affine isolated-chord null', 'Independent dense replay and exact repeat in FP64/FP32 on16 prescribed paths',
                  'Hard baseline matches parent saved metrics', 'Equal-source aggregation and paired fidelity changes'],
        limitations=['Off-manifold activation interpolation is not a newly evaluated native path.',
                     'Response means hidden-state endpoint-minus-midpoint perturbation response, not forecast accuracy.',
                     'FP64 dictionary arithmetic on saved FP32 native states; FP32 replay sampled one path/source.',
                     'Fixed gates are path-local diagnostics chosen at each actual midpoint, not a deployable global encoder.',
                     'One seed,k16,first-block residuals,16 synthetic sources; descriptive geometry, no general monosemanticity or superiority claim.'],
        evidence_class='L1 saved-array controlled methodology', native_calls=0, fit_calls=0,
        parent_wording_correction=dict(
            source='Parent dictionary_smoothness_distill.py qualification about unswitched learned maps',
            correction='Parent support measured NONZERO active coordinates. Unchanged active support at the sampled chord points implies an affine map on those points. Observed active-support changes may combine TopK membership changes and ReLU zero crossings; the parent audit did not separate them. The fixed-index/live-ReLU and fixed-index/fixed-ReLU controls now separate these mechanisms.',
            numeric_results_unchanged=True, parent_files_unchanged=True),
        next_decision='Close this branch after audited result. No broader grid, fit or native launch.')
    previous.update(event)
    previous['events'].append(event)
    write(file, previous)


def prepare(out):
    """Seal a separate child protocol without changing any parent or Hume file."""
    out.mkdir(exist_ok=False)
    try:
        parent = read(PARENT / 'protocol.json')
        assert read(PARENT / 'aggregation_audit.json')['valid']
        bindings = dict(parent['bindings'])
        for path, digest in {**bindings, **parent['code_hashes']}.items():
            assert sha(path) == digest, 'Frozen parent/source changed: ' + path
        bindings.update(parent['code_hashes'])
        for name in ('protocol.json', 'compact_record.json', 'aggregation_audit.json', 'results.json'):
            bindings[str(PARENT / name)] = sha(PARENT / name)
        arms = [a for a in parent['artifacts'] if a['k'] == 16]
        for a in arms:
            method = 'pca_adaptive' if a['method'] == 'pca' else a['method']
            file = PARENT / f"{a['model']}_locked_{method}_k16_metrics.npz"
            bindings[str(file)] = sha(file)
        code = [Path(__file__), Path(controls.__file__), Path(independent.__file__), Path(geometry.__file__),
                Path('tests/test_dictionary_smoothness_fixed_support.py'),
                Path(__file__).with_name('dictionary_smoothness_fixed_support_plants.py')]
        protocol = dict(version=1, parent=str(PARENT), response=parent['response'], artifacts=arms,
            source_count=16, paths=768, policy='locked', k=16, seed=4951000, modes=MODES,
            central_support='Stable magnitude TopK at actual saved input midpoint per path/token; stable index ties including zero ties; exactly16 indices.',
            fixed_support='Freeze TopK indices; retain state-dependent coefficients and learned ReLU.',
            fixed_affine='Freeze TopK indices and preactivation>0 ReLU gate at actual midpoint; selected signed preactivations pass without re-clipping.',
            fidelity='State reconstruction over minus,plus,actual midpoint; response reconstruction over endpoint-minus-actual-midpoint vectors. Both absolute RMS and relative errors; no forecast evaluations.',
            precision='Primary all-path FP64; independent FP32/FP64 dense replay and exact repeat first registered path/source. FP32 chooses its own central support/gate; report disagreement.',
            statistics='Equal16-source mean and paired differences; preserve six axis/radius strata, no path/token inference.',
            resource=dict(threads=1, bytes=2*2**30, seconds=60),
            bindings=bindings, code_hashes={str(x.resolve()):sha(x) for x in code})
        write(out / 'protocol.json', protocol)
        knowledge(out, 'prepare', 'complete', dict(protocol_sha256=sha(out / 'protocol.json')))
    except Exception as exc:
        knowledge(out, 'prepare', 'failed', dict(error=repr(exc)))
        raise


def worker(out):
    """Infer only from saved arrays under the external one-thread memory/time monitor."""
    assert os.environ.get('DICTIONARY_SMOOTHNESS_CHILD_MONITORED') == '1'
    assert all(os.environ.get(k) == '1' for k in THREADS)
    started = time.monotonic()
    p = read(out / 'protocol.json')
    for file, digest in {**p['bindings'], **p['code_hashes']}.items():
        assert sha(file) == digest, file
    paths = read(Path(p['response']) / 'paths.json')
    source_ids = sorted({r['source'] for r in paths})
    sample = [next(i for i,r in enumerate(paths) if r['source'] == source) for source in source_ids]
    radii = np.asarray([r['actual_radius'] for r in paths])
    results = []
    audits = []
    for model in ('bolt', 'timesfm'):
        native = np.load(Path(p['response']) / f'{model}_locked_hidden_0.npy', mmap_mode='r')
        h = np.stack([native[[r[key] for r in paths]].astype(float) for key in ('minus', 'plus', 'midpoint')])
        h = np.concatenate([h, ((h[0] + h[1]) / 2)[None]])
        reference = geometry.rms(native[[r['base'] for r in paths]], (-2, -1))
        for item in [a for a in p['artifacts'] if a['model'] == model]:
            method = 'pca_adaptive' if item['method'] == 'pca' else item['method']
            a = dict(np.load(item['path'], allow_pickle=False))
            pre = controls.coefficients(h, a, method)
            indices, gate = controls.central_masks(pre, 16, method)
            hard = controls.decode_control(pre, a, method, 'hard', indices, gate)
            pre32 = controls.coefficients(h[:,sample], a, method, np.float32)
            indices32, gate32 = controls.central_masks(pre32, 16, method)
            np.savez(out / f'{model}_{method}_gates.npz', indices=indices, gate=gate,
                     sample_indices=np.asarray(sample), FP32_indices=indices32, FP32_gate=gate32)
            parent_values = dict(np.load(PARENT / f'{model}_locked_{method}_k16_metrics.npz'))
            hard_metrics = None
            for mode in MODES:
                g = hard if mode == 'hard' else controls.decode_control(pre, a, method, mode, indices, gate)
                measured, actual, isolated = geometry.measure(h, g, None, reference, radii)
                metrics = {key:measured[key] for key in (
                    'raw_chord_RMS', 'decoded_actual_chord_RMS', 'isolated_dictionary_chord_RMS',
                    'transported_midpoint_deviation_RMS', 'chord_decomposition_error',
                    'raw_curvature', 'decoded_curvature', 'isolated_curvature')}
                metrics.update(controls.fidelity(h, g, hard, reference))
                if mode == 'hard':
                    hard_metrics = dict(metrics)
                    for key in ('raw_chord_RMS', 'decoded_actual_chord_RMS', 'isolated_dictionary_chord_RMS'):
                        assert np.allclose(metrics[key], parent_values[key], atol=1e-12, rtol=1e-10), ('parent', key)
                for key in ('decoded_actual_chord_RMS', 'isolated_dictionary_chord_RMS',
                            'state_reconstruction_RMS', 'state_reconstruction_relative_RMS',
                            'response_reconstruction_RMS', 'response_reconstruction_relative_RMS'):
                    metrics[key + '_change_from_hard'] = metrics[key] - hard_metrics[key]
                assert float(metrics['central_output_change_from_hard_RMS'].max()) < 1e-12
                if mode == 'fixed_affine' or (method == 'pca_adaptive' and mode == 'fixed_support'):
                    assert float(metrics['isolated_dictionary_chord_RMS'].max()) < 1e-10, 'Affine null'
                g32 = controls.decode_control(pre32, a, method, mode, indices32, gate32)
                check = independent.check(h[:,sample], a, method, mode, g[:,sample], g32)
                check.update(model=model, method=method, mode=mode,
                    max_FP32_FP64_output_difference=float(np.max(abs(g[:,sample] - g32))),
                    central_indices_differ=int(np.sum(np.any(indices[sample] != indices32, axis=-1))),
                    central_gates_differ=int(np.sum(np.any(gate[sample] != gate32, axis=-1))))
                audits.append(check)
                name = f'{model}_{method}_{mode}'
                np.savez(out / f'{name}_sample.npz', h=h[:,sample], FP64=g[:,sample], FP32=g32)
                np.savez_compressed(out / f'{name}_metrics.npz', **metrics)
                results.append(dict(model=model, method=method, mode=mode,
                    gates_file=f'{model}_{method}_gates.npz', gates_sha256=sha(out / f'{model}_{method}_gates.npz'),
                    metrics_file=f'{name}_metrics.npz', metrics_sha256=sha(out / f'{name}_metrics.npz'),
                    sample_file=f'{name}_sample.npz', sample_sha256=sha(out / f'{name}_sample.npz'),
                    overall={key:controls.source_summary(value, paths) for key,value in metrics.items()},
                    by_axis_radius=geometry.aggregate(metrics, paths)))
                print(dict(model=model, method=method, mode=mode, seconds=time.monotonic()-started), flush=True)
                del g, actual, isolated, g32
            del pre, hard, pre32
        del h, native
    write(out / 'results.json', dict(arms=results, source_count=16, native_calls=0, fit_calls=0))
    write(out / 'independent_replay.json', dict(valid=True, arms=audits))
    write(out / 'worker.json', dict(valid=True, seconds=time.monotonic()-started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        protocol_sha256=sha(out / 'protocol.json'), results_sha256=sha(out / 'results.json'),
        replay_sha256=sha(out / 'independent_replay.json'), native_calls=0, fit_calls=0))


def run(out):
    """Enforce60seconds and2GiB including parent RSS; preserve failures instead of retrying."""
    assert not (out / 'execution.json').exists(), 'No silent rerun'
    started = time.monotonic()
    peak = 0
    reason = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', DICTIONARY_SMOOTHNESS_CHILD_MONITORED='1',
               **{k:'1' for k in THREADS})
    try:
        with (out / 'worker.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'experiments.feature_search.dictionary_smoothness_fixed_support', 'worker'],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    try:
                        lines = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        peak = max(peak, int(next(v.split()[1] for v in lines if v.startswith('VmRSS:')))*1024)
                    except (FileNotFoundError, StopIteration):
                        pass
                    if peak + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 2*2**30 or time.monotonic()-started > 60:
                        reason = '60sec/2GiB cap exceeded'
                        break
                    time.sleep(.1)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
        worker_record = read(out / 'worker.json') if (out / 'worker.json').exists() else {}
        execution = dict(valid=proc.returncode == 0 and reason is None and worker_record.get('valid', False),
            returncode=proc.returncode, seconds=time.monotonic()-started, monitored_worker_peak=peak, reason=reason)
        write(out / 'execution.json', execution)
        assert execution['valid'], execution
    except Exception as exc:
        knowledge(out, 'compute', 'failed', dict(error=repr(exc)))
        raise
    knowledge(out, 'compute', 'complete', dict(execution=execution, worker=worker_record))


if __name__ == '__main__':
    command = sys.argv[1]
    globals()[command](OUT)
