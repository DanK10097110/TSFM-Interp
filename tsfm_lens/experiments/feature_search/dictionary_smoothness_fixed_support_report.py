"""Independent saved-result arithmetic and source audit for the closed gate-control child."""
import time
import numpy as np
from experiments.feature_search.dictionary_smoothness_fixed_support import OUT, read, write, sha, knowledge


def sample_metrics(h, g, hard, reference):
    """Direct per-path vector norms, independently of the worker metric helpers."""
    answer = {k:[] for k in ('raw_chord_RMS', 'decoded_actual_chord_RMS', 'isolated_dictionary_chord_RMS',
                            'state_reconstruction_RMS', 'response_reconstruction_RMS',
                            'state_change_from_hard_RMS', 'response_change_from_hard_RMS',
                            'state_reconstruction_relative_RMS', 'response_reconstruction_relative_RMS')}
    for index in range(h.shape[1]):
        x, y, baseline = h[:,index], g[:,index], hard[:,index]
        norm = lambda value: float(np.linalg.norm(value.ravel()) / np.sqrt(value.size))
        raw_response = x[:2] - x[2]
        decoded_response = y[:2] - y[2]
        response_error = norm(decoded_response - raw_response)
        state_error = norm(y[:3] - x[:3])
        values = dict(raw_chord_RMS=norm(x[0]+x[1]-2*x[2]),
            decoded_actual_chord_RMS=norm(y[0]+y[1]-2*y[2]),
            isolated_dictionary_chord_RMS=norm(y[0]+y[1]-2*y[3]),
            state_reconstruction_RMS=state_error,
            response_reconstruction_RMS=response_error,
            state_change_from_hard_RMS=norm(y[:3]-baseline[:3]),
            response_change_from_hard_RMS=norm(decoded_response-(baseline[:2]-baseline[2])),
            state_reconstruction_relative_RMS=state_error/reference[index],
            response_reconstruction_relative_RMS=response_error/max(norm(raw_response), np.finfo(np.float32).eps*reference[index],1e-30))
        for key,value in values.items():
            answer[key].append(value)
    return answer


def report(out):
    """Audit bindings, direct saved-array metrics and independent-source summaries."""
    assert not (out/'compact_summary.json').exists()
    started = time.monotonic()
    try:
        p = read(out/'protocol.json')
        execution = read(out/'execution.json')
        worker = read(out/'worker.json')
        assert execution['valid'] and worker['valid']
        assert execution['seconds'] <= 60 and worker['peak_rss'] <= 2*2**30
        assert worker['protocol_sha256'] == sha(out/'protocol.json')
        assert worker['results_sha256'] == sha(out/'results.json')
        assert worker['replay_sha256'] == sha(out/'independent_replay.json')
        for file,digest in {**p['bindings'],**p['code_hashes']}.items():
            assert sha(file) == digest, file
        assert read(out/'tests.json')['valid']
        plants = read(out/'plants.json')
        assert plants['valid'] and plants['complete'] and len(plants['results']) == 4
        arms = read(out/'results.json')['arms']
        replay = read(out/'independent_replay.json')
        assert len(arms) == 18 and len(replay['arms']) == 18 and replay['valid']
        paths = read(__import__('pathlib').Path(p['response'])/'paths.json')
        ids = sorted({x['source'] for x in paths})
        sample = [next(i for i,x in enumerate(paths) if x['source']==s) for s in ids]
        tables = []
        checked = 0
        maximum_summary_error = 0.
        maximum_metric_error = 0.
        maximum_affine_chord = 0.
        maximum_live_relu_chord = 0.
        maximum_control_difference = 0.
        paired = []
        for model in ('bolt','timesfm'):
            native = np.load(__import__('pathlib').Path(p['response'])/f'{model}_locked_hidden_0.npy',mmap_mode='r')
            reference = np.asarray([np.linalg.norm(native[paths[i]['base']].astype(float).ravel())/np.sqrt(native[0].size) for i in sample])
            expected = np.stack([native[[paths[i][key] for i in sample]].astype(float) for key in ('minus','plus','midpoint')])
            expected = np.concatenate([expected, ((expected[0]+expected[1])/2)[None]])
            for method in ('pca_adaptive','topk','nested'):
                hard = np.load(out/f'{model}_{method}_hard_sample.npz')['FP64']
                hard_metrics = dict(np.load(out/f'{model}_{method}_hard_metrics.npz'))
                fixed = np.load(out/f'{model}_{method}_fixed_support_sample.npz')['FP64']
                affine = np.load(out/f'{model}_{method}_fixed_affine_sample.npz')['FP64']
                maximum_control_difference=max(maximum_control_difference,float(np.max(abs(fixed-affine))))
                rows = [a for a in arms if a['model']==model and a['method']==method]
                for arm in rows:
                    mode=arm['mode']
                    for kind in ('metrics','sample','gates'):
                        assert sha(out/arm[kind+'_file'])==arm[kind+'_sha256']
                    values = dict(np.load(out/arm['metrics_file']))
                    saved = dict(np.load(out/arm['sample_file']))
                    assert np.array_equal(saved['h'],expected), 'Saved sample provenance'
                    direct = sample_metrics(saved['h'], saved['FP64'], hard, reference)
                    for key,calculated in direct.items():
                        maximum_metric_error=max(maximum_metric_error,float(np.max(abs(values[key][sample]-calculated))))
                        np.testing.assert_allclose(values[key][sample],calculated,atol=1e-12,rtol=1e-10)
                    for key,val in values.items():
                        assert val.shape==(768,) and np.isfinite(val).all()
                        if key.endswith('_change_from_hard'):
                            base=key.removesuffix('_change_from_hard')
                            np.testing.assert_array_equal(val,values[base]-hard_metrics[base])
                        groups=[(arm['overall'][key],list(range(768)))]
                        groups.extend((row['metrics'][key],[i for i,x in enumerate(paths) if x['axis']==row['axis'] and x['radius']==row['radius']]) for row in arm['by_axis_radius'])
                        for summary,indices in groups:
                            sums={s:0. for s in ids};counts={s:0 for s in ids}
                            for index in indices:
                                source=paths[index]['source'];sums[source]+=float(val[index]);counts[source]+=1
                            assert len(set(counts.values()))==1 and next(iter(counts.values())) in (8,48)
                            means=np.array([sums[s]/counts[s] for s in ids])
                            maximum_summary_error=max(maximum_summary_error,float(np.max(abs(means-summary['per_source']))))
                            np.testing.assert_allclose(means,summary['per_source'],atol=1e-12,rtol=1e-10)
                            np.testing.assert_allclose(means.sum()/16,summary['mean'],atol=1e-12,rtol=1e-10)
                            checked+=1
                    assert values['central_output_change_from_hard_RMS'].max()<1e-12
                    if mode=='fixed_affine':
                        maximum_affine_chord=max(maximum_affine_chord,float(values['isolated_dictionary_chord_RMS'].max()))
                        assert maximum_affine_chord<1e-10
                    if mode=='fixed_support':
                        maximum_live_relu_chord=max(maximum_live_relu_chord,float(values['isolated_dictionary_chord_RMS'].max()))
                    keys=('decoded_actual_chord_RMS','isolated_dictionary_chord_RMS','state_reconstruction_RMS',
                          'state_reconstruction_relative_RMS','response_reconstruction_RMS','response_reconstruction_relative_RMS',
                          'state_reconstruction_RMS_change_from_hard','response_reconstruction_RMS_change_from_hard',
                          'state_change_from_hard_RMS','response_change_from_hard_RMS')
                    tables.append(dict(model=model,method=method,mode=mode,**{k:arm['overall'][k]['mean'] for k in keys}))
                    if mode!='hard':
                        paired.append(dict(model=model,method=method,mode=mode,
                            **{key:dict(mean=arm['overall'][key]['mean'],sources_negative=int(np.sum(np.asarray(arm['overall'][key]['per_source'])<0)),n_sources=16)
                               for key in ('decoded_actual_chord_RMS_change_from_hard','state_reconstruction_RMS_change_from_hard','response_reconstruction_RMS_change_from_hard')}))
                del hard, fixed, affine
        audit=dict(valid=True, independent_source_summaries=checked, maximum_summary_error=maximum_summary_error,
            maximum_sample_metric_error=maximum_metric_error, maximum_affine_isolated_RMS=maximum_affine_chord,
            maximum_live_relu_isolated_RMS=maximum_live_relu_chord,
            maximum_sample_live_vs_fixed_relu_output_difference=maximum_control_difference,
            independent_replay='18 arms,first registered path per16 sources,both precisions,exact reference repeats; all-path hard baseline agrees with parent',
            maximum_reference_errors={precision:max(a[precision]['max_abs_error'] for a in replay['arms']) for precision in ('FP32','FP64')},
            FP32_central_index_disagreements=sum(a['central_indices_differ'] for a in replay['arms'][::3]),
            FP32_central_gate_disagreements=sum(a['central_gates_differ'] for a in replay['arms'][::3]),
            frozen_files_unchanged=True,reporter_sha256=sha(__file__),seconds=time.monotonic()-started)
        write(out/'audit.json',audit)
        summary=dict(status='COMPLETE_AUDITED_BRANCH_CLOSED',table_design='Locked k16 seed4951000; equal16-source means over the same768 paths, balanced axes/radii. Response is hidden-state perturbation response, not forecast accuracy.',
            table=tables,paired_source_changes=paired,audit=audit,execution=execution,worker_peak_rss=worker['peak_rss'],
            tests_passed=4,plants_detected=4,native_calls=0,fit_calls=0,
            parent_wording_correction=read(out/'knowledge.json')['parent_wording_correction'],
            interpretation='Read paired state and hidden-response fidelity changes alongside curvature. Fixed gates are path-local diagnostics and cannot establish a better general encoder.',
            branch_closed=True,next_decision='No broader grid, fitting or native follow-up.')
        write(out/'compact_summary.json',summary)
        knowledge(out,'audit','complete',summary)
        print(dict(status=summary['status'],audit=audit,table=tables),flush=True)
    except Exception as exc:
        knowledge(out,'audit','failed',dict(error=repr(exc)))
        raise


if __name__=='__main__':
    report(OUT)
