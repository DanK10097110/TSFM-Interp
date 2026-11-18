"""Independent source aggregation audit and compact saved-only methodology report."""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.dictionary_smoothness_study import read,write,sha,knowledge


def report(out):
    """Audit all scalar/source summaries without rerunning dictionaries or native models."""
    if (out/'compact_record.json').exists():
        raise FileExistsError('Refuse repeated smoothness report')
    try:
        p=read(out/'protocol.json');execution=read(out/'execution.json');w=read(out/'worker.json')
        if not execution['valid'] or not w['valid'] or execution['seconds']>180 or w['peak_rss']>2*2**30:
            raise ValueError('Saved-only compute resource/gate failure')
        if w['protocol_sha256']!=sha(out/'protocol.json') or w['results_sha256']!=sha(out/'results.json') or w['replay_sha256']!=sha(out/'independent_replay.json'):
            raise ValueError('Saved-only result binding failed')
        for path,digest in {**p['bindings'],**p['code_hashes']}.items():
            if sha(path)!=digest:
                raise ValueError('Frozen external/source file changed')
        paths=read(Path(p['response'])/'paths.json');results=read(out/'results.json');replay=read(out/'independent_replay.json')
        ids=sorted({v['source'] for v in paths});compact=[];checked=0;maximum=0.;linear_max=0.;decomposition_max=0.
        for arm in results['arms']:
            if sha(out/arm['metrics_file'])!=arm['sha256']:
                raise ValueError('Path metrics changed')
            with np.load(out/arm['metrics_file'],allow_pickle=False) as f:
                values={key:f[key] for key in f.files}
            if any(v.shape!=(768,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError('Invalid metric geometry/finite values')
            for row in arm['summaries']:
                for metric,saved in row['metrics'].items():
                    totals={s:0. for s in ids};counts={s:0 for s in ids}
                    for index,path in enumerate(paths):
                        if path['axis']==row['axis'] and path['radius']==row['radius']:
                            totals[path['source']]+=float(values[metric][index]);counts[path['source']]+=1
                    if any(count!=8 for count in counts.values()) or row['n_sources']!=16 or row['source_ids']!=ids:
                        raise ValueError('Independent-source grouping/count failed')
                    independent=np.asarray([totals[s]/counts[s] for s in ids])
                    error=float(np.max(abs(independent-saved['per_source'])));maximum=max(maximum,error)
                    if not np.allclose(independent,saved['per_source'],atol=1e-12,rtol=1e-12) or not np.isclose(independent.sum()/16,saved['mean'],atol=1e-12,rtol=1e-12):
                        raise ValueError('Independent source mean failed')
                    checked+=1
            if arm['method'] in ('identity','pca_fixed'):
                linear_max=max(linear_max,float(values['isolated_dictionary_chord_RMS'].max()))
                if values['isolated_dictionary_chord_RMS'].max()>1e-10 or np.any(values['chord_support_turnover']!=0):
                    raise ValueError('Linear-control isolated-chord gate failed')
            decomposition_max=max(decomposition_max,float(values['chord_decomposition_error'].max()))
            overall={}
            for metric,v in values.items():
                source_means=np.asarray([v[[i for i,path in enumerate(paths) if path['source']==s]].mean() for s in ids])
                overall[metric]=dict(mean=float(source_means.mean()),median=float(np.median(source_means)),per_source=source_means.tolist())
            conditional={}
            for label,mask in (('switch',values['any_chord_support_switch']>0),('no_switch',values['any_chord_support_switch']==0)):
                source_values=[]
                for s in ids:
                    index=[i for i,path in enumerate(paths) if path['source']==s and mask[i]]
                    if index:
                        source_values.append(float(values['isolated_dictionary_chord_RMS'][index].mean()))
                conditional[label]=dict(n_sources_with_paths=len(source_values),mean_isolated_RMS=None if not source_values else float(np.mean(source_values)))
            compact.append(dict(model=arm['model'],policy=arm['policy'],method=arm['method'],k=arm['k'],
                n_sources=16,overall=overall,by_axis_radius=arm['summaries'],support_switch_conditioning=conditional))
        if len(compact)!=36 or len(replay['arms'])!=36 or not all(r['valid'] for r in replay['arms']):
            raise ValueError('Incomplete fixed design/replay')
        audit=dict(valid=True,arms=36,source_metric_summaries_recomputed=checked,maximum_source_mean_error=maximum,
            linear_control_maximum_isolated_RMS=linear_max,maximum_chord_decomposition_error=decomposition_max,
            independent_replay='Canonical frozen encoder+dense decoder on first path/source, all36 arms, bothprecisions; exact canonical repeats',
            max_replay_error={precision:max(a['precision'][precision]['max_abs_error'] for a in replay['arms']) for precision in ('FP32','FP64')},
            external_frozen_bindings_unchanged=True,results_sha256=sha(out/'results.json'),reporter_sha256=sha(__file__))
        write(out/'aggregation_audit.json',audit)
        result=dict(status='COMPLETE_SAVED_ONLY_AUDITED',question=p['isolated'],execution=execution,audit=audit,arms=compact,
            interpretation_rules=['Only actual_input_midpoint results concern the saved native input paths',
                'Activation-interpolation chord isolates dictionary nonlinearity off manifold; it is not a new native path',
                'Adaptive signed PCA TopK is not fixed rank-k PCA projection',
                'Inspect source aggregates, reconstruction error, secant retention, support changes and precision/refloor controls jointly',
                'Nested is the frozen simplified shared-code two-prefix recipe, not a general Matryoshka result'],
            native_calls=0,fit_calls=0,verification_failure_preserved=read(out/'verification_failure.json'),
            next_decision='Report measured method-induced versus native-state roughness with qualifications; no unrequested fitting/native expansion.')
        write(out/'compact_record.json',result)
    except Exception as exc:
        knowledge(out,'aggregation_audit','failed',dict(error_type=type(exc).__name__,error=str(exc)),
            'Resolve preserved saved-array audit failure; no native or fitting.')
        raise
    knowledge(out,'report','complete',dict(audit=audit,execution=execution,arms=compact,compact_record_sha256=sha(out/'compact_record.json'),native_calls=0,fit_calls=0),result['next_decision'])
    print(dict(status=result['status'],audit=audit),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/dictionary_smoothness4965000'))
    report(parser.parse_args().out)
