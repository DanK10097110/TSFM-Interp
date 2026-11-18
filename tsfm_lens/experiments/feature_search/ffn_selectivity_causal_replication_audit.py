"""Independent crossed-control audit and optional saved-weight projection diagnostic.

No model is instantiated. All edit/output arithmetic is reconstructed from saved
arrays. The optional local linear W*delta diagnostic is post hoc and cannot
change the frozen primary, units, controls or forecasts.
"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.ffn_selectivity_causal_replication import knowledge


def source_statistics(values,ids,valid):
    """Independently reproduce source means and the fixed3999-draw intervals."""
    indices=np.flatnonzero(valid)
    if not len(indices):
        return dict(n_sources=0,source_ids=[],excluded_sources=ids,mean=None,ci95=None)
    x=np.asarray(values)[indices]
    rng=np.random.default_rng(4964099)
    means=np.sum(x[rng.integers(0,len(x),size=(3999,len(x)))],axis=1)/len(x)
    return dict(n_sources=len(indices),source_ids=[ids[i] for i in indices],excluded_sources=[s for i,s in enumerate(ids) if not valid[i]],
        mean=float(np.sum(x)/len(x)),ci95=np.percentile(means,[2.5,97.5]).tolist() if len(x)>1 else None)


def check_statistics(actual,reported):
    """Check identities/counts and all numeric source uncertainty values."""
    for key in ('n_sources','source_ids','excluded_sources'):
        if actual[key]!=reported[key]:
            raise ValueError('Source bootstrap identities/counts differ')
    for key in ('mean','ci95'):
        if actual[key] is None:
            if reported[key] is not None:
                raise ValueError('Undefined source statistic reported')
        elif not np.allclose(actual[key],reported[key],rtol=1e-10,atol=1e-12):
            raise ValueError('Independent source statistic differs: '+key)


def reconstruct(data,event,units):
    """Independently reconstruct patch support, edit norm, output reach and energies."""
    i,c,j=event['source_index'],event['condition'],event['index']
    base=data['clean_FFN'][i,0].astype(float);clean=data['clean_FFN'][i,c+1].astype(float);patched=data['patched_FFN'][j].astype(float)
    mask=np.zeros(clean.shape[-1],bool);mask[units]=True
    if not np.array_equal(patched[:,mask],base[:,mask]) or not np.array_equal(patched[:,~mask],clean[:,~mask]):
        raise ValueError('Independent selected-only edit support failed')
    removed=clean-patched;input_delta=clean-base
    edit_energy=float(removed.ravel()@removed.ravel());total_energy=float(input_delta.ravel()@input_delta.ravel())
    background=data['clean_forecasts'][i,0].astype(float);prediction=data['clean_forecasts'][i,c+1].astype(float)
    changed=data['patched_forecasts'][j].astype(float)
    response=prediction-background;change=prediction-changed;remaining=changed-background
    energy=float(response@response);change_energy=float(change@change);remaining_energy=float(remaining@remaining)
    valid=energy/len(response)>1e-8
    fraction=float(change@response/energy) if valid else None
    orthogonal=change-fraction*response if valid else None
    return dict(response_energy=energy/len(response),removed_output_energy=change_energy/len(response),
        remaining_output_energy=remaining_energy/len(response),signed_response_removal=fraction,
        remaining_energy_ratio=remaining_energy/energy if valid else None,changed_energy_ratio=change_energy/energy if valid else None,
        orthogonal_change_ratio=float(orthogonal@orthogonal/energy) if valid else None,
        removed_FFN_energy=edit_energy,removed_FFN_share=edit_energy/total_energy if total_energy>0 else None,
        actual_selected_edit_L2=float(np.sqrt(edit_energy)),output_change_L2=float(np.sqrt(change_energy)),
        output_reach_relative=float(np.max(abs(change))/max(1.,float(np.max(abs(prediction))))))


def audit(out):
    """Audit every event and crossed-control summary without calling frozen scorers."""
    if (out/'independent_audit.json').exists():
        raise FileExistsError('Refuse repeated replication audit')
    try:
        p=read(out/'protocol.json');a=read(out/'analysis.json');rows=read(out/'rows.json');assignment=read(out/'controls.json')
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen replication source changed')
        for name,digest in p['hashes'].items():
            if sha(out/name)!=digest:
                raise ValueError('Frozen input changed')
        amendment=read(out/'execution_amendment.json');w=read(out/'bolt_worker.json');e=read(out/'execution.json')['bolt']
        if (not e['valid'] or e['returncode']!=0 or w['cells']!=1088 or e['cells']!=1088
                or max(e['seconds'],w['seconds'])>300 or max(e['monitored_peak'],w['peak_rss'])>3*2**30
                or e['execution_amendment_sha256']!=sha(out/'execution_amendment.json')
                or amendment['base_protocol_sha256']!=sha(out/'protocol.json')
                or w['parameter_hash_before']!=w['parameter_hash_after']
                or w['arrays_sha256']!=sha(out/'bolt_causal.npz') or w['events_sha256']!=sha(out/'interventions.json')):
            raise ValueError('Native capture/amended cap/parameter audit failed')
        if len(w['reach'])!=32 or not all(v['relative_change']>1e-6 for v in w['reach']):
            raise ValueError('Full-FFN reach audit failed')
        with np.load(out/'bolt_causal.npz',allow_pickle=False) as archive:
            data={key:archive[key] for key in archive.files}
        if data['source_ids'].tolist()!=[r['source'] for r in rows]:
            raise ValueError('Capture source identities differ')
        lookup={};max_error=0.;reach=[]
        for event in a['events']:
            units=assignment['selected'] if event['arm']=='selected' else assignment['controls'][int(event['arm'][-1])]
            if event['units']!=units:
                raise ValueError('Frozen selected/control identity changed')
            values=reconstruct(data,event,units)
            for key,value in values.items():
                if value is None:
                    if event[key] is not None:
                        raise ValueError('Undefined response ratio reported')
                else:
                    max_error=max(max_error,abs(value-event[key]))
                    if not np.isclose(value,event[key],rtol=1e-10,atol=1e-12):
                        raise ValueError('Independent event arithmetic differs: '+key)
            lookup[(event['source'],event['kind'],event['arm'])]=values
            reach.append(dict(source=event['source'],kind=event['kind'],arm=event['arm'],
                actual_edit_L2=values['actual_selected_edit_L2'],output_reach_relative=values['output_reach_relative']))
        ids=[r['source'] for r in rows]
        if len(lookup)!=800 or len(a['events'])!=800:
            raise ValueError('Wrong crossed event count')
        count=0;ci_error=0.
        for summary in a['summaries']:
            kinds=p['targets'] if summary['types']=='target' else p['off_targets'] if summary['types']=='off_target' else [summary['types']]
            matrix=np.full((32,5),np.nan)
            for i,source in enumerate(ids):
                for j,arm in enumerate(p['arms']):
                    values=[lookup[source,kind,arm][summary['metric']] for kind in kinds]
                    if all(v is not None for v in values):
                        matrix[i,j]=sum(values)/len(values)
            valid=np.isfinite(matrix).all(axis=1);controls=matrix[:,1:];mean_control=controls.sum(axis=1)/4
            comparisons=[('selected_absolute',matrix[:,0]),('mean_control',mean_control),('paired_primary',matrix[:,0]-mean_control)]
            for key,values in comparisons:
                expected=source_statistics(values,ids,valid);check_statistics(expected,summary[key]);count+=1
                if expected['ci95'] is not None:
                    ci_error=max(ci_error,float(np.max(abs(np.asarray(expected['ci95'])-summary[key]['ci95']))))
            for j in range(4):
                check_statistics(source_statistics(controls[:,j],ids,valid),summary['individual_controls'][j]);count+=1
            spread=summary['control_spread']
            for expected,key in ((np.std(controls,axis=1,ddof=1),'per_source_sd'),(np.min(controls,axis=1),'per_source_min'),(np.max(controls,axis=1),'per_source_max')):
                if not np.allclose(expected,np.asarray(spread[key],float),atol=1e-12,rtol=1e-10,equal_nan=True):
                    raise ValueError('Control spread audit failed')
            if summary['metric']=='signed_response_removal':
                for name in ('selected_absolute','paired_primary'):
                    interval=summary[name]['ci95']
                    contained=interval is not None and interval[0]>=-.02 and interval[1]<=.02
                    if summary['practical_check'][name]['CI_contained']!=contained:
                        raise ValueError('Practical bound misapplied')
            elif 'practical_check' in summary:
                raise ValueError('Two-percent response bound applied to another metric')
        result=dict(valid=True,events_independently_recomputed=800,source_summaries_independently_recomputed=count,
            maximum_event_error=max_error,maximum_CI_error=ci_error,actual_edit_output_reach=reach,
            full_FFN_reach=w['reach'],parameters_unchanged=True,worker_300s_cap_verified=True,
            protocol_sha256=sha(out/'protocol.json'),analysis_sha256=sha(out/'analysis.json'),auditor_sha256=sha(__file__))
        write(out/'independent_audit.json',result)
    except Exception as exc:
        knowledge(out,'independent_audit','failed',dict(error_type=type(exc).__name__,error=str(exc)),'Preserve failed audit; resolve from saved outputs only.')
        raise
    knowledge(out,'independent_audit','complete',dict(events=800,summaries=count,maximum_event_error=max_error,maximum_CI_error=ci_error,
        independent_audit_sha256=sha(out/'independent_audit.json')),'Report unchanged primary, absolute effect, control spread, energy/off-target diagnostics; no expansion.')
    print(dict(valid=True,events=800,summaries=count,maximum_event_error=max_error,maximum_CI_error=ci_error),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal_replication4964000'))
    audit(parser.parse_args().out)
