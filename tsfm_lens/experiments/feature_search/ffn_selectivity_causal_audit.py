"""Independent selected-edit/output-reach audit, distinct from full-FFN reach.

The frozen worker already saves every clean/patched first-FFN tensor and forecast.
This companion measures actual interventions without adding native cells or
changing the original384-cell protocol. A zero selected effect is a valid result.
"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.ffn_selectivity_causal import knowledge


def edit_audit(background,clean,patched,units,clean_forecast,patched_forecast):
    """Verify actual patch support and separately quantify edit norm and output reach."""
    background,clean,patched=map(lambda x:np.asarray(x,float),(background,clean,patched))
    mask=np.zeros(clean.shape[-1],bool);mask[units]=True
    if not np.array_equal(patched[...,mask],background[...,mask]):
        raise ValueError('Selected coordinates did not reset to background')
    if not np.array_equal(patched[...,~mask],clean[...,~mask]):
        raise ValueError('Unselected coordinates changed')
    actual=clean-patched
    intended=np.zeros_like(clean);intended[...,mask]=(clean-background)[...,mask]
    if not np.array_equal(actual,intended):
        raise ValueError('Actual unit-delta edit mismatch')
    change=np.asarray(clean_forecast,float)-patched_forecast
    norm=float(np.linalg.norm(actual))
    output_norm=float(np.linalg.norm(change))
    relative=float(np.max(abs(change))/max(1.,float(np.max(abs(clean_forecast)))))
    return dict(actual_selected_edit_L2=norm,actual_selected_edit_RMS=float(np.sqrt(np.mean(actual**2))),
        intended_edit_L2=float(np.linalg.norm(intended)),selected_coordinate_count=len(units),
        selected_edit_nonzero=norm>0,output_change_L2=output_norm,output_change_RMS=float(np.sqrt(np.mean(change**2))),
        output_reach_relative=relative,output_exactly_changed=output_norm>0,output_reach_gt_1e6=relative>1e-6,
        full_FFN_reach_is_not_selected_reach=True)


def audit(out):
    """Reconstruct all selected/matched edits and independently verify signed removal."""
    if (out/'selected_edit_audit.json').exists():
        raise FileExistsError('Refuse repeated selected-edit audit')
    try:
        plan=read(out/'selected_edit_audit_plan.json')
        if plan['auditor_sha256']!=sha(__file__) or plan['protocol_sha256']!=sha(out/'protocol.json'):
            raise ValueError('Prelaunch audit binding changed')
        p=read(out/'protocol.json');rows=read(out/'rows.json');worker=read(out/'bolt_worker.json')
        if worker['arrays_sha256']!=sha(out/'bolt_causal.npz') or worker['interventions_sha256']!=sha(out/'interventions.json'):
            raise ValueError('Saved causal arrays/events changed')
        if worker['parameter_hash_before']!=worker['parameter_hash_after']:
            raise ValueError('Native parameter values changed')
        data=np.load(out/'bolt_causal.npz',allow_pickle=False)
        events=read(out/'interventions.json');analysis=read(out/'analysis.json')
        if len(events)!=len(analysis['events']):
            raise ValueError('Analyzed event count differs')
        records=[];maximum=0.
        for event,scored in zip(events,analysis['events']):
            i,c,j=event['source_index'],event['condition'],event['index']
            record=edit_audit(data['clean_FFN'][i,0],data['clean_FFN'][i,c+1],data['patched_FFN'][j],event['units'],
                data['clean_forecasts'][i,c+1],data['patched_forecasts'][j])
            if not np.isclose(record['actual_selected_edit_L2']**2,event['removed_FFN_energy'],rtol=1e-12,atol=1e-12):
                raise ValueError('Saved selected edit energy mismatch')
            response=data['clean_forecasts'][i,c+1].astype(float)-data['clean_forecasts'][i,0]
            change=data['clean_forecasts'][i,c+1].astype(float)-data['patched_forecasts'][j]
            norm2=float(response@response)
            energy=norm2/len(response)
            if energy>p['response_floor']:
                fraction=float(change@response/norm2)
                maximum=max(maximum,abs(fraction-scored['signed_response_removal']))
                if not np.isclose(fraction,scored['signed_response_removal'],rtol=1e-12,atol=1e-12):
                    raise ValueError('Independent signed removal mismatch')
            elif scored['signed_response_removal'] is not None:
                raise ValueError('Small-response ratio improperly reported')
            records.append(dict(event,**record))
        summaries=[]
        for group,budgets in p['budgets'].items():
            for budget in budgets:
                for arm in ('selected','matched'):
                    for kind in p['condition_types']:
                        found=[r for r in records if (r['group'],r['budget'],r['arm'],r['kind'])==(group,budget,arm,kind)]
                        summaries.append(dict(group=group,budget=budget,arm=arm,kind=kind,n_sources=len(found),
                            edited_sources=sum(r['selected_edit_nonzero'] for r in found),
                            exact_output_change_sources=sum(r['output_exactly_changed'] for r in found),
                            output_reach_gt_1e6_sources=sum(r['output_reach_gt_1e6'] for r in found),
                            actual_edit_L2=[r['actual_selected_edit_L2'] for r in found],
                            output_reach_relative=[r['output_reach_relative'] for r in found]))
        result=dict(valid=True,records=records,summaries=summaries,maximum_signed_removal_error=maximum,
            full_FFN_reach=worker['reach'],parameters_unchanged=True,
            interpretation='Actual selected and matched unit edits, their forecast changes and full-FFN reach are distinct measurements. Full-FFN reach does not establish selected-unit relevance. Zero selected output change is a valid negative result.',
            auditor_sha256=sha(__file__),analysis_sha256=sha(out/'analysis.json'),protocol_sha256=sha(out/'protocol.json'))
        write(out/'selected_edit_audit.json',result)
    except Exception as exc:
        knowledge(out,'selected_edit_audit','failed',dict(error_type=type(exc).__name__,error=str(exc)),
                  'Resolve independent saved-output audit failure; no new native launch.')
        raise
    knowledge(out,'selected_edit_audit','complete',dict(summaries=summaries,maximum_signed_removal_error=maximum,
        selected_edit_audit_sha256=sha(out/'selected_edit_audit.json')),'Report actual edit norms, output reach, removal, energy and off-target changes; no automatic expansion.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal4963000'))
    audit(parser.parse_args().out)
