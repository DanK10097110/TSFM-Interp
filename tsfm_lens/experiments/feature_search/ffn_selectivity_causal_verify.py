"""Independent saved-array recomputation of every causal source statistic and CI."""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.ffn_selectivity_causal import knowledge


def verify(out):
    """Recompute energies, response geometry and all source-paired bootstrap intervals."""
    if (out/'independent_statistics_audit.json').exists():
        raise FileExistsError('Refuse repeated statistics audit')
    try:
        p=read(out/'protocol.json');rows=read(out/'rows.json');analysis=read(out/'analysis.json')
        selected_audit=read(out/'selected_edit_audit.json')
        if not selected_audit['valid'] or selected_audit['analysis_sha256']!=sha(out/'analysis.json'):
            raise ValueError('Selected edit audit binding failed')
        arrays=np.load(out/'bolt_causal.npz',allow_pickle=False)
        calculated={};max_error=0.;precision=[]
        for event in analysis['events']:
            i,c,j=event['source_index'],event['condition'],event['index']
            background=arrays['clean_forecasts'][i,0].astype(float)
            clean=arrays['clean_forecasts'][i,c+1].astype(float)
            changed=arrays['patched_forecasts'][j].astype(float)
            response=clean-background;removed=clean-changed
            denominator=float(response@response)
            norm=float(removed@removed)
            valid=denominator/len(response)>p['response_floor']
            fraction=float(removed@response/denominator) if valid else None
            ffn_delta=arrays['clean_FFN'][i,c+1].astype(float)-arrays['clean_FFN'][i,0]
            actual=arrays['clean_FFN'][i,c+1].astype(float)-arrays['patched_FFN'][j]
            unit_energy=float(np.linalg.norm(actual.ravel())**2)
            full_energy=float(np.linalg.norm(ffn_delta.ravel())**2)
            values=dict(response_energy=denominator/len(response),removed_output_energy=norm/len(response),
                remaining_output_energy=float((changed-background)@(changed-background)/len(response)),
                signed_response_removal=fraction,changed_energy_ratio=norm/denominator if valid else None,
                remaining_energy_ratio=float((changed-background)@(changed-background)/denominator) if valid else None,
                orthogonal_change_ratio=float((removed-fraction*response)@(removed-fraction*response)/denominator) if valid else None,
                removed_FFN_energy=unit_energy,removed_FFN_share=unit_energy/full_energy if full_energy>0 else None)
            remaining32=np.mean((arrays['patched_forecasts'][j]-arrays['clean_forecasts'][i,0])**2)
            energy64=denominator/len(response)
            precision.append(dict(index=j,remaining_output_energy_float64=values['remaining_output_energy'],
                remaining_energy_ratio_float64=values['remaining_energy_ratio'],
                remaining_energy_absolute_discrepancy=abs(values['remaining_output_energy']-float(remaining32)),
                remaining_ratio_absolute_discrepancy=abs(values['remaining_energy_ratio']-float(remaining32/energy64)) if valid else None))
            values['remaining_output_energy']=float(remaining32)
            values['remaining_energy_ratio']=float(remaining32/energy64) if valid else None
            for key,value in values.items():
                if value is None:
                    if event[key] is not None:
                        raise ValueError('Undefined ratio was reported')
                elif not np.isclose(value,event[key],rtol=1e-10,atol=1e-12):
                    raise ValueError('Independent event statistic failed: '+key)
                else:
                    max_error=max(max_error,abs(value-event[key]))
            calculated[(event['source'],event['budget'],event['arm'],event['kind'])]=values
        checked=0;interval_error=0.
        for item in analysis['summaries']:
            sources=[r['source'] for r in rows if r['group']==item['group']]
            targets=p['target_types'][str(item['budget'])]
            kinds=targets if item['types']=='target' else ([k for k in p['condition_types'] if k not in targets]
                  if item['types']=='off_target' else [item['types']])
            pairs=[]
            for source in sources:
                pair=[]
                for arm in ('selected','matched'):
                    values=[calculated[source,item['budget'],arm,kind][item['metric']] for kind in kinds]
                    pair.append(None if any(v is None for v in values) else sum(values)/len(values))
                pairs.append(pair)
            arms=dict(selected=[v[0] for v in pairs],matched=[v[1] for v in pairs],
                paired_selected_minus_matched=[None if a is None or b is None else a-b for a,b in pairs])
            for arm,values in arms.items():
                indices=[i for i,v in enumerate(values) if v is not None and np.isfinite(v)]
                record=item[arm]
                if record['source_ids']!=[sources[i] for i in indices] or record['excluded_sources']!=[s for i,s in enumerate(sources) if i not in indices]:
                    raise ValueError('Bootstrap source membership mismatch')
                if record['n_sources']!=len(indices):
                    raise ValueError('Bootstrap denominator mismatch')
                if not indices:
                    if record['mean'] is not None or record['ci95'] is not None:
                        raise ValueError('Empty source summary reported')
                else:
                    values=np.asarray([values[i] for i in indices])
                    rng=np.random.default_rng(4963099)
                    bootstrap=values[rng.integers(0,len(values),size=(1999,len(values)))].sum(axis=1)/len(values)
                    expected_mean=float(sum(values)/len(values))
                    if not np.isclose(expected_mean,record['mean'],atol=1e-12,rtol=1e-10):
                        raise ValueError('Independent source mean failed')
                    if len(values)>1:
                        expected=np.percentile(bootstrap,[2.5,97.5])
                        interval_error=max(interval_error,float(np.max(abs(expected-record['ci95']))))
                        if not np.allclose(expected,record['ci95'],atol=1e-12,rtol=1e-10):
                            raise ValueError('Independent source bootstrap CI failed')
                    elif record['ci95'] is not None:
                        raise ValueError('Singleton CI improperly reported')
                checked+=1
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Original frozen source changed')
        result=dict(valid=True,events_recomputed=len(calculated),source_summaries_recomputed=checked,
            maximum_event_metric_error=max_error,maximum_CI_error=interval_error,
            precision_diagnosis=read(out/'precision_audit_diagnosis.json'),
            remaining_energy_arithmetic='Frozen float32 secondary arithmetic independently reproduced; float64 values preserved separately; primary signed removal remains float64',
            protocol_sha256=sha(out/'protocol.json'),analysis_sha256=sha(out/'analysis.json'),verifier_sha256=sha(__file__),
            selected_edit_audit_sha256=sha(out/'selected_edit_audit.json'),native_slot_released=read(out/'slot_release.json')['released'])
        write(out/'independent_statistics_audit.json',result)
        write(out/'remaining_energy_float64.json',dict(scope='Supplemental precision audit; original analysis unchanged',records=precision))
    except Exception as exc:
        knowledge(out,'independent_statistics_audit','failed',dict(error_type=type(exc).__name__,error=str(exc)),
                  'Resolve recorded saved-output audit failure; no native expansion.')
        raise
    knowledge(out,'independent_statistics_audit','complete',result,'Report audited causal results, energy and off-target qualifications; no automatic expansion.')
    print(result,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal4963000'))
    verify(parser.parse_args().out)
