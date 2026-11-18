"""Type-specific causal response removal and energy/off-target accounting.

No native calls. Exact patch support and saved hashes are checked before scoring.
Fresh k4 is the frozen source-paired contrast; reused TEST is exploratory.
"""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.ffn_selectivity_causal import read,write,sha,knowledge,patch_units,response_metrics,TYPES


def summarize(values,ids):
    """Bootstrap one value per source; keep small-response exclusions explicit."""
    eligible=[i for i,x in enumerate(values) if x is not None and np.isfinite(x)]
    if not eligible:
        return dict(n_sources=0,source_ids=[],excluded_sources=ids,mean=None,ci95=None)
    x=np.asarray([values[i] for i in eligible],float)
    rng=np.random.default_rng(4963099)
    draws=x[rng.integers(len(x),size=(1999,len(x)))].mean(axis=1)
    return dict(n_sources=len(x),source_ids=[ids[i] for i in eligible],excluded_sources=[v for i,v in enumerate(ids) if i not in eligible],
        mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist() if len(x)>1 else None)


def aggregate(records,kinds,metric):
    """Equal-type average; a normalized endpoint requires all requested types."""
    values=[record[metric] for record in records if record['kind'] in kinds]
    if len(values)!=len(kinds) or any(v is None for v in values):
        return None
    return float(np.mean(values))


def analyze(out):
    """Verify saved intervention support and publish all types/arms without tuning."""
    if (out/'analysis.json').exists():
        raise FileExistsError('Refuse repeated causal analysis')
    try:
        p=read(out/'protocol.json');rows=read(out/'rows.json');worker=read(out/'bolt_worker.json')
        for path,digest in p['sources'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen source changed')
        for name,digest in p['hashes'].items():
            if sha(out/name)!=digest:
                raise ValueError('Frozen artifact changed')
        execution=read(out/'execution.json')['bolt']
        if (not execution['valid'] or execution['returncode']!=0 or not worker['valid'] or worker['cells']!=p['cells']
                or worker['parameter_hash_before']!=worker['parameter_hash_after']
                or worker['protocol_sha256']!=sha(out/'protocol.json')
                or worker['arrays_sha256']!=sha(out/'bolt_causal.npz')
                or worker['interventions_sha256']!=sha(out/'interventions.json')
                or max(execution['monitored_peak'],worker['peak_rss'])>p['memory_cap']
                or max(execution['seconds'],worker['seconds'])>p['seconds_cap']
                or not all(worker[key] for key in ('background_repeat_exact','normalizer_identity_exact','self_patch_exact','selected_only_patch_exact'))):
            raise ValueError('Native causal gate/resource/hash failed')
        if len(worker['reach'])!=len(rows) or not all(r['relative_change']>1e-6 for r in worker['reach']):
            raise ValueError('Full FFN reach failed')
        data=np.load(out/'bolt_causal.npz',allow_pickle=False)
        if data['source_ids'].tolist()!=[r['source'] for r in rows]:
            raise ValueError('Causal source alignment failed')
        events=read(out/'interventions.json');matching=read(out/'matching.json')
        expected={(row['index'],c,k,arm) for row in rows for c in range(5) for k in row['budgets'] for arm in ('selected','matched')}
        if len(events)!=len(expected) or {(e['source_index'],e['condition'],e['budget'],e['arm']) for e in events}!=expected:
            raise ValueError('Causal intervention schedule incomplete')
        metrics=[]
        for event in events:
            i,c,j=event['source_index'],event['condition'],event['index']
            units=matching['selected'][:event['budget']] if event['arm']=='selected' else matching['draws'][rows[i]['control_draw']][:event['budget']]
            if event['units']!=units or event['source']!=rows[i]['source'] or event['kind']!=TYPES[c]:
                raise ValueError('Frozen intervention identity mismatch')
            clean_ffn=data['clean_FFN'][i,c+1];anchor=data['clean_FFN'][i,0]
            patched=data['patched_FFN'][j]
            if not np.array_equal(patch_units(clean_ffn,anchor,units),patched):
                raise ValueError('Independent selected-only patch support failed')
            delta=clean_ffn.astype(float)-anchor
            removed=clean_ffn.astype(float)-patched
            if not np.isclose(np.sum(removed**2),event['removed_FFN_energy'],rtol=1e-12,atol=1e-12):
                raise ValueError('Removed FFN energy mismatch')
            values=response_metrics(data['clean_forecasts'][i,0],data['clean_forecasts'][i,c+1],data['patched_forecasts'][j])
            metrics.append(dict(event,**values))
        summary=[]
        names=['signed_response_removal','removed_output_energy','response_energy','remaining_output_energy',
            'orthogonal_change_ratio','changed_energy_ratio','remaining_energy_ratio','removed_FFN_energy','removed_FFN_share']
        for group,budgets in p['budgets'].items():
            sources=[r for r in rows if r['group']==group]
            ids=[r['source'] for r in sources]
            for budget in budgets:
                target=p['target_types'][str(budget)]
                subsets={**{kind:[kind] for kind in TYPES},'target':target,'off_target':[k for k in TYPES if k not in target]}
                for label,kinds in subsets.items():
                    for metric in names:
                        arms={arm:[aggregate([e for e in metrics if e['source']==row['source'] and e['budget']==budget and e['arm']==arm],kinds,metric)
                                   for row in sources] for arm in ('selected','matched')}
                        paired=[None if a is None or b is None else a-b for a,b in zip(arms['selected'],arms['matched'])]
                        summary.append(dict(group=group,budget=budget,types=label,metric=metric,
                            selected=summarize(arms['selected'],ids),matched=summarize(arms['matched'],ids),
                            paired_selected_minus_matched=summarize(paired,ids),
                            primary=group=='fresh_confirmation' and budget==4 and label=='target' and metric=='signed_response_removal'))
        result=dict(protocol_sha256=sha(out/'protocol.json'),events=metrics,summaries=summary,
            primary=[r for r in summary if r['primary']][0],gates=dict(native=True,patch_support_independently_verified=True,parameters_fixed=True),
            interpretation='Signed native forecast-response removal relative to matched controls, not forecast accuracy or general monosemanticity. Positive removal may coexist with collateral off-target/orthogonal changes.',
            limitations='Small source cohorts; reused TEST exploratory; fresh confirmation restricted to chosen Bolt-first k4 and assigned matched draws; one balanced variant/type/source; no additional models or sites.')
        write(out/'analysis.json',result)
    except Exception as exc:
        knowledge(out,'analysis','failed',dict(error_type=type(exc).__name__,error=str(exc)),'Resolve saved-output audit failure; no new native launch.')
        raise
    knowledge(out,'analysis','complete',dict(primary=result['primary'],summaries=summary,gates=result['gates'],analysis_sha256=sha(out/'analysis.json')),
        'Report response removal, energy and off-target changes before any expansion.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal4963000'))
    analyze(parser.parse_args().out)
