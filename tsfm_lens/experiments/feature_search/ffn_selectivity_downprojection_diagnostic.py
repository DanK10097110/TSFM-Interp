"""Posthoc local W*removed-FFN-delta energy from cached weights; zero native calls."""
from pathlib import Path
import argparse
import numpy as np
from safetensors import safe_open
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.ffn_selectivity_causal_replication import knowledge


def projected_energy(weight,removed,units):
    """Sparse support gives the exact full downprojection delta without model loading."""
    weight=np.asarray(weight,float);removed=np.asarray(removed,float)
    outside=np.ones(removed.shape[-1],bool);outside[units]=False
    if np.any(removed[...,outside]!=0):
        raise ValueError('Removed delta has unregistered support')
    value=removed[...,units]@weight[:,units].T
    return float(np.sum(value**2))


def run(out):
    """Describe local projected energy; cannot change controls, primary or forecasts."""
    if (out/'downprojection_diagnostic.json').exists():
        raise FileExistsError('Refuse repeated saved-weight diagnostic')
    try:
        p=read(out/'protocol.json');checkpoint=p['checkpoint']
        weight_path=Path(checkpoint['path'])/next(iter(checkpoint['weights']))
        if sha(weight_path)!=checkpoint['weights'][weight_path.name]:
            raise ValueError('Frozen checkpoint changed')
        tensor_name=p['capture']['down_projection']+'.weight'
        with safe_open(str(weight_path),framework='np') as handle:
            weight=handle.get_tensor(tensor_name).astype(float)
        if weight.shape!=(512,2048):
            raise ValueError('Bolt first downprojection shape changed')
        with np.load(out/'bolt_causal.npz',allow_pickle=False) as archive:
            data={k:archive[k] for k in ('clean_FFN','patched_FFN')}
        events=read(out/'interventions.json');rows=read(out/'rows.json');assignment=read(out/'controls.json')
        records=[]
        for event in events:
            i,c,j=event['source_index'],event['condition'],event['index']
            removed=data['clean_FFN'][i,c+1].astype(float)-data['patched_FFN'][j]
            energy=projected_energy(weight,removed,event['units'])
            input_energy=float(np.sum(removed**2))
            records.append(dict(source=event['source'],kind=event['kind'],arm=event['arm'],
                actual_FFN_edit_energy=input_energy,downprojection_delta_energy=energy,
                squared_local_gain=energy/input_energy if input_energy>0 else None))
        lookup={(r['source'],r['kind'],r['arm']):r for r in records}
        summaries=[]
        for label,kinds in {**{k:[k] for k in p['types']},'target':p['targets'],'off_target':p['off_targets']}.items():
            for metric in ('actual_FFN_edit_energy','downprojection_delta_energy'):
                matrix=np.asarray([[np.mean([lookup[row['source'],kind,arm][metric] for kind in kinds])
                    for arm in p['arms']] for row in rows])
                selected=matrix[:,0];controls=matrix[:,1:];mean_control=controls.mean(axis=1)
                rng=np.random.default_rng(4964199)
                indices=rng.integers(len(rows),size=(3999,len(rows)))
                difference=selected-mean_control
                summaries.append(dict(types=label,metric=metric,n_sources=len(rows),selected_mean=float(selected.mean()),
                    mean_control=float(mean_control.mean()),individual_control_means=controls.mean(axis=0).tolist(),
                    paired_difference_mean=float(difference.mean()),paired_ci95=np.quantile(difference[indices].mean(axis=1),[.025,.975]).tolist(),
                    selected_to_mean_control_ratio=float(selected.mean()/mean_control.mean()) if mean_control.mean()>0 else None))
        units=assignment['selected']+list({u for group in assignment['controls'] for u in group})
        result=dict(scope='Posthoc saved-weight local linear diagnostic; no extra native calls, no changes to frozen controls or primary',
            tensor=tensor_name,shape=list(weight.shape),checkpoint_sha256=sha(weight_path),
            unit_column_L2={str(u):float(np.linalg.norm(weight[:,u])) for u in units},records=records,summaries=summaries,
            limitation='W*delta is the immediate downprojection contribution before downstream computation. It does not by itself measure final forecast importance. FIT input activity matching need not match this projected energy.',
            source_sha256=sha(__file__),native_calls=0)
        write(out/'downprojection_diagnostic.json',result)
    except Exception as exc:
        knowledge(out,'saved_weight_diagnostic','failed',dict(error_type=type(exc).__name__,error=str(exc)),
            'Preserve diagnostic failure; no native rerun or control changes.')
        raise
    knowledge(out,'saved_weight_diagnostic','complete',dict(summaries=summaries,unit_column_L2=result['unit_column_L2'],
        diagnostic_sha256=sha(out/'downprojection_diagnostic.json'),native_calls=0),
        'Report posthoc projected-energy qualification with frozen causal results; close branch.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_causal_replication4964000'))
    run(parser.parse_args().out)
