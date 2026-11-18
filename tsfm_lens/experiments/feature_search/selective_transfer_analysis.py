"""Source-paired transfer summaries and independent counterfactual arithmetic."""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.coherence_analysis import estimate


def run(out):
    """Keep per-domain contrasts, source counts and semantic eligibility explicit."""
    metadata=read(out/'rows.json');histories=np.load(out/'inputs.npz')['history'];source_index={r['source']:i for i,r in enumerate(metadata) if r['case']=='real'};summaries=[];audit=0;cases=[]
    for model in ['bolt','timesfm']:
        cuts=read(out/f'{model}_worker.json')['cuts'];original=np.load(out/f'{model}_states.npz')
        for suffix,period in [('transfer',128/5.35),('transfer_period19p2',19.2)]:
            rows=read(out/f'{model}_{suffix}.json');execution=read(out/f'{model}_{suffix}_execution.json')
            if sha(out/f'{model}_{suffix}.json')!=execution['sha256']:raise ValueError('Transfer changed')
            extra=np.load(out/f'{model}_{suffix}_states.npz') if suffix!='transfer' else None
            for r in rows:
                i=r['row'];j=source_index[r['donor']];c=cuts.index(r['cut']);base,natural,oracle,transferred,random=[np.asarray(r[k],np.float32) for k in ['base','natural','oracle','transferred','random_forecast']]
                energy=float(np.mean((natural-base)**2));denom=max(energy,1e-20)
                np.testing.assert_allclose([energy,float(np.mean((oracle-natural)**2)/denom),float(np.mean((transferred-natural)**2)/denom),float(np.mean((random-natural)**2)/denom)],[r[k] for k in ['response_energy','oracle_response_nmse','transfer_response_nmse','random_response_nmse']],rtol=1e-7,atol=1e-10)
                raw=np.sin(2*np.pi*np.arange(-128,0)/period+.3);expected=r['epsilon']*float(histories[i].std())*np.sin(2*np.pi*np.arange(32)/period+.3)/np.sqrt(np.mean(raw**2))
                error=float(np.mean(((natural-base)-expected)**2)/np.mean(expected**2));np.testing.assert_allclose(error,r['native_signal_nmse'],rtol=1e-6,atol=1e-8)
                if extra is None:
                    desired=original[f"{i}_sinusoid_{r['epsilon']}_locked"][c,0].astype(float)-original[f'{i}_full'][c].astype(float)
                    edit=original[f"{j}_sinusoid_{r['epsilon']}_locked"][c,0].astype(float)-original[f'{j}_full'][c].astype(float)
                else:
                    desired=extra[f'{i}_perturbed'][c].astype(float)-extra[f'{i}_full'][c].astype(float);edit=extra[f'{j}_perturbed'][c].astype(float)-extra[f'{j}_full'][c].astype(float)
                np.testing.assert_allclose(np.mean((edit-desired)**2)/max(float(np.mean(desired**2)),1e-20),r['state_effect_nmse'],rtol=1e-10,atol=1e-10)
                assert r['oracle_valid']==(energy>1e-12*float(histories[i].std())**2 and r['oracle_response_nmse']<=1e-4)
                assert r['semantic_valid']==(r['oracle_valid'] and r['native_signal_nmse']<=.5)
                assert metadata[i]['split']=='test' and metadata[j]['split']=='fit' and metadata[i]['domain']==metadata[j]['domain'];audit+=1
                cases.append(dict(model=model,period=period,**{k:v for k,v in r.items() if k not in ['period','base','natural','oracle','transferred','random_forecast']}))
            for domain in sorted({r['domain'] for r in rows}):
                for cut in cuts:
                    for epsilon in sorted({r['epsilon'] for r in rows}):
                        selected=[r for r in rows if r['domain']==domain and r['cut']==cut and r['epsilon']==epsilon]
                        near={r['source']:r for r in selected if r['distance']=='near'};far={r['source']:r for r in selected if r['distance']=='far'}
                        for scope in ['oracle_valid','semantic_valid']:
                            ids=sorted(s for s in near if near[s][scope] and far[s][scope] and near[s]['random_valid'] and far[s]['random_valid'])
                            record=dict(model=model,domain=domain,period=period,cut=cut,epsilon=epsilon,scope=scope,eligible_sources=len(ids),total_sources=len(near))
                            if ids:
                                record.update(near_nmse=float(np.mean([near[s]['transfer_response_nmse'] for s in ids])),far_nmse=float(np.mean([far[s]['transfer_response_nmse'] for s in ids])),native_signal_nmse=float(np.mean([near[s]['native_signal_nmse'] for s in ids])),near_random_advantage=estimate([near[s]['random_response_nmse']-near[s]['transfer_response_nmse'] for s in ids]),near_over_far=estimate([far[s]['transfer_response_nmse']-near[s]['transfer_response_nmse'] for s in ids]))
                            summaries.append(record)
    write(out/'transfer_summary.json',summaries);write(out/'transfer_cases.json',cases);write(out/'transfer_audit.json',dict(checked_rows=audit,passed=True,source_sha256=sha(__file__)));print('audited transfer rows',audit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
