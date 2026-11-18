"""Frozen saved-array station contrasts and orthogonal mechanism diagnostics."""
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.persistence_protocol import read,write,sha,exact
from experiments.feature_search.postrelease_weather_lock import binding
from experiments.feature_search.postrelease_weather_score import station_loss,bootstrap
from experiments.feature_search.postrelease_weather_posthoc import location,grouped,FIELDS


def analyze(out):
    """Both valid workers required; retain primary cohort and shared station draws."""
    p=binding(out)
    if (out/'analysis').exists():raise ValueError('Refuse analysis overwrite')
    assert sha(out/'evaluation/targets.npz')==p['target_sha256']
    data=np.load(out/'inference/inputs.npz');y=np.load(out/'evaluation/targets.npz')['future']
    rows=read(out/'inference/rows.json');plans=read(out/'inference/plans.json')
    indices=np.random.default_rng(p['bootstrap']['seed']).integers(0,8,(p['bootstrap']['draws'],8))
    results={};arrays=dict(bootstrap_indices=indices);partitions={}
    for model in ['bolt','timesfm']:
        execution=read(out/f'{model}_execution.json');worker=read(out/f'{model}_worker.json')
        assert execution['valid'] and worker['valid'] and worker['protocol_sha256']==sha(out/'protocol.json')
        assert worker['forecast_sha256']==sha(out/f'{model}_forecasts.npz')
        f=np.load(out/f'{model}_forecasts.npz');reference=np.load(out/'inference'/f'{model}_reference.npz')
        exact(f['native'],reference['native'],'scoring full reference')
        exact(f['ordinary_hybrid'],reference['hybrid'],'scoring ordinary hybrid reference')
        exact(f['locked_full'],f['native'],'scoring locked full identity')
        exact(f['locked_hybrid'],data['tones']+f['locked_residual'],'locked reconstruction')
        losses={};result={}
        for method in ['native','ordinary_hybrid','locked_hybrid']:
            window,station=station_loss(f[method],y,data['scales'],rows)
            arrays[model+'_'+method+'_window']=window;arrays[model+'_'+method+'_station']=station
            losses[method]=station;result[method]=dict(mean_normalized_mse=float(station.mean()),station_losses=station.tolist())
        comparisons={};model_partitions={}
        for candidate,baseline in [('locked_hybrid','ordinary_hybrid'),('locked_hybrid','native'),('ordinary_hybrid','native')]:
            label=candidate+'_vs_'+baseline
            comparisons[label]=bootstrap(losses[baseline]-losses[candidate],indices)
            values=[location(a,b,t,plan['details'],v) for a,b,t,plan,v in zip(f[baseline],f[candidate],y,plans,data['scales'])]
            groups=grouped(values,rows)
            np.testing.assert_allclose(groups['gain'],losses[baseline]-losses[candidate],rtol=1e-10,atol=2e-12)
            model_partitions[label]=dict(windows=values,station_values={k:v.tolist() for k,v in groups.items()},
                                        summary={k:bootstrap(v,indices) for k,v in groups.items()})
        results[model]=dict(losses=result,contrasts=comparisons);partitions[model]=model_partitions
    (out/'analysis').mkdir();write(out/'analysis/summary.json',results);write(out/'analysis/partitions.json',partitions)
    np.savez(out/'analysis/losses.npz',**arrays)
    k=read(out/'knowledge.json');k['exact_findings']['results']=results;k['exact_findings']['partition_summaries']={m:{label:value['summary'] for label,value in vals.items()} for m,vals in partitions.items()}
    k['evidence_class']='descriptive normalizer intervention on fixed weather cohort; no training diagnosis'
    k['next_decision']='Independent saved-array replay, then review; no tuning, native relaunch or expansion.'
    write(out/'knowledge.json',k);write(out/'analysis/knowledge.json',k);print(results)


def replay(out):
    """Independent scalar loss and normal-equation projection replay of every contrast."""
    p=binding(out);assert sha(out/'evaluation/targets.npz')==p['target_sha256']
    x=np.load(out/'inference/inputs.npz');y=np.load(out/'evaluation/targets.npz')['future']
    rows=read(out/'inference/rows.json');plans=read(out/'inference/plans.json')
    saved=np.load(out/'analysis/losses.npz');parts=read(out/'analysis/partitions.json');results=read(out/'analysis/summary.json')
    worst=0.;count=0
    for model in ['bolt','timesfm']:
        forecasts=np.load(out/f'{model}_forecasts.npz')
        for method in ['native','ordinary_hybrid','locked_hybrid']:
            window=np.array([sum((float(a)-float(b))**2 for a,b in zip(pred,t))/32/float(v)
                             for pred,t,v in zip(forecasts[method],y,x['scales'])])
            np.testing.assert_allclose(window,saved[model+'_'+method+'_window'],rtol=1e-12,atol=1e-12)
            for j,station in enumerate(p['stations']):
                ix=[i for i,row in enumerate(rows) if row['station']==station]
                np.testing.assert_allclose(sum(window[i] for i in ix)/len(ix),saved[model+'_'+method+'_station'][j],rtol=1e-12,atol=1e-12)
        for label,part in parts[model].items():
            candidate,baseline=label.split('_vs_')
            for i,plan in enumerate(plans):
                a=forecasts[baseline][i]-y[i];b=forecasts[candidate][i]-y[i];delta=b-a
                if plan['details']:
                    f=plan['details'][0]['frequency'];t=np.arange(32)
                    basis=np.column_stack([np.sin(2*np.pi*f*t/128),np.cos(2*np.pi*f*t/128)])
                    project=basis@np.linalg.inv(basis.T@basis)@basis.T
                    ap=project@a;bp=project@b;dp=project@delta
                    ms=lambda v:sum(float(z)**2 for z in v)/32/float(x['scales'][i])
                    values=[ms(a)-ms(b),ms(ap)-ms(bp),ms(a-ap)-ms(b-bp),ms(delta),ms(dp),ms(delta-dp)]
                else:
                    exact(forecasts[baseline][i],forecasts[candidate][i],'replay abstention');values=[0.]*6
                for key,value in zip(FIELDS,values):
                    diff=abs(value-part['windows'][i][key]);worst=max(worst,diff);count+=1
                    np.testing.assert_allclose(value,part['windows'][i][key],rtol=1e-10,atol=2e-12)
            gain=saved[model+'_'+baseline+'_station']-saved[model+'_'+candidate+'_station']
            draws=[sum(float(gain[i]) for i in idx)/8 for idx in saved['bootstrap_indices']]
            np.testing.assert_allclose(np.quantile(draws,[.025,.975]),results[model]['contrasts'][label]['percentile95'],rtol=1e-12,atol=1e-12)
    audit=dict(window_losses=180,station_losses=48,partition_scalars=count,contrasts=6,max_abs_partition_error=worst)
    write(out/'analysis/independent_replay.json',audit)
    k=read(out/'knowledge.json');k['exact_findings']['independent_replay']=audit
    write(out/'knowledge.json',k);write(out/'analysis/audit_knowledge.json',k);print(audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    replay(args.out) if args.replay else analyze(args.out)
