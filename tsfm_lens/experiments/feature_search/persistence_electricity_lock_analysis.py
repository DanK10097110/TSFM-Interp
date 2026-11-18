"""Descriptive paired64-source mechanism contrasts and explicit batch4 drift bounds."""
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.persistence_protocol import read,write,sha,exact,finite
from experiments.feature_search.postrelease_weather_lock import binding
from experiments.feature_search.postrelease_weather_posthoc import location,FIELDS


def drift_bound(new,old,target):
    """Cauchy-Schwarz bounds absolute per-source MSE drift without assuming roundoff."""
    new,old,target=[finite(x).astype(float) for x in (new,old,target)]
    old_loss=np.mean((old-target)**2,axis=1);new_loss=np.mean((new-target)**2,axis=1)
    rms=np.sqrt(np.mean((new-old)**2,axis=1))
    bound=2*np.sqrt(old_loss)*rms+rms**2
    observed=np.abs(new_loss-old_loss)
    if np.any(observed>bound+1e-12):raise ValueError('Loss drift exceeds algebraic bound')
    return dict(old_loss=old_loss.tolist(),new_loss=new_loss.tolist(),absolute_drift=observed.tolist(),
                per_source_upper_bound=bound.tolist(),mean_absolute_drift=float(observed.mean()),
                mean_upper_bound=float(bound.mean()))


def paired(gain,indices):
    """Shared-source bootstrap is descriptive on reused sources; ties remain in mean."""
    gain=finite(gain)
    return dict(mean_gain=float(gain.mean()),wins=int((gain>0).sum()),ties=int((gain==0).sum()),losses=int((gain<0).sum()),
                percentile95=np.quantile(gain[indices].mean(axis=1),[.025,.975]).tolist(),
                interpretation='Descriptive paired-source bootstrap on already-seen electricity sources; no fresh confirmation or population significance claim.')


def analyze(out):
    """Analyze only two complete valid singleton captures; old batch4 is a separate audit."""
    p=binding(out);assert sha(out/'evaluation/targets.npz')==p['target_sha256']
    if (out/'analysis').exists():raise ValueError('Refuse analysis overwrite')
    data=np.load(out/'inference/inputs.npz');plans=read(out/'inference/plans.json');target=np.load(out/'evaluation/targets.npz');y=target['future']
    rows=read(out/'inference/rows.json');assert np.array_equal(target['source_ids'],[r['source'] for r in rows])
    indices=np.random.default_rng(p['bootstrap']['seed']).integers(0,64,(p['bootstrap']['draws'],64))
    results={};arrays=dict(bootstrap_indices=indices,source_ids=target['source_ids']);parts={};drifts={}
    for model in ['bolt','timesfm']:
        e=read(out/f'{model}_execution.json');w=read(out/f'{model}_worker.json')
        assert e['valid'] and w['valid'] and w['protocol_sha256']==sha(out/'protocol.json') and w['forecast_sha256']==sha(out/f'{model}_forecasts.npz')
        f=np.load(out/f'{model}_forecasts.npz');old=np.load(out/'inference'/f'{model}_reference.npz')
        exact(f['locked_full'],f['native'],'saved locked self');exact(f['ordinary_hybrid'],f['ordinary_residual']+data['tones'],'ordinary assembly');exact(f['locked_hybrid'],f['locked_residual']+data['tones'],'locked assembly')
        loss={method:np.mean((finite(f[method])-y)**2,axis=1) for method in ['native','ordinary_hybrid','locked_hybrid']}
        comparisons={};parts[model]={}
        for candidate,baseline in [('locked_hybrid','ordinary_hybrid'),('locked_hybrid','native'),('ordinary_hybrid','native')]:
            label=candidate+'_vs_'+baseline;gain=loss[baseline]-loss[candidate]
            comparisons[label]=paired(gain,indices)
            values=[location(a,b,t,plan['details'],1.) for a,b,t,plan in zip(f[baseline],f[candidate],y,plans)]
            np.testing.assert_allclose([v['gain'] for v in values],gain,rtol=1e-10,atol=2e-12)
            parts[model][label]=dict(sources=values,summary={field:paired(np.array([v[field] for v in values]),indices) for field in FIELDS})
        for method,value in loss.items():arrays[model+'_'+method+'_loss']=value
        results[model]=dict(losses={method:float(value.mean()) for method,value in loss.items()},contrasts=comparisons)
        drift={method:drift_bound(f[method],old[oldname],y) for method,oldname in [('native','native'),('ordinary_hybrid','hybrid')]}
        old_gain=np.array(drift['native']['old_loss'])-np.array(drift['ordinary_hybrid']['old_loss']);fresh_gain=loss['native']-loss['ordinary_hybrid']
        upper=np.array(drift['native']['per_source_upper_bound'])+np.array(drift['ordinary_hybrid']['per_source_upper_bound'])
        assert np.all(abs(fresh_gain-old_gain)<=upper+2e-12)
        drift['gain_drift']=dict(old_mean_gain=float(old_gain.mean()),batch1_mean_gain=float(fresh_gain.mean()),mean_absolute_source_gain_drift=float(abs(fresh_gain-old_gain).mean()),mean_upper_bound=float(upper.mean()),source_sign_changes=int(np.sum(np.sign(old_gain)!=np.sign(fresh_gain))),
             interpretation='Actual gain differences and conservative algebraic bound; not an approximate-equality acceptance gate or a diagnosis of floating-point cause.')
        drifts[model]=drift
    (out/'analysis').mkdir();write(out/'analysis/summary.json',results);write(out/'analysis/partitions.json',parts);write(out/'analysis/batch4_loss_drift.json',drifts)
    np.savez(out/'analysis/arrays.npz',**arrays)
    k=read(out/'knowledge.json');k['exact_findings']['results']=results;k['exact_findings']['batch4_loss_drift']=drifts;k['exact_findings']['partition_summaries']={m:{c:v['summary'] for c,v in part.items()} for m,part in parts.items()}
    k['next_decision']='Independent saved-array arithmetic audit; report normalizer-mediated response without fresh-confirmation or training claims. No new variants.'
    write(out/'knowledge.json',k);write(out/'analysis/knowledge.json',k);print(results)


def replay(out):
    """Independent scalar losses and normal-equation partitions, no native loading."""
    p=binding(out);assert sha(out/'evaluation/targets.npz')==p['target_sha256']
    y=np.load(out/'evaluation/targets.npz')['future'];plans=read(out/'inference/plans.json');saved=np.load(out/'analysis/arrays.npz');parts=read(out/'analysis/partitions.json');results=read(out/'analysis/summary.json');drifts=read(out/'analysis/batch4_loss_drift.json')
    worst=0.;count=0
    for model in ['bolt','timesfm']:
        f=np.load(out/f'{model}_forecasts.npz');old=np.load(out/'inference'/f'{model}_reference.npz')
        for method in ['native','ordinary_hybrid','locked_hybrid']:
            scalar=np.array([sum((float(a)-float(b))**2 for a,b in zip(pred,t))/32 for pred,t in zip(f[method],y)])
            np.testing.assert_allclose(scalar,saved[model+'_'+method+'_loss'],rtol=1e-12,atol=1e-12)
            if method!='locked_hybrid':
                old_method='hybrid' if method=='ordinary_hybrid' else 'native'
                old_loss=np.array([sum((float(a)-float(b))**2 for a,b in zip(pred,t))/32 for pred,t in zip(old[old_method],y)])
                rms=np.array([np.linalg.norm(np.asarray(a,float)-b)/np.sqrt(32) for a,b in zip(f[method],old[old_method])])
                np.testing.assert_allclose(2*np.sqrt(old_loss)*rms+rms*rms,drifts[model][method]['per_source_upper_bound'],rtol=1e-12,atol=1e-12)
        for label,part in parts[model].items():
            candidate,baseline=label.split('_vs_')
            for i,plan in enumerate(plans):
                a=f[baseline][i]-y[i];b=f[candidate][i]-y[i];delta=b-a
                if plan['details']:
                    freq=plan['details'][0]['frequency'];angle=2*np.pi*freq*np.arange(32)/128;basis=np.column_stack([np.sin(angle),np.cos(angle)])
                    proj=basis@np.linalg.inv(basis.T@basis)@basis.T;ap=proj@a;bp=proj@b;dp=proj@delta
                    ms=lambda v:sum(float(z)**2 for z in v)/32
                    values=[ms(a)-ms(b),ms(ap)-ms(bp),ms(a-ap)-ms(b-bp),ms(delta),ms(dp),ms(delta-dp)]
                else:
                    exact(f[baseline][i],f[candidate][i],'replay abstention');values=[0.]*6
                for key,value in zip(FIELDS,values):
                    worst=max(worst,abs(value-part['sources'][i][key]));count+=1
                    np.testing.assert_allclose(value,part['sources'][i][key],rtol=1e-10,atol=2e-12)
            gain=saved[model+'_'+baseline+'_loss']-saved[model+'_'+candidate+'_loss']
            draws=[sum(float(gain[i]) for i in index)/64 for index in saved['bootstrap_indices']]
            np.testing.assert_allclose(np.quantile(draws,[.025,.975]),results[model]['contrasts'][label]['percentile95'],rtol=1e-12,atol=1e-12)
    audit=dict(source_losses_replayed=384,partition_scalars_replayed=count,contrasts_replayed=6,loss_drift_bounds_replayed=256,max_abs_partition_error=worst)
    write(out/'analysis/independent_replay.json',audit);k=read(out/'knowledge.json');k['exact_findings']['independent_replay']=audit
    write(out/'knowledge.json',k);write(out/'analysis/audit_knowledge.json',k);print(audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    replay(args.out) if args.replay else analyze(args.out)
