"""Independent scalar/group replay of saved weather artifacts; no model loading."""
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.persistence_protocol import read, write, sha


def replay(out, forecasts=False):
    """Recompute preprocessing/analytic forecasts; optionally replay all saved losses."""
    p=read(out/'protocol.json');inf=read(out/'inference/protocol.json')
    assert sha(out/'protocol.json')==read(out/'seal.json')['sha256']
    data=np.load(out/'inference/inputs.npz');rows=read(out/'inference/rows.json');plans=read(out/'inference/plans.json')
    station_counts={s:sum(r['station']==s for r in rows) for s in p['stations']}
    assert list(station_counts.values())==p['windows_by_station']
    worst=0.
    for i,(x,plan) in enumerate(zip(data['history'],plans)):
        assert np.array_equal(data['bank'][plan['native']],x.astype(np.float32))
        full=np.zeros(128);future=np.zeros(32)
        for item in plan['details']:
            a,b=item['coefficients'];freq=item['frequency']
            full+=a*np.sin(2*np.pi*freq*np.arange(-128,0)/128)+b*np.cos(2*np.pi*freq*np.arange(-128,0)/128)
            future+=a*np.sin(2*np.pi*freq*np.arange(32)/128)+b*np.cos(2*np.pi*freq*np.arange(32)/128)
        np.testing.assert_allclose(data['tones'][i],future,rtol=1e-12,atol=1e-12)
        error=np.max(abs(data['bank'][plan['residual']]-(x-full)))
        bound=np.max(abs(np.spacing((x-full).astype(np.float32))))
        assert error <= bound+1e-12
        worst=max(worst,float(error))
        np.testing.assert_array_equal(data['seasonal24'][i],np.array([x[104+j%24] for j in range(32)]))
        assert np.isclose(data['scales'][i],sum((x-sum(x)/128)**2)/128,rtol=1e-14)
    result=dict(windows=30,station_counts=station_counts,analytic_replay=True,
                max_residual_float32_cast_error=worst,forecast_replay=False)
    if forecasts:
        target=np.load(out/'evaluation/targets.npz')['future']
        assert sha(out/'evaluation/targets.npz')==p['target_sha256']
        saved=np.load(out/'analysis/losses.npz');summary=read(out/'analysis/summary.json')
        for model in ['bolt','timesfm']:
            capture=np.load(out/f'{model}_forecasts.npz')
            for method in p['methods']:
                window=np.array([sum((float(a)-float(b))**2 for a,b in zip(pred,y))/32/float(v)
                                 for pred,y,v in zip(capture[method],target,data['scales'])])
                station=np.array([sum(window[i] for i,r in enumerate(rows) if r['station']==s)/station_counts[s] for s in p['stations']])
                np.testing.assert_allclose(window,saved[f'{model}_{method}_window'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(station,saved[f'{model}_{method}_station'],rtol=1e-12,atol=1e-12)
                assert np.isclose(station.mean(),summary[model]['losses'][method]['equal_station_normalized_mse'],rtol=1e-12)
            for key,comparison in summary[model]['comparisons'].items():
                candidate,reference=key.split('_vs_')
                gain=saved[f'{model}_{reference}_station']-saved[f'{model}_{candidate}_station']
                draws=[sum(float(gain[j]) for j in ix)/8 for ix in saved['bootstrap_indices']]
                np.testing.assert_allclose(np.quantile(draws,[.025,.975]),comparison['percentile95'],rtol=1e-12,atol=1e-12)
        result.update(forecast_replay=True,window_losses_replayed=240,station_losses_replayed=64,paired_intervals_replayed=8)
    name='forecast_replay' if forecasts else 'preparation_replay'
    write(out/f'{name}.json',result)
    knowledge=read(out/'knowledge.json');knowledge['exact_findings'][name]=result
    write(out/'knowledge.json',knowledge);write(out/f'{name}_knowledge.json',knowledge)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--forecasts',action='store_true');args=parser.parse_args()
    replay(args.out,args.forecasts)
