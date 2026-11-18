"""Sealed station-weighted descriptive scoring, callable only on saved arrays."""
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.persistence_protocol import finite, read, sha, write
from experiments.feature_search.postrelease_weather_pilot import METHODS, STATIONS, binding


def station_loss(prediction, target, scales, rows):
    """Normalize per window first, average within station second; never pool windows."""
    prediction, target, scales = map(finite,(prediction,target,scales))
    if prediction.shape != target.shape or prediction.shape != (len(rows),32) or scales.shape != (len(rows),) or np.any(scales<=0):
        raise ValueError('Loss shape/scale')
    per_window = np.mean((prediction-target)**2,axis=1)/scales
    per_station = np.array([per_window[[r['station']==s for r in rows]].mean() for s in STATIONS])
    return finite(per_window),finite(per_station)


def bootstrap(gain, indices):
    """Positive gain favors the candidate; preserve paired station resamples."""
    draws = finite(gain)[indices].mean(axis=1)
    return dict(mean_gain=float(np.mean(gain)),station_wins=int(np.sum(gain>0)),
                percentile95=np.quantile(draws,[.025,.975]).tolist(),
                interpretation='Descriptive station bootstrap; cross-station weather dependence prevents calibrated population-mean inference.')


def score(out):
    """Refuse partial or unsealed captures and publish descriptive, paired results."""
    binding(out)
    protocol = read(out/'protocol.json')
    assert sha(out/'evaluation/targets.npz') == protocol['target_sha256']
    target = np.load(out/'evaluation/targets.npz')['future']
    scales = np.load(out/'inference/inputs.npz')['scales'];rows=read(out/'inference/rows.json')
    indices = np.random.default_rng(protocol['uncertainty']['seed']).integers(0,8,(4096,8))
    summary, arrays = {}, dict(bootstrap_indices=indices)
    for model in ['bolt','timesfm']:
        execution=read(out/f'{model}_execution.json');worker=read(out/f'{model}_worker.json')
        assert execution['valid'] and worker['valid']
        assert worker['protocol_sha256'] == sha(out/'protocol.json')
        assert worker['forecast_sha256'] == sha(out/f'{model}_forecasts.npz')
        forecasts=np.load(out/f'{model}_forecasts.npz');losses={};model_summary={}
        for method in METHODS:
            window,station=station_loss(forecasts[method],target,scales,rows)
            arrays[f'{model}_{method}_window']=window;arrays[f'{model}_{method}_station']=station
            losses[method]=station
            model_summary[method]=dict(equal_station_normalized_mse=float(station.mean()),station_losses=station.tolist())
        comparisons={}
        for candidate,reference in [('hybrid','native'),('native','seasonal24'),('hybrid','seasonal24'),('tones_zero','hybrid')]:
            gain=losses[reference]-losses[candidate]
            comparisons[f'{candidate}_vs_{reference}']=bootstrap(gain,indices)
        summary[model]=dict(losses=model_summary,comparisons=comparisons)
    if (out/'analysis').exists():
        raise ValueError('Refuse analysis overwrite')
    (out/'analysis').mkdir();write(out/'analysis/summary.json',summary)
    np.savez(out/'analysis/losses.npz',**arrays)
    knowledge=read(out/'knowledge.json');knowledge['exact_findings']['forecast_results']=summary
    knowledge['evidence_class']='behavioral chronological-observation holdout, descriptive8-station pilot'
    knowledge['next_decision']='Run independent saved-array replay; report exact results; no expansion or tuning without coordination.'
    write(out/'analysis/knowledge.json',knowledge);write(out/'knowledge.json',knowledge)
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    score(parser.parse_args().out)
