"""Saved-array weather diagnostics AFTER the frozen primary; never native or tuning.

Preserve temperature level with simple residual forecasts and reuse the natural
study's history-frequency error partition. Geometry does not identify latent
true components or establish that a training prior caused the observed error.
"""
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.persistence_protocol import exact, finite, read, sha, write
from experiments.feature_search.postrelease_weather_pilot import STATIONS, binding
from experiments.feature_search.postrelease_weather_score import station_loss, bootstrap
from experiments.feature_search.natural_gain_location import partition
from experiments.feature_search.postrelease_weather_prepare import utcnow

FIELDS=('gain','tone_gain','other_gain','update_energy','tone_update','other_update')


def level_baselines(history, details):
    """Retain mean or last observed residual, using only previously fitted tones."""
    history=finite(history).astype(float)
    if history.shape!=(128,) or len(details)>1:
        raise ValueError('Expected128 hours and at most1 already-fitted tone')
    past=np.zeros(128);future=np.zeros(32)
    for item in details:
        a,b=item['coefficients'];f=item['frequency']
        past+=a*np.sin(2*np.pi*f*np.arange(-128,0)/128)+b*np.cos(2*np.pi*f*np.arange(-128,0)/128)
        future+=a*np.sin(2*np.pi*f*np.arange(32)/128)+b*np.cos(2*np.pi*f*np.arange(32)/128)
    residual=history-past
    return finite(future+residual.mean()),finite(future+residual[-1])


def location(native,hybrid,target,details,scale):
    """Partition unmodified errors in the past-frequency plane, then normalize."""
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid history variance')
    if not details:
        exact(native,hybrid,'undefined-plane abstention')
        return {k:0. for k in FIELDS}
    scores,_=partition(native,hybrid,target,details[0]['frequency'])
    return {k:float(scores[k]/scale) for k in FIELDS}


def grouped(records,rows):
    """Retain all stations and all scheduled complete windows, including zero updates."""
    return {k:finite([np.mean([r[k] for r,row in zip(records,rows) if row['station']==s])
                      for s in STATIONS]) for k in FIELDS}


def run(source,out,natural):
    """Record posthoc recipes before calculating diagnostic outcomes, then audit."""
    binding(source)
    out.mkdir(parents=True,exist_ok=False)
    files=['protocol.json','inference/inputs.npz','inference/plans.json','inference/rows.json',
           'evaluation/targets.npz','bolt_forecasts.npz','timesfm_forecasts.npz',
           'analysis/summary.json','analysis/losses.npz','audit/final_audit.json']
    natural_protocol=read(natural/'protocol.json')
    natural_code=Path(__file__).with_name('natural_gain_location.py')
    assert sha(natural_code)==natural_protocol['code_sha256']
    protocol=dict(created_utc=utcnow(),stage='POSTHOC AFTER COMPLETED PRIMARY; no new confirmation',
        question='Do level-preserving simple residual forecasts help, and is weather hybrid harm in the history-frequency tone plane or elsewhere?',
        source=str(source.resolve()),bindings={name:sha(source/name) for name in files},
        sources={str(p.resolve()):sha(p) for p in [Path(__file__),natural_code,Path(__file__).with_name('postrelease_weather_score.py')]},
        baseline_mean='saved future tone + mean(original float64 history minus saved fitted past tone)',
        baseline_last='saved future tone + last(original float64 history minus saved fitted past tone)',
        abstention_baselines='No extracted tone: plain history mean or last observed temperature. No native fallback or method selection.',
        projection='Identical natural_gain_location.partition: QR of sin/cos at saved past-only frequency on32-step future grid; target unchanged, projected errors ONLY diagnostic.',
        abstention_partition='No plane defined; exact identical native/hybrid implies zero gain and zero update, retained in aggregates.',
        aggregation='Every gain and update-energy divided by original window history variance, then average windows within station, then equal8 stations; counts4/4/4/4/4/2/4/4.',
        uncertainty='Reuse primary4096 paired8-station bootstrap draws; descriptive because shared weather induces dependence. No new inferential or significance claim.',
        natural_comparison=dict(path=str(natural.resolve()),summary_sha256=sha(natural/'summary.json'),
              scope='Existing all64 electricity sources per model; natural forecasts/targets already own-history standardized. Compare descriptive partition signs, not a matched causal domain effect.'),
        limits=['tones_zero physical zero drops DC; its huge loss is NOT useful evidence for residual necessity.',
                'Future tone-plane projection is not ground-truth component labeling; over32 hours sin/cos can overlap trends or levels.',
                'In-plane update includes both added continuation and changed native residual response; cannot attribute it solely to bad classical continuation.',
                'Native outputs are medians, not means. Chronological observation holdout does not imply unseen locations/distribution.'],
        budget='Saved arrays only, CPU1; no downloads, native calls, fits, threshold tuning or forecast selection.')
    write(out/'protocol.json',protocol);write(out/'seal.json',dict(sha256=sha(out/'protocol.json')))
    knowledge=dict(question=protocol['question'],design=protocol,exact_findings=dict(status='prepared',native_launched=False),
                   controls='Primary bindings; fixed history-only plane; level-preserving constants; exact abstentions; orthogonal identities.',
                   limitations=protocol['limits']+[protocol['uncertainty']],evidence_class='posthoc descriptive saved-array diagnostic',
                   next_decision='Compute diagnostic only; native branch closed.')
    write(out/'knowledge.json',knowledge);write(out/'preparation_knowledge.json',knowledge)
    data=np.load(source/'inference/inputs.npz');plans=read(source/'inference/plans.json');rows=read(source/'inference/rows.json')
    target=np.load(source/'evaluation/targets.npz')['future'];old=np.load(source/'analysis/losses.npz');indices=old['bootstrap_indices']
    preds=np.array([level_baselines(x,p['details']) for x,p in zip(data['history'],plans)])
    outputs={};baseline_results={}
    for j,name in enumerate(['tone_mean_residual','tone_last_residual']):
        pred=preds[:,j];window,station=station_loss(pred,target,data['scales'],rows)
        outputs[name]=pred;outputs[name+'_window_loss']=window;outputs[name+'_station_loss']=station
        baseline_results[name]=dict(mean_normalized_mse=float(station.mean()),station_losses=station.tolist(),
            comparisons={model+'_native':bootstrap(old[model+'_native_station']-station,indices) for model in ['bolt','timesfm']})
    locations={};window_records={}
    for model in ['bolt','timesfm']:
        f=np.load(source/f'{model}_forecasts.npz')
        records=[location(n,h,y,p['details'],v) for n,h,y,p,v in zip(f['native'],f['hybrid'],target,plans,data['scales'])]
        grouped_scores=grouped(records,rows)
        np.testing.assert_allclose([r['gain'] for r in records],old[model+'_native_window']-old[model+'_hybrid_window'],rtol=1e-10,atol=2e-12)
        np.testing.assert_allclose(grouped_scores['gain'],grouped_scores['tone_gain']+grouped_scores['other_gain'],rtol=1e-10,atol=2e-12)
        np.testing.assert_allclose(grouped_scores['update_energy'],grouped_scores['tone_update']+grouped_scores['other_update'],rtol=1e-10,atol=2e-12)
        locations[model]={k:bootstrap(value,indices) for k,value in grouped_scores.items()}
        locations[model]['off_plane_update_fraction']=float(grouped_scores['other_update'].mean()/grouped_scores['update_energy'].mean())
        locations[model]['station_values']={k:v.tolist() for k,v in grouped_scores.items()}
        outputs.update({model+'_'+k:v for k,v in grouped_scores.items()});window_records[model]=records
    comparison=[r for r in read(natural/'summary.json') if r['domain']=='monash_electricity_hourly' and r['scope']=='all']
    assert len(comparison)==2
    result=dict(baselines=baseline_results,weather_gain_location=locations,electricity_reference=comparison,
                windows=30,stations=8,accepted_windows=sum(p['accepted'] for p in plans),native_launched=False)
    np.savez(out/'arrays.npz',**outputs);write(out/'window_partitions.json',window_records);write(out/'summary.json',result)
    knowledge['exact_findings']=result;knowledge['next_decision']='Independently replay partition and baseline arithmetic; report posthoc diagnostic; native branch remains closed.'
    write(out/'knowledge.json',knowledge)
    print(dict(baselines={k:v['mean_normalized_mse'] for k,v in baseline_results.items()},
               location={m:{k:v[k]['mean_gain'] for k in FIELDS} for m,v in locations.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--natural',type=Path,default=Path('tsfm_lens/runs/natural_gain_location4956300'))
    args=parser.parse_args();run(args.source,args.out,args.natural)
