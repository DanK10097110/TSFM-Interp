"""Controlled synthetic composition with a shared Gaussian context background.

Actual background forecasts replace oracle-offset subtraction. White future
noise has conditional mean zero, so all future targets remain clean. Neither
input decomposition nor these synthetic conditions represent real-data utility.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.feature_search.normalization_audit import normalization_state
from experiments.feature_search.mixture_interference import input_interference
from tsfm_lens.analysis.stats import mean_ci,bootstrap_ci
from tsfm_lens.experiments.feature_search.statistics import compare_effects

PROTOCOL_PATH=Path(__file__).with_name('mixture_noise_protocol.json')


def load_protocol():
    """Read the fixed four-claim white-background robustness protocol."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def make_corpus(protocol=None):
    """Generate9cells/level using the same context-noise vector and clean futures."""
    protocol=load_protocol() if protocol is None else protocol;context=protocol['context'];horizon=protocol['horizon']
    contexts,truths,metadata,noise=[],[],[],[];t=np.arange(-context,horizon)
    for seed in range(protocol['n_bundles']):
        rng=np.random.default_rng(protocol['seed_base']+seed);a,b=rng.uniform(*protocol['amplitudes'],2);offset=rng.uniform(*protocol['offset']);phi,psi=rng.uniform(-np.pi,np.pi,2)
        epsilon=rng.normal(size=context);noise.append(epsilon);scale=np.sqrt((a*a+b*b)/2)
        cells=[('mixed',p,q) for p in (0,1) for q in (0,1)]+[('pure_phi',p,0) for p in (0,1)]+[('pure_psi',0,q) for q in (0,1)]+[('background_only',0,0)]
        for level in protocol['noise_levels']:
            background=offset+level*scale*epsilon
            for condition,p,q in cells:
                angle_phi,angle_psi=phi+p*protocol['phase_shift'],psi+q*protocol['phase_shift'];clean=np.full(len(t),offset)
                if condition in ('mixed','pure_phi'):clean+=a*np.sin(2*np.pi*t/protocol['periods'][0]+angle_phi)
                if condition in ('mixed','pure_psi'):clean+=b*np.sin(2*np.pi*t/protocol['periods'][1]+angle_psi)
                contexts.append(clean[:context]+level*scale*epsilon);truths.append(clean[context:])
                metadata.append(dict(seed=seed,generator_seed=protocol['seed_base']+seed,split='test',noise_level=float(level),noise_std=float(level*scale),
                    condition=condition,cell_phi=p,cell_psi=q,phi=angle_phi,psi=angle_psi,a=a,b=b,offset=offset,scale=float(scale)))
    return np.array(contexts,dtype=np.float32),np.array(truths),metadata,np.array(noise)


def matched_rows(metadata,level):
    """Require complete9-cell blocks and match actual background/pure forecasts."""
    lookup={};expected={('mixed',p,q) for p in (0,1) for q in (0,1)}|{('pure_phi',p,0) for p in (0,1)}|{('pure_psi',0,q) for q in (0,1)}|{('background_only',0,0)}
    for i,row in enumerate(metadata):
        if row['noise_level']!=level:continue
        key=(row['seed'],row['condition'],row['cell_phi'],row['cell_psi'])
        if key in lookup:raise ValueError('Duplicate seed/noise factorial cell')
        lookup[key]=i
    seeds=np.unique([key[0] for key in lookup]);mixed=[];pure_phi=[];pure_psi=[];background=[];groups=[]
    for seed in seeds:
        if {(key[1],key[2],key[3]) for key in lookup if key[0]==seed}!=expected:raise ValueError('Need all9phase/pure/background cells per seed/noise level')
        for p,q in [(0,0),(0,1),(1,0),(1,1)]:
            mixed.append(lookup[seed,'mixed',p,q]);pure_phi.append(lookup[seed,'pure_phi',p,0]);pure_psi.append(lookup[seed,'pure_psi',0,q]);background.append(lookup[seed,'background_only',0,0]);groups.append(seed)
    if len(seeds)<3:raise ValueError('Need at least3 generating-seed bundles')
    return dict(seeds=seeds,mixed=np.array(mixed),pure_phi=np.array(pure_phi),pure_psi=np.array(pure_psi),background=np.array(background),groups=np.array(groups))


def compose_prediction(prediction,rows):
    """Use actual f(background), not a known offset, to avoid noise/intercept double count."""
    prediction=np.asarray(prediction,dtype=float)
    if prediction.ndim!=2 or not np.isfinite(prediction).all():raise ValueError('Need finite native forecast matrix')
    return prediction[rows['pure_phi']]+prediction[rows['pure_psi']]-prediction[rows['background']]


def validate_corpus(contexts,truth,metadata,protocol=None):
    """Verify shared-background algebra and clean conditional-mean future targets."""
    protocol=load_protocol() if protocol is None else protocol;contexts=np.asarray(contexts,dtype=float);truth=np.asarray(truth,dtype=float)
    if contexts.shape!=(len(metadata),protocol['context']) or truth.shape!=(len(metadata),protocol['horizon']) or not np.isfinite(contexts).all() or not np.isfinite(truth).all():raise ValueError('Need finite corpus-aligned context/future arrays')
    future_t=np.arange(protocol['horizon']);max_input_error=0.;seen=set()
    for row,target in zip(metadata,truth):
        a,b,scale=row['a'],row['b'],row['scale']
        if min(a,b,scale)<=0 or not np.isclose(scale**2,(a*a+b*b)/2) or not np.isclose(row['noise_std'],row['noise_level']*scale):raise ValueError('Need common clean-mixture energy and declared noise scale')
        clean=np.full(protocol['horizon'],row['offset'])
        if row['condition'] in ('mixed','pure_phi'):clean+=a*np.sin(2*np.pi*future_t/protocol['periods'][0]+row['phi'])
        if row['condition'] in ('mixed','pure_psi'):clean+=b*np.sin(2*np.pi*future_t/protocol['periods'][1]+row['psi'])
        if not np.allclose(target,clean,atol=1e-10):raise ValueError('Future truth must be clean conditional-mean components,not realized future noise')
        seen.add(row['noise_level'])
    if seen!=set(protocol['noise_levels']):raise ValueError('Both prespecified noise levels must remain present')
    for level in protocol['noise_levels']:
        rows=matched_rows(metadata,level);m,p,q,bg=(rows[key] for key in ('mixed','pure_phi','pure_psi','background'))
        discrepancy=contexts[m]-(contexts[p]+contexts[q]-contexts[bg]);max_input_error=max(max_input_error,float(np.max(np.abs(discrepancy))))
        if not np.allclose(discrepancy,0,atol=2e-6):raise ValueError('Phase/pure/background contexts do not share the same additive background')
        if not np.allclose(truth[m],truth[p]+truth[q]-truth[bg],atol=1e-10):raise ValueError('True components are not additive')
        if not np.allclose(truth[bg],np.array([metadata[i]['offset'] for i in bg])[:,None],atol=1e-12):raise ValueError('Background future conditional mean must be offset')
        for seed in rows['seeds']:
            block=[row for row in metadata if row['seed']==seed and row['noise_level']==level]
            for key in ('a','b','scale','offset','noise_std'):
                if not np.allclose([row[key] for row in block],block[0][key],atol=1e-12,rtol=1e-12):raise ValueError('All9cells must share bundle nuisances and noise scale')
            selected=[i for i,row in enumerate(metadata) if row['seed']==seed and row['noise_level']==level]
            background_index=next(i for i in selected if metadata[i]['condition']=='background_only')
            expected_noise=contexts[background_index]-metadata[background_index]['offset']
            context_t=np.arange(-protocol['context'],0)
            for i in selected:
                row=metadata[i];clean=np.full(protocol['context'],row['offset'])
                if row['condition'] in ('mixed','pure_phi'):clean+=row['a']*np.sin(2*np.pi*context_t/protocol['periods'][0]+row['phi'])
                if row['condition'] in ('mixed','pure_psi'):clean+=row['b']*np.sin(2*np.pi*context_t/protocol['periods'][1]+row['psi'])
                if not np.allclose(contexts[i]-clean,expected_noise,atol=2e-6):raise ValueError('Each phase/pure/background cell must use the same Gaussian context realization')
    return dict(max_input_additivity_roundoff=max_input_error,future_noise='not realized;white future conditional mean zero',background_cancellation='actual native forecast subtraction')


def analyze(predictions,truth,metadata,*,protocol=None):
    """Evaluate all4MSE claims and retain exploratory noisy-minus-clean error separately."""
    protocol=load_protocol() if protocol is None else protocol;truth=np.asarray(truth,dtype=float);settings=protocol['inference']
    if set(predictions)!=set(protocol['models']):raise ValueError('Both prespecified model forecasts required')
    statistic_settings=dict(n_boot=settings['n_boot'],n_permutations=settings['n_permutations'],seed=settings['seed'],ci=settings['ci']);mean_settings=dict(n_boot=settings['n_boot'],seed=settings['seed'],unit='generating_seed_bundle')
    claims,models,losses={}, {}, {};seeds=None
    for model,prediction in predictions.items():
        prediction=np.asarray(prediction,dtype=float)
        if prediction.shape!=truth.shape or not np.isfinite(prediction).all():raise ValueError('Stop:material nonfinite or misaligned native forecasts')
        entries={}
        for level,label in [(0.,'noise0'),(.1,'noise01')]:
            rows=matched_rows(metadata,level)
            if seeds is None:seeds=rows['seeds']
            elif not np.array_equal(seeds,rows['seeds']):raise ValueError('All levels/models must share64seed bundles')
            def average(values):return np.array([np.mean(values[rows['groups']==seed]) for seed in seeds])
            mixed=rows['mixed'];energy=np.array([metadata[i]['scale']**2 for i in mixed]);composed=compose_prediction(prediction,rows)
            native_raw=np.mean((prediction[mixed]-truth[mixed])**2,axis=1);composed_raw=np.mean((composed-truth[mixed])**2,axis=1);defect_raw=np.mean((prediction[mixed]-composed)**2,axis=1)
            native=average(native_raw/energy);component=average(composed_raw/energy);losses[model,label]=native
            metrics={key:mean_ci(average(value),**mean_settings) for key,value in dict(native_mse_raw=native_raw,component_sum_mse_raw=composed_raw,defect_mse_raw=defect_raw,
                native_mse_common_mixvariance=native_raw/energy,component_sum_mse_common_mixvariance=composed_raw/energy,defect_mse_common_mixvariance=defect_raw/energy,gain_common_mixvariance=(native_raw-composed_raw)/energy).items()}
            background_error=np.mean((prediction[rows['background']]-truth[rows['background']])**2,axis=1)/energy
            oracle_offset=np.array([metadata[i]['offset'] for i in mixed])[:,None];wrong=prediction[rows['pure_phi']]+prediction[rows['pure_psi']]-oracle_offset
            metrics['actual_background_forecast_mse_common_mixvariance']=mean_ci(average(background_error),**mean_settings)
            metrics['wrong_oracle_offset_sum_mse_common_mixvariance']=mean_ci(average(np.mean((wrong-truth[mixed])**2,axis=1)/energy),**mean_settings)
            fraction=None if np.min(native)<=1e-20 else bootstrap_ci(lambda idx:1-float(component[idx].mean()/native[idx].mean()),len(seeds),n_boot=settings['n_boot'],seed=settings['seed'],unit='generating_seed_bundle')
            entries[label]=dict(metrics=metrics,fraction_error_removed=fraction,noise_level=level,
                baseline='f(pure_phi+shared_background)+f(pure_psi+shared_background)-ACTUAL f(shared_background);known input decomposition still oracle information')
            phase_diagnostics={}
            for target in ('phi','psi'):
                response=input_interference(prediction[mixed],truth[mixed],[metadata[i] for i in mixed],target,forecast_start=0)
                phase_diagnostics[target]={}
                for key,values in response['metrics'].items():
                    valid=np.isfinite(values)
                    estimate=dict(untestable='fewer than3 phase-identifiable seed bundles') if valid.sum()<3 else mean_ci(values[valid],**mean_settings)
                    estimate['excluded_seeds']=int((~valid).sum());phase_diagnostics[target][key]=estimate
            entries[label]['behavioral_input_interference']=phase_diagnostics
            entries[label]['phase_caution']='Descriptive native input coupling;white-noise covariance can change normalizer sigma across phase cells;no hidden-feature conclusion'
            if model=='timesfm':claims['timesfm_component_gain_'+label]=(native,component)
        models[model]=entries
    for label in ('noise0','noise01'):claims['timesfm_minus_bolt_mixed_mse_'+label]=(losses['timesfm',label],losses['bolt',label])
    if set(claims)!=set(protocol['primary_claims']):raise ValueError('Complete frozenHolm4 family must remain intact')
    primary=compare_effects(claims,seeds,**statistic_settings)
    for value in primary.values():value.update(unit='common clean-mixture variance scaled MSE',minimum_practical=protocol['minimum_practical'],passed=bool(value['value']>=protocol['minimum_practical'] and value['lo']>0 and value['p_holm']<.05))
    exploratory=mean_ci(losses['timesfm','noise01']-losses['timesfm','noise0'],**mean_settings)
    return dict(primary=primary,all_four_pass=all(value['passed'] for value in primary.values()),models=models,n_seeds=len(seeds),n_contexts=len(metadata),
        exploratory_timesfm_noisy_minus_clean_mixed_mse=exploratory,exploratory_caution='Two-sided descriptiveCI only;unexpected direction needs fresh replication before claim',
        units='All scaled MSE uses SAME true clean oscillatory mixed variance(a*a+b*b)/2;raw MSE explicitly labeled',
        evidence='controlled synthetic shared Gaussian background,not real-world data;no hidden-feature/tangent conclusion')


def run(out):
    """Run one bounded offline native worker after protocol and input guards persist."""
    import torch
    from experiments.feature_search.activation_interchange import patched_predict
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    protocol=load_protocol();out.mkdir(parents=True,exist_ok=False);raw_protocol=PROTOCOL_PATH.read_bytes();(out/'protocol.json').write_bytes(raw_protocol)
    context,truth,metadata,noise=make_corpus(protocol);validation=validate_corpus(context,truth,metadata,protocol)
    np.savez_compressed(out/'corpus.npz',context=context,clean_future=truth,standard_context_noise_by_seed=noise)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');(out/'input_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    predictions={};normalizers={};provenance=dict(protocol_sha256=hashlib.sha256(raw_protocol).hexdigest(),models={});started=time.perf_counter()
    print(json.dumps(dict(total_contexts_per_model=len(context),estimated_seconds=[30,120],cap_seconds=600)),flush=True)
    for model,cls,surface in [('timesfm',TimesFMAdapter,'timesfm_2p5_decode'),('bolt',ChronosBoltAdapter,'chronos_bolt')]:
        if time.perf_counter()-started>600:
            (out/'paused.json').write_text(json.dumps(dict(reason='ten-minute cap before next model',model=model)),encoding='utf-8');return
        block=time.perf_counter();adapter=cls(ModelConfig(name=model,checkpoint=protocol['models'][model]),DataConfig(context_len=512,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded()
        prediction=patched_predict(adapter,context)
        if not np.isfinite(prediction).all():raise RuntimeError('Stop:material nonfinite native predictions')
        repeat=patched_predict(adapter,context[:8]);error=float(np.max(np.abs(repeat-prediction[:8])))
        if error!=0:raise RuntimeError('Native same-batch deterministic repeat failed')
        predictions[model]=prediction;np.savez_compressed(out/f'{model}_predictions.npz',prediction=prediction)
        mu,sigma=normalization_state(context,surface);entries={}
        for level in protocol['noise_levels']:
            rows=matched_rows(metadata,level);scales=np.array([metadata[i]['scale'] for i in rows['mixed']]);groups=rows['groups']
            entries[str(level)]=dict(mean_mixed_phase_location_range_over_scale=float(np.mean([np.ptp(mu[rows['mixed'][groups==seed],0])/scales[groups==seed][0] for seed in rows['seeds']])),
                mean_mixed_phase_sigma_range_over_scale=float(np.mean([np.ptp(sigma[rows['mixed'][groups==seed],0])/scales[groups==seed][0] for seed in rows['seeds']])))
        normalizers[model]=entries;provenance['models'][model]=dict(checkpoint=protocol['models'][model],repeat_max_abs=error,seconds=time.perf_counter()-block)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print(json.dumps(dict(model=model,completed=len(context),seconds=time.perf_counter()-block,repeat_max_abs=error)),flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    result=analyze(predictions,truth,metadata,protocol=protocol);result.update(input_normalizer_mixed_phase_ranges=normalizers)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(primary=result['primary'],exploratory=result['exploratory_timesfm_noisy_minus_clean_mixed_mse']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(args.out)
