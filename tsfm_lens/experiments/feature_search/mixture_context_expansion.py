"""Frozen three-context, three-model native composition expansion.

All context lengths end on complete32/16 periods, with identical underlying
seeds and global phases. PrimaryHolm6 and secondaryHolm9 remain complete.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from experiments.feature_search.mixture_replication import make_corpus,_validate_compositions
from experiments.feature_search.mixture_interference import input_interference
from experiments.feature_search.mixture_composition_audit import composition_diagnostics
from tsfm_lens.analysis.stats import mean_ci
from tsfm_lens.experiments.feature_search.statistics import compare_effects

PROTOCOL_PATH=Path(__file__).with_name('mixture_context_protocol.json')


def load_protocol():
    """Load the independently frozen context/model expansion protocol."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def make_context_corpora(protocol=None):
    """Reuse identical physical signals and generator draws at each full-period context."""
    protocol=load_protocol() if protocol is None else protocol
    if protocol['horizon']!=32 or any(not isinstance(c,int) or c<128 or c%32 for c in protocol['contexts']):
        raise ValueError('Need context lengths>=128 divisible by32 and horizon32')
    contexts,truths,metadata={},{},None
    for context in protocol['contexts']:
        config=dict(protocol,context=context)
        contexts[context],truths[context],rows=make_corpus(config)
        if metadata is None:metadata=rows
        elif rows!=metadata:raise ValueError('Context conditions changed generator metadata')
    return contexts,truths,metadata


def analyze(predictions,truths,metadata,*,protocol=None):
    """Compute exact composition matches and frozen paired global claim families."""
    protocol=load_protocol() if protocol is None else protocol
    contexts=protocol['contexts'];models=list(protocol['models'])
    if set(predictions)!=set(contexts) or set(truths)!=set(contexts):raise ValueError('Need all three frozen context conditions')
    if any(c%32 or c<128 for c in contexts) or protocol['horizon']!=32:raise ValueError('Complete-period contexts>=128 and horizon32 required')
    settings=protocol['inference'];mean_settings=dict(n_boot=settings['n_boot'],seed=settings['seed'],unit='generating_seed_bundle')
    statistic_settings=dict(n_boot=settings['n_boot'],n_permutations=settings['n_permutations'],seed=settings['seed'],ci=settings['ci'])
    primary,secondary,descriptive={},{},{}
    row_seed=np.array([row['seed'] for row in metadata]);condition=np.array([row['condition'] for row in metadata]);mixed=condition=='mixed'
    scale=np.array([row['scale'] for row in metadata]);mixed_rows=[row for row,keep in zip(metadata,mixed) if keep]
    seeds=None
    for context in contexts:
        truth=np.asarray(truths[context],dtype=float)
        current_seeds=_validate_compositions(truth,metadata,'condition',forecast_start=context)
        if seeds is None:seeds=current_seeds
        elif not np.array_equal(seeds,current_seeds):raise ValueError('Contexts must share generating bundles')
        if not np.allclose(truth,truths[contexts[0]],atol=1e-10):raise ValueError('Complete-period contexts changed global future phase')
        if set(predictions[context])!=set(models):raise ValueError('Need all three prespecified models per context')
        losses,composed_losses,entries={},{},{}
        for model in models:
            pred=np.asarray(predictions[context][model],dtype=float)
            if pred.shape!=truth.shape or not np.isfinite(pred).all():raise ValueError('Native forecasts must be finite and corpus-aligned')
            raw=np.mean((pred-truth)**2,axis=1);scaled=raw/scale**2
            losses[model]={c:np.array([scaled[(row_seed==s)&(condition==c)].mean() for s in seeds]) for c in ('mixed','pure_phi','pure_psi')}
            diagnostic,composed=composition_diagnostics(pred,truth,metadata,n_boot=settings['n_boot'],seed=settings['seed'],forecast_start=context)
            row_composed=np.mean((composed-truth[mixed])**2,axis=1)/scale[mixed]**2
            composed_losses[model]=np.array([row_composed[row_seed[mixed]==s].mean() for s in seeds])
            interference={}
            for target in ('phi','psi'):
                response=input_interference(pred[mixed],truth[mixed],mixed_rows,target,forecast_start=context,phase_floor=protocol['phase_floor_relative_to_truth_amplitude'])
                interference[target]={}
                for key,value in response['metrics'].items():
                    valid=np.isfinite(value)
                    interference[target][key]=dict(untestable='fewer than3 identifiable seeds') if valid.sum()<3 else mean_ci(value[valid],**mean_settings)
                    interference[target][key]['excluded_seeds']=int((~valid).sum())
            entries[model]=dict(composition=diagnostic,input_interference=interference,
                scaled_mse={c:float(v.mean()) for c,v in losses[model].items()},
                mixed_mse_raw=float(raw[mixed].mean()),forecast_start=context)
        primary[f'timesfm_oracle_gain_context{context}']=(losses['timesfm']['mixed'],composed_losses['timesfm'])
        mix_gap=losses['timesfm']['mixed']-losses['bolt']['mixed']
        pure_gap=.5*((losses['timesfm']['pure_phi']-losses['bolt']['pure_phi'])+(losses['timesfm']['pure_psi']-losses['bolt']['pure_psi']))
        primary[f'timesfm_minus_bolt_composition_interaction_context{context}']=(mix_gap,pure_gap)
        for first,second in itertools.combinations(models,2):
            secondary[f'{first}_minus_{second}_mixed_mse_context{context}']=(losses[first]['mixed'],losses[second]['mixed'])
        descriptive[str(context)]=entries
    if set(primary)!=set(protocol['primary_claims']) or len(primary)!=6 or len(secondary)!=9:raise ValueError('Frozen globalHolm6/9 families must remain complete')
    primary_results=compare_effects(primary,seeds,**statistic_settings);secondary_results=compare_effects(secondary,seeds,**statistic_settings)
    for result in primary_results.values():
        result.update(unit='common mixed variance scaled MSE',minimum_practical=protocol['minimum_practical'])
        result['passed']=bool(result['value']>=protocol['minimum_practical'] and result['lo']>0 and result['p_holm']<.05)
    for result in secondary_results.values():result['unit']='first-minus-second native mixed MSE;common mixed variance scaled'
    return dict(n_seeds=len(seeds),contexts=contexts,models=models,primary_family_size=6,secondary_family_size=9,
                primary=primary_results,secondary_native_mixed_loss=secondary_results,descriptive=descriptive,
                units='All scaled MSE uses SAME per-bundle (a*a+b*b)/2 for pure,mixed,and oracle sum;raw MSE explicitly labeled',
                caution='Oracle separate pure-component inputs and exact offset;not a generic forecast upgrade;context-dependent negatives retained')


def run(out):
    """Run one serial offline worker with saved protocol and a ten-minute cap."""
    import torch
    from experiments.feature_search.activation_interchange import patched_predict
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    protocol=load_protocol();out.mkdir(parents=True,exist_ok=False);protocol_bytes=PROTOCOL_PATH.read_bytes();(out/'protocol.json').write_bytes(protocol_bytes)
    contexts,truths,metadata=make_context_corpora(protocol)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for context in protocol['contexts']:np.savez_compressed(out/f'corpus_context{context}.npz',context=contexts[context],clean_future=truths[context])
    provenance=dict(protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),estimated_seconds=[120,300],models={},native_path='Existing unchanged adapter.predict;no hooks or model weights changed')
    predictions={c:{} for c in protocol['contexts']};started=time.perf_counter()
    classes=dict(timesfm=TimesFMAdapter,bolt=ChronosBoltAdapter,chronos2=Chronos2Adapter)
    print(json.dumps(dict(expected_total_contexts=4608,estimated_seconds=[120,300],cap_seconds=600,checkpoints=protocol['models'])),flush=True)
    for model,checkpoint in protocol['models'].items():
        if time.perf_counter()-started>=600:
            (out/'paused.json').write_text(json.dumps(dict(reason='ten-minute cap before next model',model=model),indent=2),encoding='utf-8');return
        adapter=classes[model](ModelConfig(name=model,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded()
        provenance['models'][model]={}
        for context in protocol['contexts']:
            elapsed=time.perf_counter()-started
            if elapsed>=600:
                (out/'paused.json').write_text(json.dumps(dict(reason='ten-minute cap before next context',model=model,context=context,elapsed=elapsed),indent=2),encoding='utf-8');return
            adapter.data_cfg=DataConfig(context_len=context,horizon=32)
            block=time.perf_counter();probe=patched_predict(adapter,contexts[context][:8]);probe_seconds=time.perf_counter()-block
            print(json.dumps(dict(model=model,context=context,estimated_block_seconds=probe_seconds*len(contexts[context])/8,elapsed_seconds=elapsed)),flush=True)
            prediction=patched_predict(adapter,contexts[context]);repeat=patched_predict(adapter,contexts[context][:8]);repeat_error=float(np.max(np.abs(repeat-prediction[:8])))
            if repeat_error!=0 or not np.array_equal(probe,repeat):raise RuntimeError(f'Native same-batch repeat failed{model}@{context}')
            if model=='chronos2':
                isolated=patched_predict(adapter,contexts[context][:8][::-1].copy())[::-1]
                if not np.allclose(isolated,repeat,rtol=1e-6,atol=1e-6):raise RuntimeError('Chronos2 batch permutation changed independent-series forecasts')
            predictions[context][model]=prediction;np.savez_compressed(out/f'{model}_context{context}_predictions.npz',prediction=prediction)
            provenance['models'][model][str(context)]=dict(repeat_max_abs=repeat_error,seconds=time.perf_counter()-block,
                token_count=int(len(adapter.token_time_spans())),native_alignment='unchanged predict/quantile median path;physical token geometry declared,no activation analysis')
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
            print(json.dumps(dict(model=model,context=context,completed=512,seconds=time.perf_counter()-block,total_elapsed=time.perf_counter()-started)),flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    result=analyze(predictions,truths,metadata,protocol=protocol)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(primary=result['primary'],secondary=result['secondary_native_mixed_loss']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(args.out)
