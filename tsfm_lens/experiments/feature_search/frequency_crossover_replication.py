"""Fresh Stage49 crossover replication with actual-background composition control.

The22-effect exact-sign family is frozen before inference. Pure-frequency error
can mimic a mixed-minus-average-pure interaction; only actual native background
forecasts define the component-sum diagnostic. No hidden states are captured.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from tsfm_lens.analysis.feature_interference import sign_comparison
from tsfm_lens.analysis.stats import holm
from experiments.feature_search import frequency_generalization as screen

CONDITIONS = ('baseline','minus','plus','pure_phi','pure_psi','background')
EFFECTS = ('g32','g40','g40_minus_g32','C32','C40','C40_minus_C32',
           'o32','o40','o40_minus_o32','G_timesfm32','G_bolt40')


def protocol():
    """Return the approved frozen replication, not an outcome-selected screen."""
    config = screen.protocol()
    config.update(seed_base=3400000,n_seeds=64,pairs=[(32,16),(40,20)],contexts=[128,512],
        bootstrap_seed=3400017,family_size=22,minimum_absolute_mean=None,
        conditions=CONDITIONS,primary_context=128,sensitivity_context=512,
        units='Every pure/mixed/background MSE uses common theoretical (a²+b²)/2, not realized finite-window variance; derivative energy uses b²/2',
        comparator_scope='S uses oracle-separated component inputs and three native calls; not a deployable repair. G includes all native computation and input-dependent normalization, not specifically hidden interactions',
        effects=EFFECTS,minimum_mse_effect=.05,minimum_off_energy_effect=.02,
        role='fresh replication of the preselected crossover; both contexts globally corrected;512 cannot rescue128',
        composition='G=L-S where S uses actual f(pure_phi)+f(pure_psi)-f(background); C=G_timesfm-G_bolt',
        legacy_interaction='mixed-minus-average-pure J is descriptive only; separable frequency error can produce J!=0 at G=0',
        ratio='sum of per-seed normalized S divided by sum of per-seed normalized L; paired whole-seed bootstrap, pointwise',
        ranking_flip='requires passing g32>0,g40<0 AND g40-g32<0 at128;512 sensitivity does not rescue',
        nonadditivity='positive passing G establishes a diagnostic error reduction; nonsignificant G does not establish absence',
        candidate_gate='No further expansion without root review')
    config['resource']['runtime_seconds']=300
    return config


def make_case(config, periods, context):
    """Generate six matched cells with shared latents and actual constant background."""
    baseline, original = screen.make_case(config,periods,context)
    inputs,futures,components,rows = [],[],[],[]
    for seed in range(config['n_seeds']):
        base = original[seed*3]
        for condition in CONDITIONS:
            row = dict(base,condition=condition,phase_shift=dict(minus=-config['delta'],plus=config['delta']).get(condition,0.))
            include_phi = condition not in ('pure_psi','background')
            include_psi = condition not in ('pure_phi','background')
            def component(t):
                return np.stack([row['a']*np.sin(2*np.pi*t/periods[0]+row['phi'])*include_phi,
                    row['b']*np.sin(2*np.pi*t/periods[1]+row['psi']+row['phase_shift'])*include_psi])
            future_components = component(np.arange(config['horizon']))
            inputs.append(row['offset']+component(np.arange(-context,0)).sum(0))
            futures.append(row['offset']+future_components.sum(0))
            components.append(future_components)
            rows.append(row)
    ids=np.asarray([f"{row['generator_seed']}:{periods[0]}:{periods[1]}:{context}:{row['condition']}" for row in rows])
    return dict(context=np.asarray(inputs,dtype=np.float32),clean_future=np.asarray(futures),
                component_future=np.asarray(components),row_ids=ids),rows


def component_metrics(prediction,data,metadata,periods,config):
    """Preserve common mixed energy, native background and full-Gram responses."""
    pred=np.asarray(prediction)
    truth=data['clean_future']
    if pred.shape!=truth.shape or not np.isfinite(pred).all():
        raise ValueError('Need finite forecasts matching corpus shape')
    lookup={(row['seed'],row['condition']):i for i,row in enumerate(metadata)}
    if len(lookup)!=len(metadata) or set(lookup)!={(s,c) for s in range(config['n_seeds']) for c in CONDITIONS}:
        raise ValueError('Need exactly six matched cells per seed')
    selected=[lookup[s,c] for s in range(config['n_seeds']) for c in CONDITIONS[:3]]
    response,diagnostics=screen.metrics(pred[selected],truth[selected],[metadata[i] for i in selected],periods,config['delta'])
    values={key:[] for key in ('L','S','G','pure_phi','pure_psi','background','J','component_sum_direct_discrepancy')}
    for seed in range(config['n_seeds']):
        indices={c:lookup[seed,c] for c in CONDITIONS}
        row=metadata[indices['baseline']]
        scale=(row['a']**2+row['b']**2)/2
        losses={c:np.mean((pred[i]-truth[i])**2)/scale for c,i in indices.items()}
        composed=pred[indices['pure_phi']]+pred[indices['pure_psi']]-pred[indices['background']]
        native=pred[indices['baseline']]
        residual=np.mean((composed-truth[indices['baseline']])**2)/scale
        values['L'].append(losses['baseline'])
        values['S'].append(residual)
        values['G'].append(losses['baseline']-residual)
        values['J'].append(losses['baseline']-.5*(losses['pure_phi']+losses['pure_psi']))
        for c in ('pure_phi','pure_psi','background'):values[c].append(losses[c])
        values['component_sum_direct_discrepancy'].append(np.mean((composed-native)**2)/scale)
    return dict({key:np.asarray(value) for key,value in values.items()},
                **{key:value for key,value in response.items() if key!='normalized_mse'}),diagnostics


def residual_fraction(native,composed,config):
    """Bootstrap paired ratio of aggregate seed losses, refusing zero denominator."""
    native,composed=np.asarray(native),np.asarray(composed)
    if native.ndim!=1 or native.shape!=composed.shape or not np.isfinite(native).all() or not np.isfinite(composed).all():
        raise ValueError('Ratio requires matched finite seed vectors')
    if native.sum()<=1e-24:
        return dict(value=None,lo=None,hi=None,untestable='zero native loss denominator')
    rng=np.random.default_rng(config['bootstrap_seed'])
    draws=rng.integers(0,len(native),size=(config['n_boot'],len(native)))
    denominator=native[draws].sum(1)
    valid=denominator>1e-24
    ratios=composed[draws].sum(1)[valid]/denominator[valid]
    lo,hi=np.quantile(ratios,[.025,.975]) if len(ratios) else (None,None)
    return dict(value=float(composed.sum()/native.sum()),lo=None if lo is None else float(lo),hi=None if hi is None else float(hi),
                excluded_bootstrap_draws=int((~valid).sum()),interval='approximate pointwise paired whole-seed ratio bootstrap; not mean of individual ratios')


def effects(values):
    """Construct the eleven frozen TF-minus-Bolt effects for one context."""
    def difference(period,key):return values[period]['timesfm'][key]-values[period]['bolt'][key]
    g32,g40=difference(32,'L'),difference(40,'L')
    c32,c40=difference(32,'G'),difference(40,'G')
    o32,o40=difference(32,'off_target_energy'),difference(40,'off_target_energy')
    return dict(zip(EFFECTS,(g32,g40,g40-g32,c32,c40,c40-c32,o32,o40,o40-o32,
                              values[32]['timesfm']['G'],values[40]['bolt']['G'])))


def ranking_flip(claims,context):
    """A significant crossover alone cannot establish opposite model rankings."""
    required={'g32':1,'g40':-1,'g40_minus_g32':-1}
    return all(claims[f'c{context}/{key}']['passed'] and
               np.sign(claims[f'c{context}/{key}']['mean']['value'])==direction
               for key,direction in required.items())


def analyze(config,cases,predictions):
    """Use the complete22 family; never replace failed effects with descriptive J."""
    claims,descriptive,vectors={}, {},{}
    for context in config['contexts']:
        values={}
        for periods in config['pairs']:
            name=f'p{periods[0]}_{periods[1]}_c{context}'
            data,metadata=cases[name]
            values[periods[0]]={}
            descriptive[name]={}
            for model in config['models']:
                model_name=model['name']
                artifact=predictions[name][model_name]
                if not np.array_equal(artifact['row_ids'],data['row_ids']):
                    raise ValueError('Forecast row identity mismatch')
                value,diagnostics=component_metrics(artifact['prediction'],data,metadata,periods,config)
                values[periods[0]][model_name]=value
                descriptive[name][model_name]=dict(metrics={key:screen.summarize(v,config) for key,v in value.items()},
                    residual_fraction=residual_fraction(value['L'],value['S'],config),diagnostics=diagnostics)
                vectors.update({f'{name}/{model_name}/{key}':v for key,v in value.items()})
        for key,difference in effects(values).items():
            effect=sign_comparison(difference)
            effect.update(mean=screen.summarize(difference,config),minimum_absolute_mean=
                config['minimum_off_energy_effect'] if key.startswith('o') else config['minimum_mse_effect'])
            claims[f'c{context}/{key}']=effect
    if set(claims)!={f'c{context}/{key}' for context in (128,512) for key in EFFECTS} or len(claims)!=config['family_size']:
        raise ValueError('Frozen22-effect family changed')
    adjusted=holm({key:value['p'] for key,value in claims.items()})
    for key,value in claims.items():
        mean=value['mean']['value']
        majority=0 if value['value'] is None else np.sign(value['value']-.5)
        value.update(p_holm=adjusted[key],passed=bool(adjusted[key]<config['alpha'] and abs(mean)>=value['minimum_absolute_mean'] and np.sign(mean)==majority),
                     scope='primary' if key.startswith('c128/') else 'sensitivity; cannot rescue128')
    gains={key:bool(claims[key]['passed'] and claims[key]['mean']['value']>0)
           for key in claims if key.split('/')[1] in ('G_timesfm32','G_bolt40')}
    return dict(protocol=config,claims=claims,descriptive=descriptive,primary_ranking_flip=ranking_flip(claims,128),
                sensitivity_ranking_flip=ranking_flip(claims,512),positive_gain_evidence=gains,
                inference='global Holm22 exact two-sided conditional sign tests; approximate pointwise mean/ratio bootstrap CIs; q is not mean superiority',
                limitation='Nonsignificant G does not establish absence. Descriptive J can be nonzero for a separable frequency-biased forecast with G=0; no hidden mechanism established'),vectors


def prepare(out):
    """Save physical inputs and exact protocol/source hashes before inference."""
    config=protocol()
    out.mkdir(parents=True,exist_ok=False)
    sources={}
    for periods in config['pairs']:
        for context in config['contexts']:
            name=f'p{periods[0]}_{periods[1]}_c{context}'
            data,metadata=make_case(config,periods,context)
            component_metrics(data['clean_future'],data,metadata,periods,config)
            np.savez_compressed(out/f'{name}.npz',**data)
            (out/f'{name}.json').write_text(json.dumps(metadata,indent=2), encoding='utf-8')
            sources[name]=dict(corpus_sha256=screen.digest(out/f'{name}.npz'),metadata_sha256=screen.digest(out/f'{name}.json'),periods=periods)
    payload=dict(config=config,sources=sources,driver_sha256=screen.digest(__file__),
                 screen_helper_sha256=screen.digest(screen.__file__),prepared_before_inference=True)
    (out/'protocol.json').write_text(json.dumps(payload,indent=2,allow_nan=False), encoding='utf-8')
    return payload


def load_prepared(prepared):
    """Reject hand-rebound inputs that disagree with the frozen physical generator."""
    payload=json.loads((prepared/'protocol.json').read_text(encoding='utf-8'))
    config=payload['config']
    if config!=json.loads(json.dumps(protocol())) or payload['driver_sha256']!=screen.digest(__file__) or payload['screen_helper_sha256']!=screen.digest(screen.__file__):
        raise ValueError('Frozen protocol or producer/helper source changed')
    expected={f'p{p[0]}_{p[1]}_c{c}' for p in config['pairs'] for c in config['contexts']}
    if set(payload['sources'])!=expected:raise ValueError('Frozen case family changed')
    cases={}
    for periods in config['pairs']:
        for context in config['contexts']:
            name=f'p{periods[0]}_{periods[1]}_c{context}'
            source=payload['sources'][name]
            if screen.digest(prepared/f'{name}.npz')!=source['corpus_sha256'] or screen.digest(prepared/f'{name}.json')!=source['metadata_sha256']:
                raise ValueError('Frozen corpus or metadata bytes changed')
            with np.load(prepared/f'{name}.npz') as stored:data={key:stored[key] for key in stored.files}
            metadata=json.loads((prepared/f'{name}.json').read_text(encoding='utf-8'))
            generated,rows=make_case(config,periods,context)
            if rows!=metadata or any(not np.array_equal(data[key],value) for key,value in generated.items()):
                raise ValueError('Frozen physical input/background differs from generator')
            cases[name]=(data,metadata)
    return payload,config,cases


def run(prepared,out,device):
    """Run one explicitly assigned offline worker; save actual memory telemetry."""
    import torch
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models import build_adapter
    from experiments.feature_search.normalization_audit import normalization_state
    payload,config,cases=load_prepared(prepared)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_bytes((prepared/'protocol.json').read_bytes())
    torch.set_num_threads(2)
    target=torch.device(device)
    if target.type=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,6*1024**3/torch.cuda.get_device_properties(target).total_memory),target)
    predictions,provenance={name:{} for name in cases},{}
    started=time.monotonic()
    for model in config['models']:
        model_started=time.monotonic()
        if target.type=='cuda':torch.cuda.reset_peak_memory_stats(target)
        adapter=build_adapter(ModelConfig(**model),DataConfig(context_len=128,horizon=config['horizon']),target,torch.float32)
        adapter.ensure_loaded()
        provenance[model['name']]=dict(config=asdict(ModelConfig(**model)),
            adapter_source_sha256=screen.digest(__import__('inspect').getfile(type(adapter))),
            checkpoint_identifier=model['checkpoint'],checkpoint_identifier_sha256=hashlib.sha256(model['checkpoint'].encode()).hexdigest(),
            checkpoint_scope='Identifier, not a checksum of weights; offline installed cached checkpoint only',cases={})
        native_config=getattr(adapter.module,'config',None)
        provenance[model['name']]['resolved_model_config_declarations']={key:getattr(native_config,key,None) for key in ('_name_or_path','_commit_hash')}
        provenance[model['name']]['resolved_declaration_scope']='Native declarations when available; null means unavailable, not verified weight bytes'
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        for name,(data,metadata) in cases.items():
            adapter.data_cfg=DataConfig(context_len=data['context'].shape[1],horizon=config['horizon'])
            blocks=[]
            for start in range(0,len(data['context']),8):
                if time.monotonic()-started>config['resource']['runtime_seconds']:raise RuntimeError('Five-minute runtime cap reached')
                with torch.inference_mode():blocks.append(adapter.predict(data['context'][start:start+8],config['horizon'],[.5])['point'])
            pred=np.concatenate(blocks)
            repeat=adapter.predict(data['context'][:8],config['horizon'],[.5])['point']
            permutation=adapter.predict(data['context'][:8][::-1].copy(),config['horizon'],[.5])['point'][::-1]
            if not np.array_equal(repeat,pred[:8]) or not np.allclose(repeat,permutation,atol=1e-6,rtol=1e-6):
                raise RuntimeError('Native repeat/permutation guard failed')
            if model['name']=='chronos2':
                singleton=np.concatenate([adapter.predict(x[None],config['horizon'],[.5])['point'] for x in data['context'][:8]])
                if not np.allclose(singleton,repeat,atol=1e-6,rtol=1e-6):raise RuntimeError('Chronos2 singleton independence failed')
                _,(mu,sigma)=adapter._inner.instance_norm(adapter.prepare(data['context']))
                mu,sigma=mu.cpu().numpy(),sigma.cpu().numpy()
                native=dict(output_patches=adapter._num_output_patches(config['horizon']),use_arcsinh=bool(adapter._inner.chronos_config.use_arcsinh),singleton_max_abs=float(np.max(abs(singleton-repeat))),normalization_source_sha256=screen.digest(__import__('inspect').getfile(type(adapter._inner.instance_norm))))
            else:
                mu,sigma=normalization_state(data['context'],'chronos_bolt' if model['name']=='bolt' else 'timesfm_2p5_decode',horizon=config['horizon'])
                if model['name']=='bolt':steps=int(adapter.pipeline.model_prediction_length)
                else:
                    from timesfm.timesfm_2p5.timesfm_2p5_base import TimesFM_2p5_200M_Definition
                    steps=int(TimesFM_2p5_200M_Definition().output_patch_len)
                if steps<config['horizon']:raise RuntimeError('Single native head horizon no longer supported')
                native=dict(output_steps=steps,normalization_audit_source_sha256=screen.digest(__import__('inspect').getfile(normalization_state)))
            artifact=dict(prediction=pred,row_ids=data['row_ids'],corpus_sha256=np.asarray(payload['sources'][name]['corpus_sha256']))
            path=out/f"{model['name']}_{name}.npz"
            np.savez_compressed(path,**artifact,native_mu=mu,native_sigma=sigma,
                empirical_mean=data['context'].mean(1),empirical_std=data['context'].std(1))
            predictions[name][model['name']]=artifact
            provenance[model['name']]['cases'][name]=dict(native=native,forward_passes=1,autoregressive_rollouts=0,
                repeat_max_abs=float(np.max(abs(repeat-pred[:8]))),permutation_max_abs=float(np.max(abs(repeat-permutation))),forecast_sha256=screen.digest(path))
            (out/'provenance.json').write_text(json.dumps(provenance,indent=2), encoding='utf-8')
            print(json.dumps(dict(model=model['name'],case=name,elapsed=time.monotonic()-started)),flush=True)
        provenance[model['name']]['wall_seconds']=time.monotonic()-model_started
        if target.type=='cuda':
            provenance[model['name']]['peak_allocated_bytes']=torch.cuda.max_memory_allocated(target)
            provenance[model['name']]['peak_reserved_bytes']=torch.cuda.max_memory_reserved(target)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2), encoding='utf-8')
        adapter._release()
        del adapter
        gc.collect()
        if target.type=='cuda':torch.cuda.empty_cache()
    result,vectors=analyze(config,cases,predictions)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False), encoding='utf-8')
    np.savez_compressed(out/'seed_metrics.npz',**vectors)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.prepare_only==bool(args.prepared):parser.error('Choose prepare-only or prepared')
    if args.prepare_only:prepare(args.out)
    else:run(args.prepared,args.out,args.device)
