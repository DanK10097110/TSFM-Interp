"""Separately authorized native forecast-only phase-orbit pilot runner."""
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
from experiments.feature_search import phase_orbit as orbit


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(source,out,checkpoint_manifest):
    """Freeze actual source/checkpoints before loading any model."""
    if out.exists():raise ValueError('Use fresh output')
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if protocol['driver_sha256']!=digest(orbit.__file__):raise ValueError('Core protocol/source changed')
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    import inspect
    import timesfm.torch.util as util
    import chronos.chronos_bolt as bolt
    import chronos.chronos2.model as c2
    import timesfm.timesfm_2p5.timesfm_2p5_torch as tf
    old=json.loads(checkpoint_manifest.read_text(encoding='utf-8'))['manifests']
    checkpoints={model:old[model]['checkpoint'] for model in ('timesfm','bolt','chronos2')}
    models={name:dict(checkpoint=path,weight_sha256=digest(path),config_sha256=digest(Path(path).parent/'config.json')) for name,path in checkpoints.items()}
    sources={inspect.getfile(module):digest(inspect.getfile(module)) for module in (TimesFMAdapter,ChronosBoltAdapter,Chronos2Adapter,util,bolt,c2,tf)}
    prospective=dict(core_source=str(source),core_protocol_sha256=digest(source/'protocol.json'),
        runner_sha256=digest(__file__),models=models,sources=sources,
        native_rows=3072,controls='repeat and random batch permutation on first8 rows/case exact0;C2 singleton independence first8 exact0;shape512×64;finite forecasts;native errorParseval<=1e-10 normalized;oracleD<=1e-20',
        resources=dict(gpu=3,cpu_threads=2,batch_size=8,cap_bytes=6*2**30,timeout_seconds=600),
        interpretation='Promotes at most alias validation; native ranking reversal failed and is not rescued; no hidden captures',
        source_note='PRIMARY phaseDFT of raw forecast waveforms and exact errorParseval;SECONDARY joint temporal coefficients conditional on temporal-span adequacy')
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(prospective,indent=2),encoding='utf-8')


def vector_profile(values,origins):
    """Phase power of physical vectors; temporal coordinates stay unaltered."""
    coefficients,modes=orbit.orbit_fft(values,origins)
    energy=np.mean(abs(coefficients)**2,axis=tuple(range(2,coefficients.ndim))) if coefficients.ndim>2 else abs(coefficients)**2
    return dict(modes=modes.tolist(),energy=energy.tolist(),
        parseval_absolute_error=float(abs(np.mean(abs(values)**2)-energy.sum())))


def row_ids(rows,stem):
    return np.asarray([f"{stem}:{r['generator_seed']}:{r['bundle']}:{r['grid']}:{r['cell']}" for r in rows])


def load_prediction(path,model,stem,source):
    """Require producer identity and exact ordered physical-corpus binding."""
    rows=json.loads((source/f'{stem}.json').read_text(encoding='utf-8'))
    expected=dict(model=model,case=stem,corpus_sha256=digest(source/f'{stem}.npz'),
                  metadata_sha256=digest(source/f'{stem}.json'),core_protocol_sha256=digest(source/'protocol.json'))
    with np.load(path,allow_pickle=False) as file:
        for key,value in expected.items():
            if key not in file or file[key].item()!=value:raise ValueError('Prediction identity/corpus binding mismatch')
        if 'phase_row_ids' not in file or not np.array_equal(file['phase_row_ids'],row_ids(rows,stem)):
            raise ValueError('Prediction ordered phase-row identity mismatch')
        if file['prediction'].shape!=(len(rows),64):raise ValueError('Prediction horizon/row shape mismatch')
        return {key:file[key].copy() for key in file.files}


def analyze_case(prediction,context,future,rows,pair,model):
    if prediction.shape!=(512,64) or not np.isfinite(prediction).all():raise ValueError('Native pilot must preserve512×64')
    fit=orbit.temporal_components(prediction,pair)
    normalized,mu,sigma=orbit.normalized_input(context,model)
    records=[]
    for bundle in range(8):
        sl=slice(bundle*64,(bundle+1)*64);m=rows[bundle*64]
        if any(row['bundle']!=bundle or row['cell']!=index for index,row in enumerate(rows[sl])):
            raise ValueError('Orbit ordering changed')
        origins=(m['origin_phi'],m['origin_psi'])
        features={name:orbit.mode_summary(fit['z'][sl,index].reshape(8,8),origins,m[amp],name)
                  for index,(name,amp) in enumerate((('phi','a'),('psi','b')))}
        waveform=prediction[sl].reshape(8,8,64)
        records.append(dict(bundle=bundle,features=features,
            primary_waveform_budget=orbit.waveform_budget(waveform,future[sl].reshape(8,8,64),origins,m['scale'],pair[0]),
            scaled_future_MSE=float(np.mean((prediction[sl]-future[sl])**2)/m['scale']**2),
            temporal_residual_energy=float(np.mean(fit['residual'][sl]**2)/m['scale']**2),
            full_waveform_phase_profile=vector_profile(waveform,origins),
            raw_input_phase_profile=vector_profile(context[sl].reshape(8,8,128),origins),
            normalized_input_phase_profile=vector_profile(normalized[sl].reshape(8,8,128),origins),
            native_mu_phase_profile=vector_profile(mu[sl].reshape(8,8,-1),origins),
            native_sigma_phase_profile=vector_profile(sigma[sl].reshape(8,8,-1),origins)))
    return records,(normalized,mu,sigma)


def run(out):
    import torch
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models import build_adapter
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    source=Path(p['core_source'])
    if p['runner_sha256']!=digest(__file__) or p['core_protocol_sha256']!=digest(source/'protocol.json'):
        raise ValueError('Frozen driver/protocol changed')
    if (out/'results.json').exists() or any(out.glob('*_prediction.npz')):raise ValueError('Never overwrite a partial/completed run')
    for path,value in p['sources'].items():
        if digest(path)!=value:raise ValueError('Installed source binding failed')
    core=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if core['driver_sha256']!=digest(orbit.__file__):raise ValueError('Core source changed')
    cases={}
    for stem,binding in core['bindings'].items():
        for suffix,value in binding.items():
            if digest(source/f'{stem}.{suffix}')!=value:raise ValueError('Corpus binding failed')
        data=np.load(source/f'{stem}.npz');rows=json.loads((source/f'{stem}.json').read_text(encoding='utf-8'))
        cases[stem]=(data['context'],data['clean_future'],rows)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(1.,6*2**30/torch.cuda.get_device_properties(0).total_memory))
    started=time.monotonic();results={};telemetry={};all_controls=True
    adapters={'timesfm':'timesfm','bolt':'chronos_bolt','chronos2':'chronos2'}
    for model,manifest in p['models'].items():
        checkpoint=Path(manifest['checkpoint'])
        if digest(checkpoint)!=manifest['weight_sha256'] or digest(checkpoint.parent/'config.json')!=manifest['config_sha256']:
            raise ValueError('Checkpoint binding changed')
        torch.cuda.reset_peak_memory_stats();model_started=time.monotonic()
        adapter=build_adapter(ModelConfig(name=model,adapter=adapters[model],checkpoint=str(checkpoint.parent)),DataConfig(context_len=128,horizon=64),torch.device('cuda:0'),torch.float32)
        adapter.ensure_loaded();results[model]={};controls={}
        if model=='timesfm':native=int(adapter.tfm.model.o)
        elif model=='bolt':native=int(adapter.pipeline.model_prediction_length)
        else:native=16*adapter._num_output_patches(64)
        if native<64:raise ValueError('Native horizon unsupported')
        for stem,(context,future,rows) in cases.items():
            forecasts=[]
            for start in range(0,len(context),8):
                if time.monotonic()-started>600:raise RuntimeError('Ten-minute cap reached')
                with torch.inference_mode():forecasts.append(adapter.predict(context[start:start+8],64,[.5])['point'])
            forecast=np.concatenate(forecasts)
            repeat=adapter.predict(context[:8],64,[.5])['point']
            perm=np.random.default_rng(3500017).permutation(8)
            shuffled=adapter.predict(context[:8][perm],64,[.5])['point'][np.argsort(perm)]
            checks=dict(repeat=float(abs(forecast[:8]-repeat).max()),permutation=float(abs(shuffled-repeat).max()))
            if model=='chronos2':
                isolated=np.concatenate([adapter.predict(row[None],64,[.5])['point'] for row in context[:8]])
                checks['singleton']=float(abs(isolated-repeat).max())
            controls[stem]=checks;all_controls=all_controls and all(value==0 for value in checks.values())
            # Save failed controls too; never hide hard cases or infer modes from them.
            pair=tuple(int(part.lstrip('p')) for part in stem.split('_'))
            records,normalization=analyze_case(forecast,context,future,rows,pair,model)
            closure=max(r['primary_waveform_budget']['error_Parseval_absolute_error'] for r in records)
            oracle_D=[]
            for bundle in range(8):
                row=rows[bundle*64];truth=future[bundle*64:(bundle+1)*64].reshape(8,8,64)
                oracle_D.append(orbit.waveform_budget(truth,truth,(row['origin_phi'],row['origin_psi']),row['scale'],pair[0])['primary_D'])
            checks['error_Parseval_max_abs']=closure;checks['oracle_D_max']=max(oracle_D)
            all_controls=all_controls and closure<=1e-10 and max(oracle_D)<=1e-20
            results[model][stem]=records
            np.savez_compressed(out/f'{model}_{stem}_prediction.npz',prediction=forecast,
                native_normalized_input=normalization[0],native_mu=normalization[1],native_sigma=normalization[2],
                model=np.asarray(model),case=np.asarray(stem),phase_row_ids=row_ids(rows,stem),
                corpus_sha256=np.asarray(digest(source/f'{stem}.npz')),
                metadata_sha256=np.asarray(digest(source/f'{stem}.json')),
                core_protocol_sha256=np.asarray(p['core_protocol_sha256']))
            load_prediction(out/f'{model}_{stem}_prediction.npz',model,stem,source)
        telemetry[model]=dict(wall_seconds=time.monotonic()-model_started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            controls=controls,native_supported_horizon=native,adapter_config=asdict(adapter.cfg))
        adapter._release();del adapter;gc.collect();torch.cuda.empty_cache()
    candidates={}
    for first,second in combinations(p['models'],2):
        for stem in cases:
            a=np.array([r['primary_waveform_budget']['primary_D'] for r in results[first][stem]])
            b=np.array([r['primary_waveform_budget']['primary_D'] for r in results[second][stem]])
            candidates[f'{first}-{second}/{stem}']=orbit.pilot_interestingness(a,b)
    if len(candidates)!=6:raise ValueError('Complete fixed pilot candidate family changed')
    result=dict(records=results,candidates=candidates,controls_pass=bool(all_controls),telemetry=telemetry,
        total_wall_seconds=time.monotonic()-started,
        alias_validation_eligible=bool(all_controls and any(v['promotes_alias_validation'] for v in candidates.values())),
        scope='Exploratory deterministic amplitude grid; no mode claim before alias/offgrid checks; no hidden feature capture or native-ranking rescue')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(controls_pass=all_controls,alias_validation_eligible=result['alias_validation_eligible'],telemetry=telemetry),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--source',type=Path)
    parser.add_argument('--checkpoint-manifest',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:prepare(args.source,args.out,args.checkpoint_manifest)
    else:run(args.out)
