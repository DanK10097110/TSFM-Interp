"""Frozen native-forecast replication with mixed and matched pure compositions.

Four prespecified paired claims use generating bundles, not1024 forecast rows,
with one shared mixed-energy denominator across all composition conditions.
No trainable parameter or model layer is selected in this experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.feature_search.mixture_interference import harmonic_components, input_interference
from tsfm_lens.analysis.stats import holm, mean_ci
from tsfm_lens.experiments.feature_search.statistics import paired_effect

PROTOCOL_PATH = Path(__file__).with_name('mixture_replication_protocol.json')


def load_protocol():
    """Read the prespecified protocol; write its exact bytes before inference."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def make_corpus(protocol=None):
    """Create8 matched compositions per fresh generating seed, with shared nuisances."""
    protocol = load_protocol() if protocol is None else protocol
    contexts,futures,metadata = [],[],[]
    t = np.arange(protocol['context']+protocol['horizon'])
    for index in range(protocol['n_bundles']):
        rng = np.random.default_rng(protocol['seed_base']+index)
        a,b = rng.uniform(*protocol['amplitudes'],2)
        offset = rng.uniform(*protocol['offset'])
        phi,psi = rng.uniform(-np.pi,np.pi,2)
        compositions = [('mixed',p,q) for p in (0,1) for q in (0,1)]
        compositions += [('pure_phi',p,0) for p in (0,1)]+[('pure_psi',0,q) for q in (0,1)]
        for condition,p,q in compositions:
            angle_phi = phi+p*protocol['phase_shift']
            angle_psi = psi+q*protocol['phase_shift']
            value = np.full(len(t),offset)
            if condition != 'pure_psi':
                value += a*np.sin(2*np.pi*t/protocol['periods'][0]+angle_phi)
            if condition != 'pure_phi':
                value += b*np.sin(2*np.pi*t/protocol['periods'][1]+angle_psi)
            contexts.append(value[:protocol['context']]);futures.append(value[protocol['context']:])
            metadata.append(dict(seed=index,generator_seed=protocol['seed_base']+index,split='test',condition=condition,
                                 cell_phi=p,cell_psi=q,phi=angle_phi,psi=angle_psi,a=a,b=b,offset=offset,scale=float(np.sqrt((a*a+b*b)/2))))
    return np.array(contexts,dtype=np.float32),np.array(futures),metadata


def _validate_compositions(truth,metadata,condition_key,*,forecast_start=128):
    """Validate truth, exactly8 cells, nuisance matching and shared mixed energy."""
    if len(truth) != len(metadata):
        raise ValueError('Truth and metadata must align')
    coefficients = harmonic_components(truth,start=forecast_start)['coefficients']
    expected = []
    seen = set()
    for row in metadata:
        condition = row[condition_key]
        if condition not in ('mixed','pure_phi','pure_psi'):
            raise ValueError('Unknown composition condition')
        a,b,scale = row['a'],row['b'],row['scale']
        if min(a,b,scale) <= 0 or not np.isfinite([a,b,scale,row['offset'],row['phi'],row['psi']]).all() or not np.isclose(scale**2,(a*a+b*b)/2):
            raise ValueError('All compositions need the same positive mixed-energy scale')
        key = (row['seed'],condition,row['cell_phi'],row['cell_psi'])
        if key in seen:
            raise ValueError('Duplicate composition cell')
        seen.add(key)
        av = a if condition != 'pure_psi' else 0.
        bv = b if condition != 'pure_phi' else 0.
        expected.append([av*np.cos(row['phi']),av*np.sin(row['phi']),bv*np.cos(row['psi']),bv*np.sin(row['psi']),row['offset']])
    if not np.allclose(coefficients,expected,atol=1e-8,rtol=1e-7):
        raise ValueError('Composition truth disagrees with declared physical components')
    if not np.allclose(harmonic_components(truth,start=forecast_start)['residual'],0,atol=1e-8):
        raise ValueError('Composition truth contains undeclared residual signal')
    seeds = np.unique([row['seed'] for row in metadata])
    for seed in seeds:
        bundle = [row for row in metadata if row['seed']==seed]
        expected_cells = {('mixed',p,q) for p in (0,1) for q in (0,1)}|{('pure_phi',p,0) for p in (0,1)}|{('pure_psi',0,q) for q in (0,1)}
        if {(row[condition_key],row['cell_phi'],row['cell_psi']) for row in bundle} != expected_cells:
            raise ValueError('Need complete8-cell generating bundles')
        for field in ('a','b','offset','scale'):
            if not np.allclose([row[field] for row in bundle],bundle[0][field],atol=1e-12,rtol=1e-12):
                raise ValueError('Pure and mixed conditions must share amplitude,offset and mixed-energy scale')
    return seeds


def analyze(predictions,truth,metadata,*,protocol=None,condition_key='condition'):
    """Evaluate the fixedHolm4 family, preserving seed pairing and untestable claims."""
    protocol = load_protocol() if protocol is None else protocol
    if set(predictions) != {'bolt','timesfm'}:
        raise ValueError('Need both prespecified native model forecasts')
    truth = np.asarray(truth,dtype=float)
    seeds = _validate_compositions(truth,metadata,condition_key)
    if len(seeds)<3:
        raise ValueError('Need at least3 independent generating bundles')
    mixed = np.array([row[condition_key]=='mixed' for row in metadata])
    mixed_rows = [row for row,keep in zip(metadata,mixed) if keep]
    scale = np.array([row['scale'] for row in metadata])
    row_seed = np.array([row['seed'] for row in metadata])
    row_condition = np.array([row[condition_key] for row in metadata])
    errors,interference = {},{}
    for model,prediction in predictions.items():
        prediction = np.asarray(prediction,dtype=float)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError('Need finite native forecasts aligned to corpus')
        loss = np.mean((prediction-truth)**2,axis=1)/scale**2
        errors[model] = {condition:np.array([loss[(row_seed==seed)&(row_condition==condition)].mean() for seed in seeds]) for condition in ('mixed','pure_phi','pure_psi')}
        for target in ('phi','psi'):
            response = input_interference(prediction[mixed],truth[mixed],mixed_rows,target,phase_floor=protocol['phase_floor_relative_to_truth_amplitude'])
            if not np.array_equal(response['seeds'],seeds):
                raise ValueError('Phase and loss comparisons need matched seed bundles')
            interference[model,target] = response['metrics']['other_phase_change_radians']
    pairs = {f'mixed_{target}_untouched_phase_timesfm_minus_bolt':(interference['timesfm',target],interference['bolt',target]) for target in ('phi','psi')}
    pairs['mixed_scaled_mse_timesfm_minus_bolt'] = (errors['timesfm']['mixed'],errors['bolt']['mixed'])
    mixture_gap = errors['timesfm']['mixed']-errors['bolt']['mixed']
    pure_gap = .5*((errors['timesfm']['pure_phi']-errors['bolt']['pure_phi'])+(errors['timesfm']['pure_psi']-errors['bolt']['pure_psi']))
    pairs['model_by_composition_interaction'] = (mixture_gap,pure_gap)
    settings = protocol['inference']
    results = {}
    for name,(a,b) in pairs.items():
        valid = np.isfinite(a)&np.isfinite(b)
        if valid.sum()<3:
            results[name] = dict(p=1.,untestable='fewer than3 jointly identifiable seed bundles',passed=False)
        else:
            result = paired_effect(a[valid],b[valid],seeds[valid],n_boot=settings['n_boot'],n_permutations=settings['n_permutations'],seed=settings['seed'],ci=settings['ci'])
            result['excluded_seeds'] = int((~valid).sum())
            results[name] = result
    adjusted = holm({name:result['p'] for name,result in results.items()})
    for name,result in results.items():
        result['p_holm'] = adjusted[name]
        minimum = protocol['claims'][name]['minimum_practical']
        result['minimum_practical'] = minimum
        result['unit'] = protocol['claims'][name]['unit']
        result['passed'] = bool('untestable' not in result and result['value']>=minimum and result['lo']>0 and result['p_holm']<.05)
    pure_interval = mean_ci(pure_gap,n_boot=settings['n_boot'],seed=settings['seed'],unit='generating_seed_bundle')
    summary = {model:{condition:float(values.mean()) for condition,values in by_condition.items()} for model,by_condition in errors.items()}
    report = dict(evidence='behavioral causal input intervention and predictive model comparison;no hidden-feature uniqueness',
                  n_seeds=len(seeds),n_contexts=len(metadata),family_size=4,primary=results,all_four_pass=all(result['passed'] for result in results.values()),
                  scaled_mse=summary,scaled_mse_units='MSE divided by SAME per-bundle mixed variance (a*a+b*b)/2 across mixed and pure conditions',pure_gap_descriptive_interval=pure_interval,
                  ranking_flip_descriptive=bool(results['mixed_scaled_mse_timesfm_minus_bolt'].get('lo',-np.inf)>0 and pure_interval['hi']<0),
                  warning='Positive model×composition interaction alone does not imply a model ranking flip;pure-gap interval is descriptive')
    return report


def run(out,alignment_source):
    """Run one offline native forecast job after persisting the exact frozen protocol."""
    import torch
    from experiments.feature_search.activation_interchange import patched_predict
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    protocol = load_protocol()
    out.mkdir(parents=True,exist_ok=False)
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    (out/'protocol.json').write_bytes(protocol_bytes)
    contexts,truth,metadata = make_corpus(protocol)
    np.savez_compressed(out/'corpus.npz',context=contexts,clean_future=truth)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    alignment_bytes = alignment_source.read_bytes()
    provenance = dict(protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),alignment_reused_from=str(alignment_source),
                      alignment_sha256=hashlib.sha256(alignment_bytes).hexdigest(),alignment_reuse='same native adapters/checkpoints FP32 context128 horizon32;no hidden interventions',models={})
    predictions = {}
    for name,cls,checkpoint in [('bolt',ChronosBoltAdapter,'amazon/chronos-bolt-small'),('timesfm',TimesFMAdapter,'google/timesfm-2.5-200m-pytorch')]:
        adapter = cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32)
        adapter.ensure_loaded()
        prediction = patched_predict(adapter,contexts)
        repeat = patched_predict(adapter,contexts[:8])
        repeat_error = float(np.max(np.abs(repeat-prediction[:8])))
        if repeat_error != 0:
            raise RuntimeError(f'Native same-batch deterministic repeat failed{name}: {repeat_error}')
        predictions[name] = prediction
        np.savez_compressed(out/f'{name}_predictions.npz',prediction=prediction)
        provenance['models'][name] = dict(checkpoint=checkpoint,repeat_max_abs=repeat_error,dtype='float32',batch=8)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        print(json.dumps(dict(model=name,completed_contexts=len(contexts),repeat_max_abs=repeat_error)),flush=True)
        del adapter
        gc.collect();torch.cuda.empty_cache()
    report = analyze(predictions,truth,metadata,protocol=protocol)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--alignment-source',type=Path,required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(args.out,args.alignment_source)
