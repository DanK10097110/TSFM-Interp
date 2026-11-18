"""Exact native head-branch replay along cached natural phase tangents.

Only head tensors are loaded. Signed projection shares describe native skip,
nonlinear, normalization and replay-residual contributions; branch energies
are not additive without their cross terms. This is descriptive computation
attribution, not a probe-gate rescue, unique feature claim or repair experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from mixture_features import harmonic_fit


def point_channel(values,model):
    """Select the verified median channel from each native head's flat output."""
    x=np.asarray(values)
    if x.ndim!=2:raise ValueError('Head output must be [batch,flat_output] after selecting last head position')
    if model=='timesfm' and x.shape[1]==1280:return x.reshape(len(x),128,10)[:,:32,5]
    if model=='bolt' and x.shape[1]==576:return x.reshape(len(x),9,64)[:,4,:32]
    raise ValueError('Unexpected model or native head output width')


def branches(head_input,weights,activation):
    """Replay source-defined skip and nonlinear branches with every branch bias."""
    h=torch.as_tensor(np.asarray(head_input),dtype=torch.float64)
    if h.ndim!=3:raise ValueError('Need full native [batch,position,width] head input')
    h=h[:,-1];w={k:torch.as_tensor(v,dtype=torch.float64) for k,v in weights.items()}
    hidden=torch.nn.functional.linear(h,w['hidden_layer.weight'],w.get('hidden_layer.bias'))
    if activation=='silu':hidden=torch.nn.functional.silu(hidden)
    elif activation=='relu':hidden=torch.nn.functional.relu(hidden)
    else:raise ValueError('Unsupported verified head activation')
    nonlinear=torch.nn.functional.linear(hidden,w['output_layer.weight'],w.get('output_layer.bias'))
    skip=torch.nn.functional.linear(h,w['residual_layer.weight'],w.get('residual_layer.bias'))
    return skip.numpy(),nonlinear.numpy()


def normalizers(context,model):
    """Replay source population statistics in float32 before returning float64 values."""
    x=torch.tensor(np.asarray(context),dtype=torch.float32)
    if model=='bolt':
        loc=torch.nan_to_num(torch.nanmean(x,dim=-1),nan=0.);scale=torch.nan_to_num(torch.nanmean((x-loc[:,None])**2,dim=-1).sqrt(),nan=1.);scale=torch.where(scale==0,1e-5,scale)
    elif model=='timesfm':
        n=torch.zeros(len(x));loc=torch.zeros(len(x));scale=torch.zeros(len(x))
        if x.shape[1]%32:raise ValueError('TimesFM requires complete32-step patches')
        for patch in x.reshape(len(x),-1,32).unbind(1):
            inc_n=torch.full_like(n,32.);inc_mu=patch.sum(-1)/inc_n;inc_sigma=(((patch-inc_mu[:,None])**2).sum(-1)/inc_n).sqrt();new_n=n+inc_n;new_mu=(n*loc+inc_mu*inc_n)/new_n
            variance=(n*scale**2+inc_n*inc_sigma**2+n*(loc-new_mu)**2+inc_n*(inc_mu-new_mu)**2)/new_n
            n,loc,scale=new_n,new_mu,variance.clamp_min(0).sqrt()
    else:raise ValueError('Unknown normalization model')
    return loc.numpy().astype(float),scale.numpy().astype(float)


def derivative(values,conditions,delta):
    """Use all cached seed bundles and the fixed psi±delta condition identities."""
    plus=next(i for i,c in enumerate(conditions) if c[1:]==['psi',1,delta]);minus=next(i for i,c in enumerate(conditions) if c[1:]==['psi',-1,delta]);values=np.asarray(values,dtype=float)
    if len(values)%len(conditions):raise ValueError('Incomplete cached perturbation bundle')
    cells=values.reshape((-1,len(conditions))+values.shape[1:]);return (cells[:,plus]-cells[:,minus])/(2*delta)


def off_coefficients(future_derivative,latent_scale):
    """Project onto period32,which is off-target for the fixed period16 perturbation."""
    fit=harmonic_fit(future_derivative)['coefficients'];return fit[:,[1,0]]/latent_scale[:,None]


def bootstrap(values,denominator=None):
    """Use paired seed resampling for means or ratios of aggregate sums."""
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(165901);indices=rng.integers(0,len(values),(2000,len(values)))
    if denominator is None:point=values.mean();draws=values[indices].mean(1)
    else:
        denominator=np.asarray(denominator,dtype=float)
        if denominator.sum()<=0:raise ValueError('Native total energy is zero;attribution undefined')
        point=values.sum()/denominator.sum();draws=values[indices].sum(1)/denominator[indices].sum(1)
    return dict(value=float(point),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),n_seeds=len(values))


def attribution(total,components):
    """Return exact signed projection shares,raw energies and all cross energies."""
    total=np.asarray(total);energy=np.sum(total**2,axis=1);result=dict(total_energy=bootstrap(energy),branches={},cross_energy={})
    summed=np.zeros_like(total)
    for name,value in components.items():
        summed+=value;result['branches'][name]=dict(signed_projection_share=bootstrap(np.sum(total*value,axis=1),energy),raw_energy=bootstrap(np.sum(value**2,axis=1)))
    for a,b in combinations(components,2):result['cross_energy'][a+'__'+b]=bootstrap(2*np.sum(components[a]*components[b],axis=1))
    result['coefficient_closure_max_abs']=float(abs(total-summed).max());result['projection_share_sum']=float(sum(v['signed_projection_share']['value'] for v in result['branches'].values()))
    result['energy_closure_error']=float(energy.mean()-sum(v['raw_energy']['value'] for v in result['branches'].values())-sum(v['value'] for v in result['cross_energy'].values()))
    return result


def audit(source,capture_provenance,weight_paths,out):
    """Load only checkpoint head tensors and audit cached native computations on CPU."""
    out.mkdir(parents=True,exist_ok=False);source_protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'));source_result=json.loads((source/'results.json').read_text(encoding='utf-8'));provenance=json.loads(capture_provenance.read_text(encoding='utf-8'))
    if source_protocol['seed_base']!=1659000 or source_protocol['n_seeds']!=128 or source_protocol['targets']!=['psi']:raise ValueError('Require the complete frozen1659000psi replication cache')
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'));context=np.load(source/'corpus.npz')['context'];conditions=source_protocol['conditions'];n_conditions=len(conditions);base_scale=np.array([m['scale'] for m in metadata[::n_conditions]])
    protocol=dict(source=str(source),source_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),capture_sha256={model:hashlib.sha256((source/f'{model}_capture.npz').read_bytes()).hexdigest() for model in ['bolt','timesfm']},evidence='descriptive native-computation attribution;no probe comparisons,rescue,repair or unique-feature claims',all_conditions=True,primary_delta=.01,other_delta=.02,
        replay_gates=dict(forecast_latent_normalized_RMS=1e-5,off_frequency_derivative_relative_RMS=.01),precision='float64 head arithmetic on captured float32 inputs/checkpoint tensors;native float32 normalizers',
        attribution='sum <native_total,branch> / sum ||native_total||²;include all pairwise cross energies;no significance tests',
        components=['skip at base scale','nonlinear at base scale','normalization residual','numerical replay residual'],native_indexing=dict(timesfm='reshape[128,10],channel5,first32',bolt='reshape[9,64],quantile4,first32'),models={})
    loaded={}
    for model in ['bolt','timesfm']:
        prefix='output_patch_embedding' if model=='bolt' else 'output_projection_point';path=weight_paths[model];model_source=Path(provenance[model]['source_path']);source_hash=hashlib.sha256(model_source.read_bytes()).hexdigest()
        if source_hash!=source_result['models'][model]['source_sha256']:raise ValueError('Installed model source differs from capture source')
        with safe_open(path,framework='np') as handle:weights={k[len(prefix)+1:]:handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix+'.')}
        expected={'hidden_layer.weight','output_layer.weight','residual_layer.weight'}|({'hidden_layer.bias','output_layer.bias','residual_layer.bias'} if model=='bolt' else set())
        if set(weights)!=expected:raise ValueError(f'Unexpected native head tensor set:{model}:{set(weights)}')
        additional_sources={'model':model_source}
        if model=='timesfm':additional_sources.update(dense=model_source.parents[1]/'torch/dense.py',normalization=model_source.parents[1]/'torch/util.py')
        protocol['models'][model]=dict(checkpoint=str(path),tensor_shapes={k:list(v.shape) for k,v in weights.items()},tensor_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in weights.items()},source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in additional_sources.items()},activation='relu' if model=='bolt' else 'silu',biases='both branches included' if model=='bolt' else 'none',dropout='evaluation identity',layer_norm=False,normalization='native affine population scaling;mean added exactly once')
        loaded[model]=weights
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');report=dict(models={},scope=protocol['evidence']);saved={}
    for model,weights in loaded.items():
        cache=np.load(source/f'{model}_capture.npz');h=cache['head_inputs'];native=cache['prediction'];raw_skip,raw_nonlinear=branches(h,weights,protocol['models'][model]['activation']);skip=point_channel(raw_skip,model);nonlinear=point_channel(raw_nonlinear,model);loc,scale=normalizers(context,model);replayed=(skip+nonlinear)*scale[:,None]+loc[:,None]
        latent=np.array([m['scale'] for m in metadata]);forecast_error=float(np.sqrt(np.mean(((replayed-native)/latent[:,None])**2)));base_native_scale=scale.reshape(128,n_conditions)[:,0];checks={};components={}
        for delta in [.01,.02]:
            actual=derivative(native,conditions,delta);rep=derivative(replayed,conditions,delta);actual_coeff=off_coefficients(actual,base_scale);rep_coeff=off_coefficients(rep,base_scale);den=np.sum(actual_coeff**2)
            relative=None if den==0 else float(np.sqrt(np.sum((actual_coeff-rep_coeff)**2)/den));checks[str(delta)]=dict(off_frequency_derivative_relative_RMS=relative)
            skip_d=derivative(skip,conditions,delta)*base_native_scale[:,None];nonlinear_d=derivative(nonlinear,conditions,delta)*base_native_scale[:,None]
            wave_components=dict(skip=skip_d,nonlinear=nonlinear_d,normalization_residual=rep-skip_d-nonlinear_d,numerical_replay_residual=actual-rep);components[delta]=(actual_coeff,{k:off_coefficients(v,base_scale) for k,v in wave_components.items()})
        passed=bool(forecast_error<=1e-5 and all(v['off_frequency_derivative_relative_RMS'] is not None and v['off_frequency_derivative_relative_RMS']<=.01 for v in checks.values()));result=dict(forecast_latent_normalized_RMS=forecast_error,replay_checks=checks,replay_passed=passed,deltas={})
        if passed:
            for delta,(total,parts) in components.items():
                result['deltas'][str(delta)]=attribution(total,parts)
                saved[model+'_'+str(delta)+'_native_off_coefficient_derivative']=total
                for name,value in parts.items():saved[model+'_'+str(delta)+'_'+name]=value
        else:result['attribution_skip_reason']='Native replay gate failed;no branch attribution inferred'
        report['models'][model]=result;saved[model+'_replayed_forecast']=replayed;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({model:result}),flush=True)
    np.savez_compressed(out/'source_data.npz',**saved)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--capture-provenance',type=Path,required=True);p.add_argument('--bolt-weights',type=Path,required=True);p.add_argument('--timesfm-weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1);audit(args.source,args.capture_provenance,{'bolt':args.bolt_weights,'timesfm':args.timesfm_weights},args.out)
