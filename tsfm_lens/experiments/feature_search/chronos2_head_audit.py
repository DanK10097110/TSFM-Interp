"""Accurate-comparator head audit with an explicit common sinh secant allocation.

Chronos-2 applies sinh AFTER adding skip and MLP outputs. The branches cannot
be inverted separately. A common stable divided difference allocates the
native finite difference exactly, with normalization and replay residuals
retained. Absolute energies share the same analytic denominator across models;
near-zero native totals never produce unstable projection-share claims.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from head_branch_audit import bootstrap,derivative,normalizers,off_coefficients


def common_secant(plus,minus):
    """Return cosh(mid)*sinh(half-difference)/half-difference with a continuous limit."""
    plus,minus=np.asarray(plus,dtype=float),np.asarray(minus,dtype=float);mid=(plus+minus)/2;half=(plus-minus)/2
    factor=np.ones_like(half);large=abs(half)>1e-6;np.divide(np.sinh(half),half,out=factor,where=large);factor[~large]=1+half[~large]**2/6+half[~large]**4/120
    value=np.cosh(mid)*factor
    if not np.isfinite(value).all():raise ValueError('Nonfinite common inverse secant')
    return value


def median_point(flat):
    """Keep BOTH forecast patches and select quantile10 from each [21,16] block."""
    flat=np.asarray(flat)
    if flat.ndim!=3 or flat.shape[1:]!=(2,336):raise ValueError('Require exact [batch,2,336] Chronos-2 head outputs')
    return flat.reshape(len(flat),2,21,16)[:,:,10,:].reshape(len(flat),32)


def replay_branches(heads,weights):
    """Evaluate the verified ReLU residual head without dropping a forecast patch."""
    if np.asarray(heads).shape[1:]!=(2,768):raise ValueError('Require full [batch,2,768] forecast-placeholder input')
    h=torch.tensor(heads,dtype=torch.float64);w={k:torch.tensor(v,dtype=torch.float64) for k,v in weights.items()};hidden=torch.nn.functional.relu(torch.nn.functional.linear(h,w['hidden_layer.weight'],w['hidden_layer.bias']))
    nonlinear=torch.nn.functional.linear(hidden,w['output_layer.weight'],w['output_layer.bias']);skip=torch.nn.functional.linear(h,w['residual_layer.weight'],w['residual_layer.bias'])
    return median_point(skip.numpy()),median_point(nonlinear.numpy())


def replay_gate(forecast_rms,actual,replayed,strength):
    """Apply the frozen relative OR analytic-strength absolute derivative replay gate."""
    mismatch=float(np.sum((actual-replayed)**2));energy=float(np.sum(actual**2));relative=None if energy==0 else float(np.sqrt(mismatch/energy));absolute=float(np.sqrt(mismatch/np.sum(strength)))
    return dict(relative_RMS=relative,absolute_RMS_over_analytic_strength=absolute,passed=bool(forecast_rms<=1e-5 and ((relative is not None and relative<=.01) or absolute<=1e-5)))


def absolute_attribution(total,parts,strength):
    """Report common-denominator energies and only stable total-relative projections."""
    energy=np.sum(total**2,axis=1);ratio=bootstrap(energy,strength);stable=ratio['value']>=1e-4;result=dict(total_energy_ratio=ratio,projection_shares_stable=stable,branches={},cross_energy_ratio={})
    for name,value in parts.items():
        share=bootstrap(np.sum(total*value,axis=1),energy) if stable else dict(value=None,reason='total off-frequency energy / analytic strength <1e-4;projection unstable')
        result['branches'][name]=dict(energy_ratio=bootstrap(np.sum(value**2,axis=1),strength),signed_projection_share=share)
    for a,b in combinations(parts,2):result['cross_energy_ratio'][a+'__'+b]=bootstrap(2*np.sum(parts[a]*parts[b],axis=1),strength)
    skip=result['branches']['skip']['energy_ratio']['value'];nl=result['branches']['nonlinear']['energy_ratio']['value'];cross=result['cross_energy_ratio']['skip__nonlinear']['value']
    if skip<.02 and nl<.02:label='weak branch mixing'
    elif skip>.05 and nl>.05 and ratio['value']<.02 and -cross>=.5*(skip+nl):label='cancellation'
    else:label='mixed/unresolved'
    result['descriptive_label']=label;result['label_scope']='prespecified point-estimate rule;CIs shown;not a unique causal attribution';result['coefficient_closure_max_abs']=float(abs(total-sum(parts.values())).max())
    return result


def capture_native_heads(adapter,module_name,context,positions,width,*,independent_groups=False,permutation_seed=1659000):
    """Capture exact native head inputs with repeat,hook and batch-order controls."""
    module=dict(adapter.module.named_modules())[module_name]
    def predict(batch):
        if not independent_groups:return adapter.predict(batch,32,[.5])['point']
        def groups(m,args,kwargs):
            kwargs=dict(kwargs);kwargs['group_ids']=torch.arange(len(batch),device=adapter.device,dtype=torch.long);return args,kwargs
        handle=adapter.module.register_forward_pre_hook(groups,with_kwargs=True)
        try:return adapter.predict(batch,32,[.5])['point']
        finally:handle.remove()
    controls=dict(default_vs_independent=0.,repeat=0.,hook=0.,batch_permutation=0.);heads=[];predictions=[]
    for start in range(0,len(context),8):
        batch=context[start:start+8];hits=[]
        with torch.inference_mode():
            default=adapter.predict(batch,32,[.5])['point'];native=predict(batch) if independent_groups else default;repeat=predict(batch)
            def capture(m,inputs):
                if not isinstance(inputs,tuple) or len(inputs)!=1 or not isinstance(inputs[0],torch.Tensor) or tuple(inputs[0].shape)!=(len(batch),positions,width):raise ValueError('Unexpected exact native head input container/shape')
                hits.append(inputs[0].detach().float().cpu().numpy().copy())
            handle=module.register_forward_pre_hook(capture)
            try:hooked=predict(batch)
            finally:handle.remove()
            permutation=np.random.default_rng(permutation_seed+start).permutation(len(batch));permuted=predict(batch[permutation])[np.argsort(permutation)]
        if len(hits)!=1:raise ValueError('Expected one native head call per batch')
        for key,a,b in [('default_vs_independent',default,native),('repeat',native,repeat),('hook',native,hooked),('batch_permutation',native,permuted)]:controls[key]=max(controls[key],float(abs(a-b).max()))
        heads.append(hits[0]);predictions.append(native)
    return np.concatenate(heads),np.concatenate(predictions),controls


def run(source,previous,checkpoint,out,prepare_only=False):
    """Capture one cached native model,then compare all three heads in identical units."""
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    import chronos.chronos2.model as model_source
    import chronos.chronos2.layers as layer_source
    import chronos.chronos_bolt as norm_source
    source_protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'));conditions=source_protocol['conditions'];metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'));corpus=np.load(source/'corpus.npz');context=corpus['context'];future=corpus['clean_future']
    if source_protocol['seed_base']!=1659000 or len(context)!=640 or source_protocol['targets']!=['psi']:raise ValueError('Require all frozen640 inputs from128-seed psi replication')
    prior_protocol=json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    if prior_protocol['source_protocol_sha256']!=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest():raise ValueError('Prior branch audit is not bound to this exact source protocol')
    if any(prior_protocol['capture_sha256'][model]!=hashlib.sha256((source/f'{model}_capture.npz').read_bytes()).hexdigest() for model in ['bolt','timesfm']):raise ValueError('Prior model captures do not match this exact corpus cache')
    config=json.loads((checkpoint.parent/'config.json').read_text(encoding='utf-8'))
    if config['chronos_config']['use_arcsinh'] is not True or config['dense_act_fn']!='relu':raise ValueError('Unexpected Chronos-2 inverse or activation')
    with safe_open(checkpoint,framework='np') as handle:weights={k[len('output_patch_embedding.'):]:handle.get_tensor(k) for k in handle.keys() if k.startswith('output_patch_embedding.')}
    expected={'hidden_layer.weight','hidden_layer.bias','output_layer.weight','output_layer.bias','residual_layer.weight','residual_layer.bias'}
    if set(weights)!=expected:raise ValueError('Unexpected Chronos-2 head tensor set or head normalization')
    protocol=dict(source=str(source),audit_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_seeds=128,n_inputs=640,primary_delta=.01,deltas=[.01,.02],head_input_shape=['batch',2,768],head_output_shape=['batch',2,336],indexing='two forecast patches;each quantile-major21x16;median index10;concatenate patches',
        source_hashes={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in [('model',model_source),('layers',layer_source),('normalization',norm_source)]},checkpoint=str(checkpoint),tensor_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in weights.items()},
        corpus_sha256=hashlib.sha256((source/'corpus.npz').read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),prior_branch_data_sha256=hashlib.sha256((previous/'source_data.npz').read_bytes()).hexdigest(),
        inverse='sinh(skip+MLP),then population scale and mean once',allocation='common stable secant cosh(mid)*sinh(halfdelta)/halfdelta;continuous limit;shared-inverse allocation convention,not unique',
        controls='native/default vs explicit independent groupIDs;repeat;head-capture;batch-permutation all exactly0',replay_gates=dict(forecast_normalized_RMS=1e-5,derivative_relative_RMS=.01,derivative_absolute_RMS_over_analytic_strength=1e-5),
        units='all model branch,total and cross energies divided by SAME sum analytic own-feature strength²',projection_floor=1e-4,label_rule='weak if both branch ratios<.02;cancellation if both>.05,total<.02,negative cross offsets>=50%sumbranch energies;else mixed/unresolved',
        evidence='descriptive native computation attribution;no probes,learned fits,gate rescue or repair')
    if out.exists():
        if {p.name for p in out.iterdir()}!={'protocol.json'} or json.loads((out/'protocol.json').read_text(encoding='utf-8'))!=protocol:raise ValueError('Output is not an unchanged prepared protocol-only directory')
    else:
        out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if prepare_only:
        print('Prospective protocol and hashes saved;no model constructed or inference run.',flush=True);return
    adapter=Chronos2Adapter(ModelConfig(name='chronos2',checkpoint=str(checkpoint.parent)),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded()
    h,native,controls=capture_native_heads(adapter,'output_patch_embedding',context,2,768,independent_groups=True);np.savez_compressed(out/'chronos2_capture.npz',head_inputs=h,prediction=native);del adapter;gc.collect();torch.cuda.empty_cache()
    report=dict(capture_controls=controls,capture_passed=all(v==0 for v in controls.values()),models={},evidence=protocol['evidence'])
    if not report['capture_passed']:
        report['skip_reason']='Native identity/group/batch controls failed;no branch attribution';(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));return
    skip,nl=replay_branches(h,weights);loc,scale=normalizers(context,'bolt');z=skip+nl;replayed=np.sinh(z)*scale[:,None]+loc[:,None];base_meta=metadata[::5];latent=np.array([m['scale'] for m in metadata]);base_latent=latent.reshape(128,5)[:,0];base_native_scale=scale.reshape(128,5)[:,0];strength=np.array([(m['b']/m['scale'])**2 for m in base_meta]);forecast_rms=float(np.sqrt(np.mean(((native-replayed)/latent[:,None])**2)));components={};checks={};saved={}
    for delta in [.01,.02]:
        plus=next(i for i,c in enumerate(conditions) if c[1:]==['psi',1,delta]);minus=next(i for i,c in enumerate(conditions) if c[1:]==['psi',-1,delta]);cells=z.reshape(128,5,32);j=common_secant(cells[:,plus],cells[:,minus]);actual=derivative(native,conditions,delta);rep=derivative(replayed,conditions,delta);skip_d=derivative(skip,conditions,delta)*j*base_native_scale[:,None];nl_d=derivative(nl,conditions,delta)*j*base_native_scale[:,None]
        total=off_coefficients(actual,base_latent);replay_off=off_coefficients(rep,base_latent);checks[str(delta)]=replay_gate(forecast_rms,total,replay_off,strength);parts={name:off_coefficients(value,base_latent) for name,value in dict(skip=skip_d,nonlinear=nl_d,normalization_residual=rep-skip_d-nl_d,numerical_replay_residual=actual-rep).items()};components[delta]=(total,parts)
    report['chronos2_replay']=dict(forecast_normalized_RMS=forecast_rms,deltas=checks,passed=all(v['passed'] for v in checks.values()))
    if not report['chronos2_replay']['passed']:
        report['skip_reason']='Native replay failed;no attribution inferred';(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report));return
    previous_data=np.load(previous/'source_data.npz')
    for model in ['bolt','timesfm','chronos2']:
        result={}
        for delta in [.01,.02]:
            if model=='chronos2':total,parts=components[delta]
            else:
                stem=model+'_'+str(delta)+'_';total=previous_data[stem+'native_off_coefficient_derivative'];parts={name:previous_data[stem+name] for name in ['skip','nonlinear','normalization_residual','numerical_replay_residual']}
            if total.shape!=(128,2):raise ValueError('Prior model seed/coefficient layout does not match current128 bundles')
            result[str(delta)]=absolute_attribution(total,parts,strength);saved[model+'_'+str(delta)+'_native']=total
            for name,value in parts.items():saved[model+'_'+str(delta)+'_'+name]=value
        report['models'][model]=result
    report['denominator']=dict(analytic_strength_sum=float(strength.sum()),analytic_strength_mean=float(strength.mean()),unit='normalized true psi16 coefficient derivative squared;identical128seeds for all3models');np.savez_compressed(out/'source_data.npz',analytic_strength=strength,**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--previous-branches',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();torch.set_num_threads(1 if a.prepare_only else 2)
    if not a.prepare_only and torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(a.source,a.previous_branches,a.weights,a.out,a.prepare_only)
