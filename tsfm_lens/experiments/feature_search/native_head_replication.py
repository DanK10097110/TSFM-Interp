"""One frozen fresh replication of three-model native-head phase mixing.

No probe or hidden intervention is used. Exact conditional sign tests concern
the majority direction of paired seed energy differences. Joint bootstrap
ratio intervals are approximate; those two estimands do not establish
assumption-free mean superiority. Native branch attribution remains descriptive.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from safetensors import safe_open
from head_branch_audit import branches,point_channel,normalizers,derivative,off_coefficients,bootstrap
from chronos2_head_audit import common_secant,median_point,replay_branches,replay_gate,absolute_attribution,capture_native_heads

CONDITIONS=[('base',None,0,0.)]+[(f'psi_{sign}_{delta}','psi',sign,delta) for delta in [.01,.02] for sign in [1,-1]]


def make_fresh_corpus(n_seeds=128):
    """Generate disjoint2225000bundles with all five frozen native-input conditions."""
    context=[];future=[];metadata=[];t=np.arange(160)
    for seed in range(n_seeds):
        generator_seed=2225000+seed;rng=np.random.default_rng(generator_seed);a,b=rng.uniform(.5,2,2);offset=rng.uniform(-2,2);phi,psi=rng.uniform(-np.pi,np.pi,2);scale=np.sqrt((a*a+b*b)/2)
        for condition,target,sign,delta in CONDITIONS:
            angle=psi+sign*delta;x=offset+a*np.sin(2*np.pi*t/32+phi)+b*np.sin(2*np.pi*t/16+angle);context.append(x[:128]);future.append(x[128:]);metadata.append(dict(seed=seed,generator_seed=generator_seed,condition=condition,a=a,b=b,offset=offset,phi=phi,psi=angle,scale=scale))
    return np.asarray(context,dtype=np.float32),np.asarray(future),metadata


def exact_sign_counts(differences):
    """Test conditional majority among non-ties;never replace signs by the mean sign."""
    d=np.asarray(differences,dtype=float)
    if d.ndim!=1 or not np.isfinite(d).all():raise ValueError('Need finite paired seed numerator differences')
    positive=int(np.sum(d>0));negative=int(np.sum(d<0));ties=int(np.sum(d==0));n=positive+negative
    if not n:return dict(positive=0,negative=0,ties=ties,n_non_ties=0,q=None,q_cp95=[0.,1.],p=1.,majority_positive=False)
    tested=binomtest(positive,n,p=.5,alternative='two-sided');ci=tested.proportion_ci(confidence_level=.95,method='exact')
    return dict(positive=positive,negative=negative,ties=ties,n_non_ties=n,q=positive/n,q_cp95=[float(ci.low),float(ci.high)],p=float(tested.pvalue),majority_positive=positive>n/2)


def paired_numerator_effect(a,b,strength):
    """Combine a ratio-of-sums interval with an exact paired ENERGY-numerator sign test."""
    difference=np.asarray(a)-np.asarray(b);effect=bootstrap(difference,np.asarray(strength));effect['sign_test']=exact_sign_counts(difference)
    effect['scope']='ratio interval approximate;exact sign test establishes conditional majority,not assumption-free mean superiority'
    return effect


def finalize_comparisons(comparisons,gates):
    """Apply the complete frozen Holm2 family,practical effects and majority directions."""
    from tsfm_lens.analysis.stats import holm
    minimums={'timesfm_minus_chronos2':.10,'bolt_minus_chronos2':.02}
    if set(comparisons)!=set(minimums):raise ValueError('Need the complete two-comparison family')
    adjusted=holm({key:value['sign_test']['p'] for key,value in comparisons.items()})
    for key,effect in comparisons.items():
        model=key.split('_minus_')[0];minimum=minimums[key];effect['minimum_point_effect']=minimum;effect['p_holm']=adjusted[key];effect['passed']=bool(gates[model] and gates['chronos2'] and effect['value']>=minimum and effect['lo']>0 and effect['sign_test']['majority_positive'] and effect['p_holm']<.05)
    return comparisons


def setup(weight_paths,out,prepare_only):
    """Freeze distributions,controls,hashes and the complete two-claim family before inference."""
    import chronos.chronos_bolt as bolt_source
    import chronos.chronos2.model as c2_source
    import chronos.chronos2.layers as c2_layers
    import timesfm.timesfm_2p5.timesfm_2p5_torch as tf_source
    import timesfm.torch.dense as tf_dense
    import timesfm.torch.util as tf_norm
    weights={};manifests={}
    for model,path in weight_paths.items():
        prefix='output_projection_point' if model=='timesfm' else 'output_patch_embedding'
        with safe_open(path,framework='np') as handle:weights[model]={k[len(prefix)+1:]:handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix+'.')}
        expected={'hidden_layer.weight','output_layer.weight','residual_layer.weight'}|({'hidden_layer.bias','output_layer.bias','residual_layer.bias'} if model!='timesfm' else set())
        if set(weights[model])!=expected:raise ValueError('Unexpected exact native head tensor set')
        sources={'bolt':[bolt_source],'timesfm':[tf_source,tf_dense,tf_norm],'chronos2':[c2_source,c2_layers,bolt_source]}[model]
        manifests[model]=dict(checkpoint=str(path),tensor_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in weights[model].items()},source_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in sources})
    c2_config=json.loads((weight_paths['chronos2'].parent/'config.json').read_text(encoding='utf-8'))
    if c2_config['chronos_config']['use_arcsinh'] is not True:raise ValueError('Chronos-2 common sinh inverse convention no longer matches checkpoint')
    protocol=dict(seed_base=2225000,n_seeds=128,n_contexts_per_model=640,periods=[32,16],context=128,horizon=32,conditions=CONDITIONS,amplitudes=[.5,2],offset=[-2,2],phase=[-float(np.pi),float(np.pi)],noise=0,
        primary_delta=.01,convergence_delta=.02,claims={'timesfm_minus_chronos2':.10,'bolt_minus_chronos2':.02},
        inference='global Holm2 of TWO-SIDED exact sign tests on paired per-seed energy numerator differences;positive conditional majority;ratio-of-aggregate-energies bootstrap lower95>0;minimum point effects fixed',bootstrap_draws=2000,bootstrap_seed=165901,sign_test='scipy binomtest p=.5,two-sided;Clopper-Pearson95 exact;ties explicitly counted',
        caveat='Sign tests establish conditional majority,not assumption-free mean superiority;ratio intervals approximate;ties reported and retained in ratio estimates',
        controls='native repeat,capture hook,batch permutation,independent Chronos-2 groupIDs all exact0',replay_gates=dict(forecast_normalized_RMS=1e-5,derivative_relative_RMS=.01,derivative_absolute_RMS_over_analytic_strength=1e-5),
        branch_scope='descriptive only;common sinh secant for Chronos-2;absolute branch energies over shared analytic target-strength sum',projection_floor=1e-4,
        labels='weak if bothbranchratios<.02;cancellation if both>.05,total<.02,negative cross offsets>=50%sumbranch energies;else mixed/unresolved',
        no_probes=True,no_hidden_interventions=True,manifests=manifests,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),capture_helper_sha256=hashlib.sha256(Path(capture_native_heads.__code__.co_filename).read_bytes()).hexdigest(),replay_helper_sha256=hashlib.sha256(Path(branches.__code__.co_filename).read_bytes()).hexdigest())
    context,future,metadata=make_fresh_corpus()
    if out.exists():
        if {p.name for p in out.iterdir()}!={'protocol.json','corpus.npz','metadata.json'}:raise ValueError('Output is not a prepared-only fresh replication directory')
        previous=json.loads((out/'protocol.json').read_text(encoding='utf-8'));binding={k:previous.pop(k) for k in ['corpus_sha256','metadata_sha256']}
        if previous!=json.loads(json.dumps(protocol)) or binding['corpus_sha256']!=hashlib.sha256((out/'corpus.npz').read_bytes()).hexdigest() or binding['metadata_sha256']!=hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest():raise ValueError('Prepared protocol or exact corpus binding changed')
        stored=np.load(out/'corpus.npz')
        if not np.array_equal(stored['context'],context) or not np.array_equal(stored['clean_future'],future):raise ValueError('Frozen fresh generator changed')
    else:
        out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'corpus.npz',context=context,clean_future=future);(out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');protocol.update(corpus_sha256=hashlib.sha256((out/'corpus.npz').read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest());(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    return weights,context,future,metadata


def run(weight_paths,out,prepare_only=False):
    """Capture all three native models sequentially and evaluate only the frozen two claims."""
    weights,context,future,metadata=setup(weight_paths,out,prepare_only)
    if prepare_only:print('Prospective fresh protocol,corpus and hashes saved;no model constructed.',flush=True);return
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    conditions=json.loads((out/'protocol.json').read_text(encoding='utf-8'))['conditions'];base=metadata[::5];base_scale=np.array([m['scale'] for m in base]);latent=np.array([m['scale'] for m in metadata]);strength=np.array([(m['b']/m['scale'])**2 for m in base]);report=dict(models={},comparisons={},scope='fresh native-computation replication;no probes or interventions');energies={};saved={}
    for model,cls,module_name,positions,width in [('bolt',ChronosBoltAdapter,'output_patch_embedding',1,512),('timesfm',TimesFMAdapter,'output_projection_point',4,1280),('chronos2',Chronos2Adapter,'output_patch_embedding',2,768)]:
        adapter=cls(ModelConfig(name=model,checkpoint=str(weight_paths[model].parent)),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded();h,native,controls=capture_native_heads(adapter,module_name,context,positions,width,independent_groups=model=='chronos2',permutation_seed=2225000);np.savez_compressed(out/f'{model}_capture.npz',head_inputs=h,prediction=native);del adapter;gc.collect();torch.cuda.empty_cache()
        if model=='chronos2':skip,nl=replay_branches(h,weights[model])
        else:
            raw_skip,raw_nl=branches(h,weights[model],'silu' if model=='timesfm' else 'relu');skip,nl=point_channel(raw_skip,model),point_channel(raw_nl,model)
        loc,scale=normalizers(context,'timesfm' if model=='timesfm' else 'bolt');z=skip+nl;replayed=(np.sinh(z) if model=='chronos2' else z)*scale[:,None]+loc[:,None];forecast_rms=float(np.sqrt(np.mean(((native-replayed)/latent[:,None])**2)));base_native_scale=scale.reshape(128,5)[:,0]
        result=dict(capture_controls=controls,capture_passed=all(v==0 for v in controls.values()),forecast_normalized_RMS=forecast_rms,deltas={});all_replay=True
        for delta in [.01,.02]:
            actual=derivative(native,conditions,delta);rep=derivative(replayed,conditions,delta);total=off_coefficients(actual,base_scale);rep_off=off_coefficients(rep,base_scale);gate=replay_gate(forecast_rms,total,rep_off,strength);all_replay=all_replay and gate['passed']
            if model=='chronos2':
                plus=next(i for i,c in enumerate(conditions) if c[1:]==['psi',1,delta]);minus=next(i for i,c in enumerate(conditions) if c[1:]==['psi',-1,delta]);cells=z.reshape(128,5,32);factor=common_secant(cells[:,plus],cells[:,minus])
            else:factor=1.
            skip_d=derivative(skip,conditions,delta)*base_native_scale[:,None]*factor;nl_d=derivative(nl,conditions,delta)*base_native_scale[:,None]*factor;parts={name:off_coefficients(value,base_scale) for name,value in dict(skip=skip_d,nonlinear=nl_d,normalization_residual=rep-skip_d-nl_d,numerical_replay_residual=actual-rep).items()}
            result['deltas'][str(delta)]=dict(replay=gate,absolute_attribution=absolute_attribution(total,parts,strength) if result['capture_passed'] and gate['passed'] else None)
            saved[model+'_'+str(delta)+'_native_off_derivative']=total
            for name,value in parts.items():saved[model+'_'+str(delta)+'_'+name]=value
            if delta==.01:energies[model]=np.sum(total**2,axis=1)
        result['all_gates_passed']=bool(result['capture_passed'] and all_replay);native_loss=np.mean((native.reshape(128,5,32)[:,0]-future.reshape(128,5,32)[:,0])**2,axis=1)/base_scale**2;result['base_native_scaled_mse']=bootstrap(native_loss);report['models'][model]=result;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({model:dict(capture=controls,replay_passed=result['all_gates_passed'],native_off_energy_ratio=bootstrap(energies[model],strength))}),flush=True)
    for model,minimum in [('timesfm',.10),('bolt',.02)]:
        effect=paired_numerator_effect(energies[model],energies['chronos2'],strength);effect['minimum_point_effect']=minimum;report['comparisons'][model+'_minus_chronos2']=effect
    finalize_comparisons(report['comparisons'],{model:result['all_gates_passed'] for model,result in report['models'].items()})
    report['all_primary_passed']=all(v['passed'] for v in report['comparisons'].values());report['inference_caveat']='Exact sign tests establish a conditional majority among non-ties;approximate ratio intervals do not establish assumption-free mean superiority. All128seeds remain in energy ratios.'
    np.savez_compressed(out/'source_data.npz',analytic_strength=strength,**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['comparisons'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bolt-weights',type=Path,required=True);p.add_argument('--timesfm-weights',type=Path,required=True);p.add_argument('--chronos2-weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();torch.set_num_threads(1 if a.prepare_only else 2)
    if not a.prepare_only and torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run({'bolt':a.bolt_weights,'timesfm':a.timesfm_weights,'chronos2':a.chronos2_weights},a.out,a.prepare_only)
