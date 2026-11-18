"""Local natural phase-tangent causal-use audit at native forecast-head inputs.

This instrument follows a task-defined feature field near each input. It does
not search global disentangled directions or label neurons. Frozen supervised
readouts diagnose retained spectral derivatives; internal tangent injections
test their local causal route through the native head, without weight edits.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import inspect
from pathlib import Path
import numpy as np

from mixture_features import harmonic_fit

CONDITIONS=[('base',None,0,0.)]+[(f'{target}_{sign}_{delta}',target,sign,delta) for delta in [.01,.02] for target in ['phi','psi'] for sign in [1,-1]]
PROTOCOL=dict(seed_base=1437000,n_seeds=64,context=128,horizon=32,periods=[32,16],noise=0,slope=0,
    conditions=CONDITIONS,primary_delta=.01,convergence_delta=.02,
    frozen_probe='feature_search_20261111_native_head_readout_fp64;no refit',
    cutpoints={'bolt':'output_patch_embedding full[B,1,512] input','timesfm':'output_projection_point full[B,4,1280] input'},
    intervention='base head input +/-delta*centered natural phase derivative;existing input_slice_ablate whole feature slice,all positions',
    gates={'hook_identity_max_abs':0,'self_patch_max_abs':0,'delta_relative_RMS':.05,'internal_to_actual_relative_RMS':.05,'probe_target_relative_RMS':.10,'probe_offtarget_relative_RMS':.10},
    metric='sum off-target normalized coefficient derivative energy / sum analytic own-target derivative strength²',
    inference='paired seed bootstrap numerator+denominator jointly;signflip numerator;Holm4 native-minus-probe;separate Holm2 crossmodel',
    missingness='keep all64seeds;gatefailures remain with reason,never exclude hard seeds',
    evidence='local task-tangent causal route;supervised retained-feature diagnostic;not global uniqueness or neuron labeling')


def condition_map(targets=('phi','psi')):
    """Freeze perturbations in declared order;never add an unrequested target."""
    targets=tuple(targets)
    if not targets or len(set(targets))!=len(targets) or any(t not in ['phi','psi'] for t in targets):raise ValueError('Need unique declared phase targets')
    return [('base',None,0,0.)]+[(f'{target}_{sign}_{delta}',target,sign,delta) for delta in [.01,.02] for target in targets for sign in [1,-1]]


def make_corpus(seed_base=1437000,n_seeds=64,targets=('phi','psi')):
    """Create base and eight small perturbations with mathematically fixed normalizers."""
    contexts=[];future=[];metadata=[];t=np.arange(160)
    for seed in range(n_seeds):
        rng=np.random.default_rng(seed_base+seed);a,b=rng.uniform(.5,2,2);offset=rng.uniform(-2,2);phi,psi=rng.uniform(-np.pi,np.pi,2);scale=np.sqrt((a*a+b*b)/2)
        for condition,target,sign,delta in condition_map(targets):
            p=phi+(sign*delta if target=='phi' else 0);q=psi+(sign*delta if target=='psi' else 0)
            x=offset+a*np.sin(2*np.pi*t/32+p)+b*np.sin(2*np.pi*t/16+q);contexts.append(x[:128]);future.append(x[128:]);metadata.append(dict(seed=seed,condition=condition,phi=p,psi=q,a=a,b=b,offset=offset,scale=scale))
    return np.asarray(contexts,dtype=np.float32),np.asarray(future),metadata


def analytic_derivative(metadata,target):
    """Differentiate [a sinφ/s,a cosφ/s,b sinψ/s,b cosψ/s] in that exact order."""
    result=np.zeros((len(metadata),4))
    for i,m in enumerate(metadata):
        if target=='phi':result[i,:2]=m['a']/m['scale']*np.array([np.cos(m['phi']),-np.sin(m['phi'])])
        elif target=='psi':result[i,2:]=m['b']/m['scale']*np.array([np.cos(m['psi']),-np.sin(m['psi'])])
        else:raise ValueError('Unknown phase target')
    return result


def centered_derivative(values,target,delta,conditions=None):
    """Read the frozen condition map instead of guessing flattened row offsets."""
    conditions=CONDITIONS if conditions is None else conditions
    plus=next(i for i,c in enumerate(conditions) if c[1:]==(target,1,delta));minus=next(i for i,c in enumerate(conditions) if c[1:]==(target,-1,delta))
    if len(values)%len(conditions):raise ValueError('Incomplete perturbation bundle')
    array=np.asarray(values,dtype=float).reshape((len(values)//len(conditions),len(conditions))+np.asarray(values).shape[1:])
    return (array[:,plus]-array[:,minus])/(2*delta)


def coefficient_derivative(future_derivative,scales):
    """Extract global normalized phase-order spectral derivatives by joint LS."""
    fit=harmonic_fit(future_derivative)['coefficients'];return fit[:,[1,0,3,2]]/np.asarray(scales)[:,None]


def probe_derivative(head_derivative,weights):
    """Differentiate the persisted old-trained linear probe;intercept cancels."""
    value=np.asarray(head_derivative,dtype=float)
    if value.ndim!=3:raise ValueError('Expected full[B,T,D] head derivative')
    return (value[:,-1]/weights['scaler_scale'])@weights['coefficients'].T


def relative_rms(actual,reference):
    """Use aggregate energy, exposing an undefined dead reference explicitly."""
    actual,reference=np.asarray(actual,dtype=float),np.asarray(reference,dtype=float);denominator=np.sum(reference**2)
    return None if denominator==0 else float(np.sqrt(np.sum((actual-reference)**2)/denominator))


def energy_ratio_effect(native,probe,strength,seed=1437):
    """Joint seed-bootstrap a ratio of sums;sign-flip its paired numerator only."""
    a,b,d=[np.asarray(v,dtype=float) for v in [native,probe,strength]]
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=d.shape or len(a)<3 or not np.isfinite(np.column_stack([a,b,d])).all() or np.any(d<0) or d.sum()==0:raise ValueError('Need finite equal seed vectors and positive aggregate strength')
    difference=a-b;rng=np.random.default_rng(seed);indices=rng.integers(0,len(a),(2000,len(a)));ratios=difference[indices].sum(1)/d[indices].sum(1)
    observed=abs(difference.sum());exceed=sum(abs(np.sum(difference*rng.choice([-1,1],len(a))))>=observed for _ in range(1999))
    return dict(value=float(difference.sum()/d.sum()),lo=float(np.quantile(ratios,.025)),hi=float(np.quantile(ratios,.975)),p=(1+exceed)/2000,n_seeds=len(a),metric='difference of aggregate energy ratios;not mean per-seed ratios')


def apply_full_head_patch(root,module_name,replacement):
    """Use the existing canonical input-slice patcher for an exact full[B,T,D] cache."""
    from tsfm_lens.extraction.hooks import input_slice_ablate
    if replacement.ndim!=3:raise ValueError('Full head cache must retain batch,position,feature axes')
    return input_slice_ablate(root,module_name,slice(None),replacement)


def complete_family(primary,interactions,replication=False):
    """Adjust the entire prospective family,including the crossmodel replication claim."""
    from tsfm_lens.analysis.stats import holm
    if replication:
        if set(primary)!={'bolt_psi','timesfm_psi'} or set(interactions)!={'psi'}:raise ValueError('Replication needs exactly its complete three-claim family')
        family={**primary,'timesfm_minus_bolt_psi':interactions['psi']};adjusted=holm({k:v['p'] for k,v in family.items()})
        for k,v in family.items():v['p_holm']=adjusted[k]
        return family
    for family in [primary,interactions]:
        adjusted=holm({k:v['p'] for k,v in family.items()})
        for k,v in family.items():v['p_holm']=adjusted[k]
    return None


def experiment_protocol(probe_source=None,seed_base=1437000,n_seeds=64,targets=('phi','psi'),replication=False):
    """Describe the exact prospective run before any model inference."""
    if replication and (seed_base!=1659000 or n_seeds!=128 or tuple(targets)!=('psi',)):raise ValueError('Frozen replication requires seed1659000,n128,psi only')
    protocol=dict(PROTOCOL,seed_base=seed_base,n_seeds=n_seeds,conditions=condition_map(targets),targets=list(targets),replication=replication)
    if probe_source is not None:protocol['frozen_probe_sha256']={name:hashlib.sha256((probe_source/f'{name}_frozen_probe.npz').read_bytes()).hexdigest() for name in ['bolt','timesfm']}
    if replication:protocol.update(inference='single Holm3:2model native-minus-probe +TFminusBolt;pairedjointseedbootstrap/signflip numerator',practical_model_gain=.03,practical_crossmodel_gain=.10,missingness='retain all128seeds;no gate rescue')
    return protocol


def run(out,probe_source,seed_base=1437000,n_seeds=64,targets=('phi','psi'),replication=False):
    """Execute the reviewed local-tangent protocol;retain all seeds and failed gates."""
    import torch
    import gc
    from head_readout import head_tensor
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    conditions=condition_map(targets);n_conditions=len(conditions);protocol=experiment_protocol(probe_source,seed_base,n_seeds,targets,replication)
    out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    context,future,metadata=make_corpus(seed_base,n_seeds,targets);base_meta=metadata[::n_conditions];base=context.reshape(n_seeds,n_conditions,128)[:,0];scales=np.array([m['scale'] for m in base_meta]);truth_derivatives={target:analytic_derivative(base_meta,target) for target in targets}
    np.savez_compressed(out/'corpus.npz',context=context,clean_future=future);(out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report=dict(models={},normalizers=dict(mean_within_bundle_max=float(np.ptp(context.reshape(n_seeds,n_conditions,128).mean(-1),axis=1).max()),std_within_bundle_max=float(np.ptp(context.reshape(n_seeds,n_conditions,128).std(-1),axis=1).max())));numerators={};denominators={};primary={};saved={}
    for name,cls,checkpoint,module_name,positions,width in [('bolt',ChronosBoltAdapter,'amazon/chronos-bolt-small','output_patch_embedding',1,512),('timesfm',TimesFMAdapter,'google/timesfm-2.5-200m-pytorch','output_projection_point',4,1280)]:
        adapter=cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded();module=dict(adapter.module.named_modules())[module_name]
        source_path=Path(inspect.getsourcefile(type(adapter.module)));weights_path=probe_source/f'{name}_frozen_probe.npz';weights=np.load(weights_path);fullheads=[];predictions=[];hook_error=0.
        for start in range(0,len(context),8):
            batch=context[start:start+8];hits=[]
            with torch.inference_mode():
                before=adapter.predict(batch,32,[.5])['point']
                def capture(m,inputs):
                    head_tensor(inputs,batch=len(batch),positions=positions,width=width);hits.append(inputs[0].detach().float().cpu().numpy().copy())
                handle=module.register_forward_pre_hook(capture)
                try:prediction=adapter.predict(batch,32,[.5])['point']
                finally:handle.remove()
            if len(hits)!=1:raise RuntimeError(f'Wrong actual head firing count:{name}:{len(hits)}')
            hook_error=max(hook_error,float(abs(prediction-before).max()));fullheads.append(hits[0]);predictions.append(prediction)
        h=np.concatenate(fullheads);native=np.concatenate(predictions);baseh=h.reshape(n_seeds,n_conditions,positions,width)[:,0]
        def patched(replacement=None):
            outputs=[]
            for start in range(0,n_seeds,8):
                with torch.inference_mode():
                    if replacement is None:value=adapter.predict(base[start:start+8],32,[.5])['point']
                    else:
                        cache=torch.tensor(replacement[start:start+8],device=adapter.device,dtype=torch.float32)
                        with apply_full_head_patch(adapter.module,module_name,cache):value=adapter.predict(base[start:start+8],32,[.5])['point']
                outputs.append(value)
            return np.concatenate(outputs)
        original=patched();self_prediction=patched(baseh);self_error=float(abs(original-self_prediction).max())
        model=dict(head_module=module_name,shape=list(h.shape),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),frozen_probe_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),hook_identity_max_abs=hook_error,self_patch_max_abs=self_error,targets={})
        for target in targets:
            own=slice(0,2) if target=='phi' else slice(2,4);other=slice(2,4) if target=='phi' else slice(0,2);ideal=truth_derivatives[target];strength=np.sum(ideal[:,own]**2,axis=1);denominators[target]=strength;results={};derivatives={}
            for delta in [.01,.02]:
                dh=centered_derivative(h,target,delta,conditions);actual=centered_derivative(native,target,delta,conditions);native_coefficient=coefficient_derivative(actual,scales);probe_coefficient=probe_derivative(dh,weights)
                plus=patched(baseh+delta*dh);minus=patched(baseh-delta*dh);internal=(plus-minus)/(2*delta)
                relative=relative_rms(internal,actual);own_error=relative_rms(probe_coefficient[:,own],ideal[:,own]);off_error=float(np.sqrt(np.sum(probe_coefficient[:,other]**2)/strength.sum()))
                native_energy=np.sum(native_coefficient[:,other]**2,axis=1);probe_energy=np.sum(probe_coefficient[:,other]**2,axis=1)
                results[str(delta)]=dict(internal_to_actual_relative_RMS=relative,probe_own_relative_RMS=own_error,probe_offtarget_relative_RMS=off_error,native_offtarget_energy_ratio=float(native_energy.sum()/strength.sum()),probe_offtarget_energy_ratio=float(probe_energy.sum()/strength.sum()),native_own_relative_RMS=relative_rms(native_coefficient[:,own],ideal[:,own]))
                derivatives[delta]=(dh,actual,native_coefficient,probe_coefficient)
                for label,value in [('head_derivative',dh),('actual_forecast_derivative',actual),('internal_forecast_derivative',internal),('native_coefficient_derivative',native_coefficient),('probe_coefficient_derivative',probe_coefficient)]:saved[name+'_'+target+'_'+str(delta)+'_'+label]=value
                if delta==.01:
                    primary[name+'_'+target]=energy_ratio_effect(native_energy,probe_energy,strength);numerators[name,target]=native_energy-probe_energy
            finite=dict(head_relative_RMS=relative_rms(derivatives[.02][0],derivatives[.01][0]),native_output_relative_RMS=relative_rms(derivatives[.02][1],derivatives[.01][1]),probe_coefficient_relative_RMS=relative_rms(derivatives[.02][3],derivatives[.01][3]))
            reasons=[]
            if hook_error!=0 or self_error!=0:reasons.append('hook/self identity failed')
            if any(v is None or v>.05 for v in finite.values()):reasons.append('finite-difference convergence gate failed')
            if any(v['internal_to_actual_relative_RMS'] is None or v['internal_to_actual_relative_RMS']>.05 for v in results.values()):reasons.append('local internal causal-route agreement failed')
            if results['0.01']['probe_own_relative_RMS']>.10 or results['0.01']['probe_offtarget_relative_RMS']>.10:reasons.append('frozen-probe local retained-feature gate failed')
            model['targets'][target]=dict(deltas=results,finite_difference=finite,interpretation_allowed=not reasons,gate_failure_reasons=reasons,n_seeds=n_seeds)
        report['models'][name]=model;np.savez_compressed(out/f'{name}_capture.npz',head_inputs=h,prediction=native);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({name:model}),flush=True);del adapter,module,h,fullheads;gc.collect();torch.cuda.empty_cache()
    report['primary_native_minus_probe']=primary;interactions={target:energy_ratio_effect(numerators['timesfm',target],numerators['bolt',target],denominators[target]) for target in targets};family=complete_family(primary,interactions,replication)
    if replication:
        report['replication_family']=family;passes={name:bool(report['models'][name]['targets']['psi']['interpretation_allowed'] and primary[name+'_psi']['value']>=.03 and primary[name+'_psi']['lo']>0 and primary[name+'_psi']['p_holm']<.05) for name in ['bolt','timesfm']}
        passes['crossmodel']=bool(all(passes.values()) and interactions['psi']['value']>=.10 and interactions['psi']['lo']>0 and interactions['psi']['p_holm']<.05);report['replication_passes']=passes
    report['crossmodel_difference']=interactions;report['interpretation']=f'All{n_seeds}seeds retained;gatefailed metrics descriptive unknown for retained-feature/causal-route claims;no global feature uniqueness.'
    np.savez_compressed(out/'derivatives.npz',**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(primary=primary,crossmodel=interactions),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path);p.add_argument('--out',type=Path);p.add_argument('--probe-source',type=Path);p.add_argument('--seed-base',type=int,default=1437000);p.add_argument('--n-seeds',type=int,default=64);p.add_argument('--targets',nargs='+',default=['phi','psi']);p.add_argument('--replication',action='store_true');args=p.parse_args()
    if args.protocol:args.protocol.write_text(json.dumps(experiment_protocol(args.probe_source,args.seed_base,args.n_seeds,args.targets,args.replication),indent=2),encoding='utf-8')
    elif args.out and args.probe_source:
        import torch
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
        run(args.out,args.probe_source,args.seed_base,args.n_seeds,args.targets,args.replication)
    else:p.error('Need --protocol or both --out and --probe-source')
