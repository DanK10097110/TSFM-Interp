"""Frozen target/nuisance selective donor edits for a two-frequency factorial.

Four primary claims average both directions and other-phase cells per seed.
Specialized use requires practical selectivity plus superiority to both random
and top-PC controls. Direct model interactions are a separate two-claim family.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from mixture_features import PROTOCOL,pairs,metrics,harmonic_fit
from mixture_directions import fit_direction_frames
from activation_interchange import response_bases,projected_replacement,patched_predict
from tsfm_lens.config import ModelConfig,DataConfig
from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
from tsfm_lens.experiments.feature_search.statistics import compare_effects,paired_effect


def by_seed(values,groups):
    """Average direction/cell observations within a generating bundle."""
    return np.array([np.mean(np.asarray(values)[groups==g]) for g in np.unique(groups)])


def nuisance_pass(summary):
    """Enforce the fixed unchanged-phase/amplitude/offset limits."""
    return all(summary[key]<=.1 for key in ['other_phase_change','a_relative_change','b_relative_change','offset_change_normalized'])


def run(source,out):
    """Fit all training directions, freeze validation choices, then test once."""
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(PROTOCOL,version=1,methods=['encoding2','haufe2','decoder2','contrast2','contrast4'],alpha=1.,random_seed=87,
        validation_score='target circular error+otherphasechange+arelchange+brelchange+offsetchange',controls='equal per-token edit-energy covariance16 and topPC;whole,self,zero',
        secondary='2targetphases paired crossmodel selectivitygain interactions Holm2'),indent=2),encoding='utf-8')
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'));extraction=json.loads((source/'extraction.json').read_text(encoding='utf-8'))
    context=np.load(source/'corpus.npz')['context'];tr=np.array([m['split']=='train' for m in metadata]);claims={};gains={};report={}
    for name,cls,checkpoint in [('bolt',ChronosBoltAdapter,'amazon/chronos-bolt-small'),('timesfm',TimesFMAdapter,'google/timesfm-2.5-200m-pytorch')]:
        adapter=cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded()
        data=np.load(source/f'{name}_activations.npz');layers=extraction['models'][name]['layers'];starts=adapter.token_time_spans()[:,0]
        frames={};controls={};diagnostics={};candidates={target:[] for target in ['phi','psi']}
        for layer in layers:
            h=data[layer];_,top,random=response_bases(h[tr],h[tr],max_rank=4,covariance_samples=h[tr],seed=87);controls[layer]=(top,random)
            for target in ['phi','psi']:
                fitted=fit_direction_frames(h,metadata,starts,target=target,alpha=1.,train_mask=tr)
                frames[layer,target]=fitted['frames'];diagnostics[layer+'_'+target]=fitted['diagnostics']
        baseline=harmonic_fit(data['prediction']);test=np.array([m['split']=='test' for m in metadata]);native={}
        for phase,amp in [('phi','a'),('psi','b')]:
            truth=np.array([m[phase] for m in metadata]);trueamp=np.array([m[amp] for m in metadata]);native[phase]=dict(phase_error=float(abs(np.angle(np.exp(1j*(baseline[phase][test]-truth[test])))).mean()),amplitude_relative_error=float(np.mean(abs(baseline[amp][test]/trueamp[test]-1))))
        gates=[]
        for target in ['phi','psi']:
            for layer in layers:
                h=data[layer];values={method:[] for method in frames[layer,target]}
                for reverse in [False,True]:
                    receiver,donor,groups=pairs(metadata,'validation',target,reverse);rows=[metadata[i] for i in receiver]
                    base=patched_predict(adapter,context[receiver]);donorpred=patched_predict(adapter,context[donor]);selfpred=patched_predict(adapter,context[receiver],layer,h[receiver]);full=patched_predict(adapter,context[receiver],layer,h[donor])
                    base_error=metrics(base,base,donorpred,rows,target)['target_error'];full_error=metrics(full,base,donorpred,rows,target)['target_error']
                    reached=bool(abs(base-selfpred).max()==0 and base_error.mean()>1e-4 and full_error.mean()<=.5*base_error.mean())
                    gates.append(dict(target=target,layer=layer,reverse=reverse,passed=reached,self_max_abs=float(abs(base-selfpred).max()),base_target_error=float(base_error.mean()),full_target_error=float(full_error.mean())))
                    if not reached:break
                    for method,basis in frames[layer,target].items():
                        if basis.shape[1]==0:continue
                        replace,_=projected_replacement(h[receiver],h[donor],basis);prediction=patched_predict(adapter,context[receiver],layer,replace);score=metrics(prediction,base,donorpred,rows,target)
                        values[method].append({k:float(v.mean()) for k,v in score.items()})
                for method,records in values.items():
                    if len(records)!=2:continue
                    summary={k:float(np.mean([v[k] for v in records])) for k in records[0]};score=sum(summary[k] for k in ['target_error','other_phase_change','a_relative_change','b_relative_change','offset_change_normalized'])
                    candidates[target].append(dict(layer=layer,method=method,rank=frames[layer,target][method].shape[1],score=score,nuisance_eligible=nuisance_pass(summary),validation_metrics=summary))
        selected={}
        for target in ['phi','psi']:
            eligible=[c for c in candidates[target] if c['nuisance_eligible']]
            selected[target]=min(eligible,key=lambda c:c['score']) if eligible else None
        frozen=dict(selected=selected,candidates=candidates,gates=gates,native_forecast=native,direction_diagnostics=diagnostics)
        frozen_frames={};fit_hashes={}
        for target,choice in selected.items():
            if choice is None:continue
            layer,method=choice['layer'],choice['method'];top,random=controls[layer];rank=frames[layer,target][method].shape[1]
            frozen_frames[target+'_learned']=frames[layer,target][method]
            frozen_frames[target+'_top_PC']=top[:,:rank]
            frozen_frames[target+'_random']=np.stack([frame[:,:rank] for frame in random])
            frozen_frames[target+'_train_center']=data[layer][tr].mean(axis=(0,1))
            fit_hashes[target]=dict(training_activation_sha256=hashlib.sha256(np.ascontiguousarray(data[layer][tr]).tobytes()).hexdigest(),
                training_metadata_sha256=hashlib.sha256(json.dumps([m for m,t in zip(metadata,tr) if t],sort_keys=True).encode('utf-8')).hexdigest(),
                source_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),checkpoint=checkpoint,layer=layer,method=method,rank=rank)
        frozen['fit_hashes']=fit_hashes
        np.savez_compressed(out/f'{name}_frozen_frames.npz',**frozen_frames)
        (out/f'{name}_frozen_selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');print(json.dumps(dict(model=name,frozen_selection=selected)),flush=True)
        model=dict(frozen,directions={});saved={}
        for target in ['phi','psi']:
            chosen=selected[target]
            if chosen is None:model['directions'][target]=dict(skipped='no validation candidate passed nuisance and reach gates');continue
            layer,method=chosen['layer'],chosen['method'];h=data[layer];basis=frames[layer,target][method];rank=basis.shape[1];top,random=controls[layer]
            accumulator={};allgroups=[];perdirection={}
            for reverse in [False,True]:
                direction='reverse' if reverse else 'forward';receiver,donor,groups=pairs(metadata,'test',target,reverse);rows=[metadata[i] for i in receiver]
                preds=dict(base=patched_predict(adapter,context[receiver]),donor=patched_predict(adapter,context[donor]),self=patched_predict(adapter,context[receiver],layer,h[receiver]),zero=patched_predict(adapter,context[receiver],layer,np.zeros_like(h[receiver])),full=patched_predict(adapter,context[receiver],layer,h[donor]))
                if abs(preds['self']-preds['base']).max()!=0:raise RuntimeError('heldout self patch failed')
                replace,energy=projected_replacement(h[receiver],h[donor],basis);preds['learned']=patched_predict(adapter,context[receiver],layer,replace)
                for key,frame in [('top_PC',top[:,:rank])]+[(f'random{i}',frame[:,:rank]) for i,frame in enumerate(random)]:
                    replacement,_=projected_replacement(h[receiver],h[donor],frame,target_energy=energy);preds[key]=patched_predict(adapter,context[receiver],layer,replacement)
                measured={key:metrics(pred,preds['base'],preds['donor'],rows,target) for key,pred in preds.items()}
                targettruth=np.array([metadata[i][target] for i in donor]);donortruth=abs(np.angle(np.exp(1j*(harmonic_fit(preds['donor'])[target]-targettruth))))
                perdirection[direction]=dict(donor_truephase_error=float(donortruth.mean()),metrics={key:{k:float(v.mean()) for k,v in value.items()} for key,value in measured.items()})
                for key,values in measured.items():
                    for metric,value in values.items():accumulator.setdefault((key,metric),[]).append(value)
                allgroups.append(groups)
                for key,pred in preds.items():saved[target+'_'+direction+'_'+key]=pred
                saved[target+'_'+direction+'_seeds']=groups
            groups=np.concatenate(allgroups);aggregated={key:by_seed(np.concatenate(values),groups) for key,values in accumulator.items()};seeds=np.unique(groups)
            random_error=np.mean([aggregated[f'random{i}','target_error'] for i in range(16)],axis=0);learned=aggregated['learned','target_error'];top_error=aggregated['top_PC','target_error'];claims[name+'_'+target]=(random_error,learned);gains[name,target]=random_error-learned
            summary={key:float(aggregated['learned',key].mean()) for key in measured['learned']};restoration=1-float(learned.mean()/aggregated['base','target_error'].mean());random_adv=float((random_error-learned).mean());top_adv=float((top_error-learned).mean())
            practical=bool(restoration>=.5 and nuisance_pass(summary) and random_adv>=.1 and top_adv>=.1)
            model['directions'][target]=dict(directions=perdirection,aggregate_metrics=summary,restoration_fraction=restoration,random_advantage=random_adv,top_PC_advantage=top_adv,practical_gate=practical,
                top_PC_comparison=paired_effect(top_error,learned,seeds,n_boot=2000,n_permutations=1999,seed=87))
        report[name]=model;np.savez_compressed(out/f'{name}_predictions.npz',**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        del adapter,data,frames,controls;gc.collect();torch.cuda.empty_cache()
    seeds=np.arange(144,192)
    for name in ['bolt','timesfm']:
        for target in ['phi','psi']:
            claims.setdefault(name+'_'+target,(np.zeros(48),np.zeros(48)))
    report['primary']=compare_effects(claims,seeds,n_boot=2000,n_permutations=1999,seed=87)
    for name in ['bolt','timesfm']:
        for target in ['phi','psi']:
            result=report[name]['directions'][target]
            if 'skipped' in result:continue
            effect=report['primary'][name+'_'+target]
            result['specialized_gate_passed']=bool(result['practical_gate'] and effect['p_holm']<.05 and effect['lo']>0 and result['top_PC_comparison']['lo']>0)
    interactions={target:(gains['timesfm',target],gains['bolt',target]) for target in ['phi','psi'] if ('timesfm',target) in gains and ('bolt',target) in gains}
    if interactions:report['crossmodel_interaction']=compare_effects(interactions,seeds,n_boot=2000,n_permutations=1999,seed=87)
    report['claim_rule']='shared only if both positive;unique never from significance difference;model interaction direct paired test;native donor truth errors retained;specialized requires practical and randomHolm/topPCCI gates'
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k in ['primary','crossmodel_interaction']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(args.source,args.out)
