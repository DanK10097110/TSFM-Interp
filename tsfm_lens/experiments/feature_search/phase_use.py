"""Conditional phase representation versus causal-use POC on fresh paired seeds.

Validation chooses one layer/subspace per model; heldout results compare
natural projected donor edits to equal-energy controls. A failure does not
establish a unique representation or a model defect. Context-only canonical
patches preserve REG; full raw states are retained for a later readout study.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch

from phase_features import PROTOCOL, make_phase_corpus, harmonic_fit, fit_phase_probe, phase_metrics, normalization_diagnostics
from activation_interchange import response_bases, projected_replacement, patched_predict, paired_rows
from tsfm_lens.config import DataConfig,ModelConfig
from tsfm_lens.extraction.hooks import _primary
from tsfm_lens.extraction.alignment import run_alignment_gate
from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
from tsfm_lens.experiments.feature_search.statistics import paired_effect,compare_effects


def circular_error(a,b):
    """Return absolute wrapped angular differences in radians."""
    return abs(np.angle(np.exp(1j*(a-b))))


def run(out):
    """Extract fresh phase pairs, freeze validation choices and test once."""
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(PROTOCOL,selection_score='mean bidirectional donor circular error+relative amplitude change+normalized offset change',
                  validation_nuisance_gate=.10,phase_restoration_fraction_gate=.50,
                  test_control_advantage_radians=.10,test_nuisance_gate=.10,
                  inference='Holm4 random comparison p<.05;bootstrap lowerCI>0 random and topPC;specialized requires both',
                  evidence='phase mechanism POC;period32 validation case,not an aliasing novelty claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    context,future,metadata=make_phase_corpus()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    np.savez_compressed(out/'corpus.npz',context=context,clean_future=future)
    phases=np.array([m['phase'] for m in metadata]);amplitudes=np.array([m['amplitude'] for m in metadata]);splits=np.array([m['split'] for m in metadata])
    tr,va,te=[splits==s for s in ['train','validation','test']]
    report=dict(normalization=normalization_diagnostics(context),models={});claims={};gate_rows={}
    for name,cls,checkpoint in [('bolt',ChronosBoltAdapter,'amazon/chronos-bolt-small'),('timesfm',TimesFMAdapter,'google/timesfm-2.5-200m-pytorch')]:
        adapter=cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded()
        all_layers=adapter.all_layer_names();layers=[all_layers[i] for i in [0,len(all_layers)//2,len(all_layers)-1]]
        alignment=run_alignment_gate(adapter,32,layers)
        collected={k:[] for k in layers};raw={k:[] for k in layers};modules=dict(adapter.module.named_modules());handles=[]
        for layer in layers:
            def capture(m,i,o,key=layer):
                hidden=_primary(o);raw[key].append(hidden.detach().float().cpu().numpy())
                collected[key].append(adapter.postprocess_tokens(key,hidden).detach().float().cpu().numpy())
            handles.append(modules[layer].register_forward_hook(capture))
        prediction=patched_predict(adapter,context)
        for h in handles:h.remove()
        tokens={k:np.concatenate(v) for k,v in collected.items()};fullraw={'raw__'+k:np.concatenate(v) for k,v in raw.items()}
        np.savez_compressed(out/f'{name}_activations.npz',prediction=prediction,**tokens,**fullraw)
        del collected,raw,fullraw
        repeat=patched_predict(adapter,context[:8]);repeat_error=float(abs(repeat-prediction[:8]).max())
        if repeat_error!=0:raise RuntimeError(f'deterministic same-batch gate failed {name}:{repeat_error}')
        train_receiver,train_donor,_=paired_rows(metadata,'train')
        probes={};frames={};candidates=[];validation_gates=[]
        for layer in layers:
            h=tokens[layer]
            fit=fit_phase_probe(h[:,-1].astype(float),phases,tr,va)
            probe_results={}
            for method,options in [('ridge',{}),('neuron',{'coordinate':True}),('rotated_neuron',{'coordinate':True,'rotation_seed':73})]:
                result=fit if method=='ridge' else fit_phase_probe(h[:,-1].astype(float),phases,tr,va,**options)
                angle=np.arctan2(result['prediction'][:,0],result['prediction'][:,1])
                probe_results[method]=dict(validation_circular_error=float(circular_error(angle[va],phases[va]).mean()),test_circular_error=float(circular_error(angle[te],phases[te]).mean()),alpha=result['alpha'])
            probes[layer]=probe_results
            response,top,random=response_bases(h[train_receiver],h[train_donor],max_rank=4,covariance_samples=h[tr],seed=73)
            layer_frames={'phase_probe2':fit['frame'],'response2':response[:,:2],'response4':response[:,:4]}
            frames[layer]=(layer_frames,top,random)
            scores={key:[] for key in layer_frames};passed=True
            for reverse in [False,True]:
                receiver,donor,_=paired_rows(metadata,'validation',reverse);base=patched_predict(adapter,context[receiver]);selfpred=patched_predict(adapter,context[receiver],layer,h[receiver]);full=patched_predict(adapter,context[receiver],layer,h[donor]);donorpred=patched_predict(adapter,context[donor])
                before=circular_error(harmonic_fit(base)['phase'],harmonic_fit(donorpred)['phase']).mean();after=circular_error(harmonic_fit(full)['phase'],harmonic_fit(donorpred)['phase']).mean()
                valid_gate=bool(abs(base-selfpred).max()==0 and before>1e-4 and after<=.5*before)
                validation_gates.append(dict(layer=layer,reverse=reverse,self_max_abs=float(abs(base-selfpred).max()),base_donor_error=float(before),full_donor_error=float(after),passed=valid_gate))
                if not valid_gate:passed=False;break
                for candidate,basis in layer_frames.items():
                    replace,_=projected_replacement(h[receiver],h[donor],basis);edited=patched_predict(adapter,context[receiver],layer,replace)
                    metrics=phase_metrics(edited,base,donorpred,amplitudes[receiver]);amp=metrics['amplitude_relative_change'].mean();offset=metrics['offset_change_normalized'].mean()
                    scores[candidate].append(dict(score=float(metrics['donor_phase_error'].mean()+amp+offset),amp=float(amp),offset=float(offset)))
            if passed:
                for candidate,values in scores.items():
                    eligible=all(v['amp']<=.10 and v['offset']<=.10 for v in values)
                    candidates.append(dict(layer=layer,basis=candidate,validation_score=float(np.mean([v['score'] for v in values])),nuisance_eligible=eligible,details=values))
        probe_layer=min(layers,key=lambda k:probes[k]['ridge']['validation_circular_error'])
        model=dict(alignment=alignment,repeat_max_abs=repeat_error,probes=probes,selected_probe_layer=probe_layer,selected_probe_test_error=probes[probe_layer]['ridge']['test_circular_error'],validation_gates=validation_gates,candidates=candidates)
        eligible=[c for c in candidates if c['nuisance_eligible']]
        if not eligible:
            model['skipped']='no candidate cleared validation phase-reach and nuisance gates';report['models'][name]=model
            (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');del adapter,tokens,frames,modules;gc.collect();torch.cuda.empty_cache();continue
        chosen=min(eligible,key=lambda c:c['validation_score']);layer=chosen['layer'];basis=frames[layer][0][chosen['basis']];rank=basis.shape[1];h=tokens[layer];model['selection']=chosen;model['directions']={}
        print(json.dumps(dict(model=name,frozen_selection=chosen)),flush=True)
        saved={}
        for reverse in [False,True]:
            direction='reverse' if reverse else 'forward';receiver,donor,seeds=paired_rows(metadata,'test',reverse)
            preds=dict(base=patched_predict(adapter,context[receiver]),donor=patched_predict(adapter,context[donor]),self=patched_predict(adapter,context[receiver],layer,h[receiver]),zero=patched_predict(adapter,context[receiver],layer,np.zeros_like(h[receiver])),full=patched_predict(adapter,context[receiver],layer,h[donor]))
            if abs(preds['self']-preds['base']).max()!=0:raise RuntimeError('heldout self gate failed')
            replace,energy=projected_replacement(h[receiver],h[donor],basis);preds['learned']=patched_predict(adapter,context[receiver],layer,replace)
            top,random=frames[layer][1:]
            for key,frame in [('top_PC',top[:,:rank])]+[(f'random{i}',frame[:,:rank]) for i,frame in enumerate(random)]:
                replace,_=projected_replacement(h[receiver],h[donor],frame,target_energy=energy);preds[key]=patched_predict(adapter,context[receiver],layer,replace)
            metrics={key:phase_metrics(value,preds['base'],preds['donor'],amplitudes[receiver]) for key,value in preds.items()}
            error={key:value['donor_phase_error'] for key,value in metrics.items()};random_error=np.mean([error[f'random{i}'] for i in range(16)],axis=0)
            claims[name+'_'+direction]=(random_error,error['learned'])
            top_effect=paired_effect(error['top_PC'],error['learned'],seeds,n_boot=2000,n_permutations=1999,seed=73)
            mean_metrics={key:{k:float(v.mean()) for k,v in value.items()} for key,value in metrics.items()}
            restoration=1-float(error['learned'].mean()/error['base'].mean());nuisance=mean_metrics['learned'];random_adv=float((random_error-error['learned']).mean());top_adv=float((error['top_PC']-error['learned']).mean())
            model['directions'][direction]=dict(metrics=mean_metrics,phase_restoration_fraction=restoration,random_advantage_radians=random_adv,top_advantage_radians=top_adv,top_PC_comparison=top_effect,
                                                practical_gate=bool(restoration>=.5 and nuisance['amplitude_relative_change']<=.1 and nuisance['offset_change_normalized']<=.1 and random_adv>=.1 and top_adv>=.1))
            gate_rows[name+'_'+direction]=model['directions'][direction]
            for key,value in preds.items():saved[direction+'_'+key]=value
            saved[direction+'_seeds']=seeds
        report['models'][name]=model;np.savez_compressed(out/f'{name}_predictions.npz',**saved)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        del adapter,tokens,frames,modules;gc.collect();torch.cuda.empty_cache()
    if claims:
        report['primary']=compare_effects(claims,paired_rows(metadata,'test')[2],n_boot=2000,n_permutations=1999,seed=73)
    report['warning']='Specialized claim additionally requires primary Holm4 p<.05 and positive bootstrap lowerCI for both random and topPC; no uniqueness claim.'
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(args.out)
