"""Supervised spectral recoverability versus native zero-shot forecast readout.

Matched physical last32-step states are concatenated in native token order,
not pooled. Four normalized harmonic coefficients reconstruct future samples
with an analytic basis and externally available context normalization. These
extra supervised labels make this a representation diagnostic, never a fair
zero-shot forecast improvement claim or evidence of causal feature use.
Bolt's separate decoder query cross-attends ALL encoder memory states; REG
is not its sole readout. Encoder-state probes do not inspect the decoder query.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mixture_features import harmonic_fit


def semantic_targets(metadata):
    """Return [a sinφ/s,a cosφ/s,b sinψ/s,b cosψ/s] in GLOBAL phase."""
    phi,psi,a,b,scale=[np.array([m[k] for m in metadata],dtype=float) for k in ['phi','psi','a','b','scale']]
    return np.column_stack([a*np.sin(phi)/scale,a*np.cos(phi)/scale,b*np.sin(psi)/scale,b*np.cos(psi)/scale])


def basis(start,length):
    """Match target sin/cos PHASE order with cos/sin TIME basis explicitly."""
    t=np.arange(start,start+length)
    return np.column_stack([np.cos(2*np.pi*t/32),np.sin(2*np.pi*t/32),np.cos(2*np.pi*t/16),np.sin(2*np.pi*t/16)])


def reconstruct(coefficients,context,start=128,length=32):
    """Undo known per-series normalization using only available context values."""
    loc=np.mean(context,axis=1);scale=np.std(context,axis=1)
    return (np.asarray(coefficients)@basis(start,length).T)*scale[:,None]+loc[:,None]


def input_oracle(context,start=128,length=32):
    """Recover known-frequency coefficients directly from inputs by joint LS."""
    context_design=np.column_stack([basis(0,context.shape[1]),np.ones(context.shape[1])])
    fitted=np.linalg.lstsq(context_design,np.asarray(context,dtype=float).T,rcond=None)[0]
    return (np.column_stack([basis(start,length),np.ones(length)])@fitted).T


def fit_candidates(features,targets,train,validation,alphas=(.01,1,100)):
    """Fit scaling/ridge on train only; validation chooses regularization."""
    scaler=StandardScaler().fit(features[train]);z=scaler.transform(features)
    fits=[Ridge(alpha=a).fit(z[train],targets[train]) for a in alphas]
    losses=[float(np.mean((fit.predict(z[validation])-targets[validation])**2)) for fit in fits]
    index=int(np.argmin(losses))
    return dict(prediction=fits[index].predict(z),alpha=float(alphas[index]),validation_coefficient_mse=losses[index])


def seed_summary(values,seeds):
    """Average four cells then bootstrap generating bundles, keeping predictions fixed."""
    values=np.asarray(values);unique=np.unique(seeds);cluster=np.array([values[seeds==s].mean() for s in unique])
    rng=np.random.default_rng(9201);means=cluster[rng.integers(0,len(cluster),(2000,len(cluster)))].mean(1)
    return dict(mean=float(cluster.mean()),lo=float(np.quantile(means,.025)),hi=float(np.quantile(means,.975)),n_seeds=len(cluster),inference='fixed predictions;generating_seed_bundle')


def audit(source,out):
    """Freeze layer/alpha on validation and report untouched heldout spectral losses."""
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(evidence='supervised linear representation diagnostic;not causal and not zero-shot improvement',
                  target_order=['a*sinphi/scale','a*cosphi/scale','b*sinpsi/scale','b*cospsi/scale'],
                  context=128,horizon=32,representation='native ordered concatenation of physical[96,128) tokens',
                  layers='first,middle,last from extraction',alpha=[.01,1,100],normalization='actual context populationmean/std,external to hidden',
                  selection='train-only scale/ridge;alpha and layer by validation normalized-coefficient MSE;freeze before test',
                  grouping='train96/val48/test48 generating bundles,all4cells together',baseline='native forecast and input known-frequency joint-LS oracle')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'));corpus=np.load(source/'corpus.npz');context=corpus['context'].astype(float);truth=corpus['clean_future'];targets=semantic_targets(metadata)
    splits=np.array([m['split'] for m in metadata]);train,val,test=[splits==s for s in ['train','validation','test']];seeds=np.array([m['seed'] for m in metadata]);scale=np.array([m['scale'] for m in metadata])
    oracle=input_oracle(context);oracle_mse=np.mean((oracle-truth)**2,axis=1)/scale**2
    oracle_fit=harmonic_fit(oracle);report=dict(input_oracle_scaled_mse=seed_summary(oracle_mse[test],seeds[test]),normalizer_max_abs_difference_from_latent=dict(mean=float(abs(context.mean(1)-np.array([m['offset'] for m in metadata])).max()),scale=float(abs(context.std(1)-scale).max())),models={})
    saved=dict(test_seeds=seeds[test],test_targets=targets[test],input_oracle=oracle[test],truth=truth[test]);frozen={}
    models={}
    for name,count in [('bolt',2),('timesfm',1)]:
        store=np.load(source/f'{name}_activations.npz');layers=[k for k in store.files if k!='prediction' and not k.startswith('raw__')];candidates=[];predictions={}
        for layer in layers:
            tokens=store[layer]
            expected=8 if name=='bolt' else 4
            if tokens.shape[1]!=expected:raise ValueError(f'Unexpected token geometry:{name}:{tokens.shape}')
            features=tokens[:,-count:].reshape(len(tokens),-1).astype(float)
            result=fit_candidates(features,targets,train,val);predictions[layer]=result['prediction'];candidates.append(dict(layer=layer,feature_dim=features.shape[1],tokens_used=count,physical_span=[96,128],alpha=result['alpha'],validation_coefficient_mse=result['validation_coefficient_mse']))
        chosen=min(candidates,key=lambda c:c['validation_coefficient_mse']);frozen[name]=dict(selection=chosen,candidates=candidates);models[name]=(store,predictions[chosen['layer']])
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    for name,(store,predicted) in models.items():
        reconstruction=reconstruct(predicted,context);native=store['prediction'];native_coeff=harmonic_fit(native)
        native_targets=np.column_stack([native_coeff['coefficients'][:,1],native_coeff['coefficients'][:,0],native_coeff['coefficients'][:,3],native_coeff['coefficients'][:,2]])/scale[:,None]
        native_loss=np.mean((native-truth)**2,axis=1)/scale**2;probe_loss=np.mean((reconstruction-truth)**2,axis=1)/scale**2
        model=dict(frozen[name],native_scaled_future_mse=seed_summary(native_loss[test],seeds[test]),probe_scaled_future_mse=seed_summary(probe_loss[test],seeds[test]),
                   native_minus_probe_loss=seed_summary((native_loss-probe_loss)[test],seeds[test]),frequency_coefficients={})
        for label,selection in [('period32',slice(0,2)),('period16',slice(2,4))]:
            model['frequency_coefficients'][label]=dict(probe_mse=seed_summary(np.mean((predicted[:,selection]-targets[:,selection])**2,axis=1)[test],seeds[test]),
                native_mse=seed_summary(np.mean((native_targets[:,selection]-targets[:,selection])**2,axis=1)[test],seeds[test]))
        report['models'][name]=model;saved[name+'_predicted_coefficients']=predicted[test];saved[name+'_reconstructed_future']=reconstruction[test];saved[name+'_native_future']=native[test];saved[name+'_probe_scaled_loss']=probe_loss[test];saved[name+'_native_scaled_loss']=native_loss[test]
    report['caveat']='Probes use synthetic supervised target labels and analytic known frequencies plus external scale/offset;no fair zero-shot improvement or causal-use claim.'
    np.savez_compressed(out/'test_predictions.npz',**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


ARCHITECTURE_SCOPE=dict(correction='Prior conversation claim that Bolt head reads REG only was incorrect.',
    verified_source='chronos.chronos_bolt.ChronosBoltModelForForecasting.forward/decode',
    verified_behavior='encode returns all hidden_states;decode passes them as encoder_hidden_states to a decoder_start query;output_patch_embedding consumes decoder output',
    scope='Context and REG are encoder-memory diagnostics;neither is the decoder query or a complete native-head input audit.')


def fixed_last_layer_profile(source,out):
    """Audit fixed final-layer full context,last32 and Bolt REG without layer selection."""
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'));corpus=np.load(source/'corpus.npz');context=corpus['context'].astype(float);truth=corpus['clean_future'];targets=semantic_targets(meta)
    split=np.array([m['split'] for m in meta]);train,validation,test=[split==s for s in ['train','validation','test']];seeds=np.array([m['seed'] for m in meta]);scale=np.array([m['scale'] for m in meta]);choices={};fitted={};native={}
    protocol=dict(scope=ARCHITECTURE_SCOPE,layer='fixed final captured layer;no layer selection',surfaces=['native full-context ordered concat','native last32 ordered concat','Bolt REG separately'],
        fitting='train-only StandardScaler/Ridge;alpha[.01,1,100] validation-only per surface;all choices saved before test',evidence='supervised recoverability diagnostic;not causal or fair zero-shot improvement')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for name,layer,lastcount in [('bolt','encoder.block.5',2),('timesfm','stacked_xf.19',1)]:
        store=np.load(source/f'{name}_activations.npz');tokens=store[layer];surfaces={'full_context_concat':tokens.reshape(len(tokens),-1),'last32_concat':tokens[:,-lastcount:].reshape(len(tokens),-1)}
        if name=='bolt':
            raw=store['raw__'+layer]
            if raw.shape[1]!=tokens.shape[1]+1:raise ValueError('Expected exactly one separately captured Bolt REG token')
            surfaces['REG']=raw[:,-1]
        native[name]=store['prediction'];choices[name]={}
        for surface,features in surfaces.items():
            fit=fit_candidates(features.astype(float),targets,train,validation);fitted[name,surface]=fit['prediction'];choices[name][surface]=dict(layer=layer,feature_dim=features.shape[1],alpha=fit['alpha'],validation_coefficient_mse=fit['validation_coefficient_mse'])
    (out/'frozen_selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    report=dict(architecture_scope=ARCHITECTURE_SCOPE,models={});saved=dict(test_seeds=seeds[test],targets=targets[test])
    for name in ['bolt','timesfm']:
        report['models'][name]={};native_loss=np.mean((native[name]-truth)**2,axis=1)/scale**2
        for surface,choice in choices[name].items():
            prediction=fitted[name,surface];future=reconstruct(prediction,context);loss=np.mean((future-truth)**2,axis=1)/scale**2
            result=dict(choice,scaled_future_mse=seed_summary(loss[test],seeds[test]),native_scaled_future_mse=seed_summary(native_loss[test],seeds[test]),native_minus_probe_loss=seed_summary((native_loss-loss)[test],seeds[test]),frequency_coefficients={})
            for label,columns in [('period32',slice(0,2)),('period16',slice(2,4))]:result['frequency_coefficients'][label]=seed_summary(np.mean((prediction[:,columns]-targets[:,columns])**2,axis=1)[test],seeds[test])
            report['models'][name][surface]=result;saved[name+'_'+surface+'_coefficients']=prediction[test];saved[name+'_'+surface+'_future']=future[test]
    oracle=input_oracle(context);report['input_oracle_scaled_mse']=seed_summary((np.mean((oracle-truth)**2,axis=1)/scale**2)[test],seeds[test])
    np.savez_compressed(out/'test_predictions.npz',**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fixed-last-layer',action='store_true');args=p.parse_args();(fixed_last_layer_profile if args.fixed_last_layer else audit)(args.source,args.out)
