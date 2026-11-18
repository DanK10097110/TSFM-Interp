"""Cached forecast-output versus frozen SH-25 head-input linear recoverability.

Extra supervised semantic labels and known harmonic basis make this an
exploratory representation diagnostic, never a deployable forecast repair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from head_readout import fit_frozen_head
from mixture_readout import semantic_targets,reconstruct,input_oracle,seed_summary

MODELS=('bolt','timesfm')
FROZEN_KEYS=('scaler_mean','scaler_scale','coefficients','intercept')


def digest(path):
    """Bind exact cached bytes before fitting or evaluating a readout."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(metadata,context,truth):
    """Check complete generating bundles, ordered physical labels and scales."""
    context,truth=np.asarray(context,float),np.asarray(truth,float)
    if context.shape!=(len(metadata),128) or truth.shape!=(len(metadata),32) or not np.isfinite(context).all() or not np.isfinite(truth).all():
        raise ValueError('Finite context128/future32 arrays must match ordered metadata')
    ids=np.array([m['generator_seed'] for m in metadata])
    unique=np.unique(ids)
    row_ids=[]
    for seed in unique:
        rows=[m for m in metadata if m['generator_seed']==seed]
        if len(rows)!=4 or {(m['cell_phi'],m['cell_psi']) for m in rows}!={(0,0),(0,1),(1,0),(1,1)} or len({m['split'] for m in rows})!=1:
            raise ValueError('All four phase cells must stay in one generating-seed split')
        if any(len({m[key] for m in rows})!=1 for key in ('a','b','offset','scale')):
            raise ValueError('Amplitude/scale nuisance changed inside seed bundle')
    for m in metadata:
        if m['a']<=0 or m['b']<=0 or not np.isclose(m['scale']**2,(m['a']**2+m['b']**2)/2,rtol=1e-12):raise ValueError('Invalid common mix energy')
        row_ids.append(f"{m['generator_seed']}:{m['cell_phi']}:{m['cell_psi']}")
    if len(set(row_ids))!=len(row_ids):raise ValueError('Duplicate physical row identity')
    t=np.arange(160)
    expected=np.array([m['offset']+m['a']*np.sin(2*np.pi*t/32+m['phi'])+m['b']*np.sin(2*np.pi*t/16+m['psi']) for m in metadata])
    if not np.allclose(context,expected[:,:128],atol=3e-7,rtol=1e-7) or not np.allclose(truth,expected[:,128:],atol=3e-7,rtol=1e-7):
        raise ValueError('Physical row order, labels or time origin mismatch')
    return np.array(row_ids),ids


def output_features(prediction,context):
    """Normalize physical forecasts using the same external context chart as SH-25."""
    context=np.asarray(context,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
    if prediction.shape!=(len(context),32) or not np.isfinite(prediction).all():raise ValueError('Finite native32-step forecasts required')
    scale=context.std(1)
    if np.any(scale<=0):raise ValueError('Zero context scale')
    return (prediction-context.mean(1)[:,None])/scale[:,None]


def fit_output(old,fresh,old_context,fresh_context,targets,train,validation):
    """Reuse exact SH-25 float64 train-scaler/ridge/validation machinery."""
    return fit_frozen_head(output_features(old,old_context),output_features(fresh,fresh_context),targets,train,validation)


def frozen_prediction(features,weights):
    """Apply the existing saved head probe without refitting or reselection."""
    return ((np.asarray(features,float)-weights['scaler_mean'])/weights['scaler_scale'])@weights['coefficients'].T+weights['intercept']


def gates(output_mean,excess_mean):
    """Descriptive predeclared point gates, not a population equivalence test."""
    return bool(output_mean<=.01 and excess_mean<=.01)


def load_datasets(source):
    """Validate actual old/fresh cached rows before touching learned readouts."""
    datasets={}
    for key,n,base in [('old',192,1104000),('fresh',128,1215000)]:
        metadata=json.loads((source/f'{key}_metadata.json').read_text(encoding='utf-8'))
        with np.load(source/f'{key}_corpus.npz',allow_pickle=False) as file:context=file['context'].astype(float);truth=file['clean_future'].astype(float)
        row_ids,seeds=validate_rows(metadata,context,truth)
        if set(seeds)!=set(range(base,base+n)):raise ValueError('Unexpected generating-seed cohort')
        split=np.array([m['split'] for m in metadata])
        if key=='old' and any(len(set(seeds[split==s]))!=count for s,count in [('train',96),('validation',48),('test',48)]):raise ValueError('SH-25 old96/48/48 split changed')
        datasets[key]=dict(metadata=metadata,context=context,truth=truth,row_ids=row_ids,seeds=seeds,split=split,targets=semantic_targets(metadata),scale=np.array([m['scale'] for m in metadata]))
    if set(datasets['old']['seeds'])&set(datasets['fresh']['seeds']):raise ValueError('Generating cohorts overlap')
    return datasets


def run(source,head_probe,out):
    """Save protocol before fitting, freeze both output readouts before evaluation."""
    out.mkdir(parents=True,exist_ok=False)
    source_names=['protocol.json','provenance.json']+[f'{key}_{kind}' for key in ('old','fresh') for kind in ('metadata.json','corpus.npz')]+[f'{m}_{key}_head.npz' for m in MODELS for key in ('old','fresh')]
    probe_names=['frozen_selection.json','predictions.npz','results.json']+[f'{m}_frozen_probe.npz' for m in MODELS]
    bindings={'captures':{name:digest(source/name) for name in source_names},'head_probe':{name:digest(head_probe/name) for name in probe_names}}
    helpers={Path(fn.__code__.co_filename).resolve() for fn in (fit_frozen_head,semantic_targets,reconstruct,seed_summary)}
    protocol=dict(scope='Exploratory cached-cohort linear recoverability;both outcomes previously inspected;not new heldout confirmation',models=MODELS,source=str(source.resolve()),frozen_head_probe=str(head_probe.resolve()),bindings=bindings,driver_sha256=digest(__file__),helper_sha256={str(path):digest(path) for path in helpers},
        targets='identical SH25[a sinφ/s,a cosφ/s,b sinψ/s,b cosψ/s]',input='native32-step point forecast;normalize by float64 context population mean/std,identical external chart available to both reconstructions',
        fitting='exact reused fit_frozen_head:float64 StandardScaler onold96trainseeds only;Ridge alpha[.01,1,100] chosen onold48validationseeds;no refit onvalidation;old48test+fresh128evaluation',
        primary='fresh128 generating-bundle outputdecoder scaledfutureMSE and paired outputminusfrozenhead excess',gates='outputpointmean<=.01 AND excesspointmean<=.01;descriptive only,no inferential equivalence claim',intervals='pointwise approximate paired generating-seed bootstrap2000 seed9201;four cells averaged perseed;no pvalues',
        normalization='physical forecast outputs renormalized using context only;knownharmonic reconstruction plus externalcontextmean/std matches SH25;commonmixvariance(a²+b²)/2 forloss',
        interpretation='Success means task-specific linear recovery from forecast output,not fairzero-shot improvement or deployable repair;failure doesnotprove information destruction or absence',
        provenance_limit='Original legacy head caches have no embedded rowIDs;this protocol binds their exact bytes to validated ordered corpus/metadata and producer hookidentity declarations,then crosschecks frozen head predictions. It cannot independently prove original row-order execution.',
        time_budget_seconds=180,cpu_threads=1,no_nonlinear_fallback=True)
    text=json.dumps(protocol,indent=2);(out/'protocol.json').write_text(text,encoding='utf-8')
    capture_protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'));provenance=json.loads((source/'provenance.json').read_text(encoding='utf-8'))
    if capture_protocol['horizon']!=32 or capture_protocol['context']!=128:raise ValueError('Capture task changed')
    datasets=load_datasets(source);old,fresh=datasets['old'],datasets['fresh'];train=old['split']=='train';validation=old['split']=='validation'
    archived_report=json.loads((head_probe/'results.json').read_text(encoding='utf-8'))
    fits={};stored={};head_coeff={}
    with np.load(head_probe/'predictions.npz',allow_pickle=False) as archived:
        for model in MODELS:
            stored[model]={};head_coeff[model]={}
            with np.load(head_probe/f'{model}_frozen_probe.npz',allow_pickle=False) as file:weights={k:file[k].copy() for k in FROZEN_KEYS}
            for key,dataset in datasets.items():
                with np.load(source/f'{model}_{key}_head.npz',allow_pickle=False) as file:features=file['features'].copy();prediction=file['prediction'].copy()
                declaration=provenance[model]['datasets'][key]
                if declaration['hook_identity_max_abs']!=0 or declaration['n_rows']!=len(dataset['metadata']) or list(features.shape)!=declaration['shape']:raise ValueError('Legacy native capture declaration mismatch')
                if not np.isfinite(features).all():raise ValueError('Nonfinite head input')
                mask=dataset['split']=='test' if key=='old' else np.ones(len(prediction),bool)
                row_loss=np.mean((np.asarray(prediction,float)-dataset['truth'])**2,axis=1)/dataset['scale']**2
                previous=archived_report['models'][model]['datasets'][key]['native_scaled_mse']['mean']
                if not np.isclose(row_loss[mask].mean(),previous,rtol=1e-10,atol=1e-12):raise ValueError('Cached native forecast/label archive mismatch')
                stored[model][key]=dict(prediction=prediction)
                coefficients=frozen_prediction(features,weights);head_coeff[model][key]=coefficients
                mask=dataset['split']=='test' if key=='old' else np.ones(len(coefficients),bool)
                if not np.allclose(coefficients[mask],archived[f'{model}_{key}_coefficients'],rtol=1e-9,atol=1e-10):raise ValueError('Frozen head probe or cached row lineage changed')
            fits[model]=fit_output(stored[model]['old']['prediction'],stored[model]['fresh']['prediction'],old['context'],fresh['context'],old['targets'],train,validation)
            np.savez_compressed(out/f'{model}_frozen_output_readout.npz',**{k:fits[model][k] for k in FROZEN_KEYS})
    choices={model:dict(alpha=fit['alpha'],validation_coefficient_mse=fit['validation_mse']) for model,fit in fits.items()}
    (out/'frozen_selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    result=dict(scope=protocol['scope'],protocol_sha256=hashlib.sha256(text.encode()).hexdigest(),models={});saved={}
    for model in MODELS:
        result['models'][model]=dict(selection=choices[model],datasets={})
        for key,dataset in datasets.items():
            mask=dataset['split']=='test' if key=='old' else np.ones(len(dataset['metadata']),bool)
            native=stored[model][key]['prediction'];output=reconstruct(fits[model][key+'_prediction'],dataset['context']);head=reconstruct(head_coeff[model][key],dataset['context']);truth=dataset['truth'];scale=dataset['scale'];seeds=dataset['seeds']
            losses={name:np.mean((value-truth)**2,axis=1)/scale**2 for name,value in [('native',native),('output_readout',output),('frozen_head_readout',head),('known_frequency_input_oracle',input_oracle(dataset['context']))]}
            report={name:seed_summary(value[mask],seeds[mask]) for name,value in losses.items()}
            excess=losses['output_readout']-losses['frozen_head_readout'];report['output_minus_head_excess']=seed_summary(excess[mask],seeds[mask]);report['descriptive_point_gate']=gates(report['output_readout']['mean'],report['output_minus_head_excess']['mean'])
            result['models'][model]['datasets'][key]=report
            for name,value in losses.items():saved[f'{model}_{key}_{name}_loss']=value[mask]
            saved[f'{model}_{key}_output_coefficients']=fits[model][key+'_prediction'][mask];saved[f'{model}_{key}_output_future']=output[mask]
        result['models'][model]['fresh_gate_pass']=result['models'][model]['datasets']['fresh']['descriptive_point_gate']
    for key,dataset in datasets.items():
        mask=dataset['split']=='test' if key=='old' else np.ones(len(dataset['metadata']),bool)
        saved[key+'_row_ids']=dataset['row_ids'][mask];saved[key+'_generator_ids']=dataset['seeds'][mask]
    np.savez_compressed(out/'source_data.npz',**saved)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--head-probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.head_probe,a.out)
