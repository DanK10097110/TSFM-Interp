"""Native forecast-head input capture and frozen supervised spectral diagnostics.

Bolt's output_patch_embedding consumes its decoder query, not a REG-only
encoder readout. TimesFM output_projection_point consumes transformer outputs,
and decode selects the last context position for this short horizon. Capture
requires exact declared shapes and a single hook firing; it never changes a
prediction. Supervised recovery is task-specific and does not establish a
general mechanism or a fair zero-shot performance improvement.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mixture_readout import semantic_targets,reconstruct,input_oracle,seed_summary
from mixture_features import harmonic_fit


def head_tensor(inputs,*,batch,positions,width):
    """Require one tensor positional argument with the exact prospective shape."""
    if not isinstance(inputs,tuple) or len(inputs)!=1 or not isinstance(inputs[0],torch.Tensor):
        raise TypeError('Head prehook requires exactly one positional Tensor input')
    value=inputs[0]
    if tuple(value.shape)!=(batch,positions,width):
        raise ValueError(f'Unexpected head input shape:{tuple(value.shape)};expected{(batch,positions,width)}')
    if not torch.isfinite(value).all():raise ValueError('Nonfinite native head input')
    return value[:,-1,:]


def fit_frozen_head(old,fresh,targets,train,validation):
    """Fit on old train,choose alpha on old validation,and leave both tests untouched."""
    old=np.asarray(old,dtype=np.float64);fresh=np.asarray(fresh,dtype=np.float64);targets=np.asarray(targets,dtype=np.float64)
    scaler=StandardScaler().fit(old[train]);z=scaler.transform(old);zz=scaler.transform(fresh)
    fits=[Ridge(alpha=a).fit(z[train],targets[train]) for a in [.01,1,100]]
    loss=[float(np.mean((m.predict(z[validation])-targets[validation])**2)) for m in fits];index=int(np.argmin(loss));chosen=fits[index]
    return dict(old_prediction=chosen.predict(z),fresh_prediction=chosen.predict(zz),alpha=[.01,1,100][index],validation_mse=loss[index],
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,coefficients=chosen.coef_,intercept=chosen.intercept_)


def capture(old_source,fresh_source,out):
    """Capture actual native head inputs offline,with exact no-hook identity gates."""
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    import gc
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(evidence='native-head-input supervised recoverability;not causal/head editing/zero-shot improvement',old_seed_base=1104000,fresh_seed_base=1215000,
                  old_split='96train48validation48testbundles',fresh='128entireheldoutbundles,condition mixed only,no head slice',
                  head_modules=dict(bolt='output_patch_embedding prehook:[B,1,512],position0 decoder query',timesfm='output_projection_point prehook:[B,4,1280],position3 transformer output'),
                  shape_policy='exact singleTensor tuple,onehookfire,no guessedreshape',alpha=[.01,1,100],selection='oldtrain fit,oldvalidation alpha frozen,oldtest+fresh independent report',
                  identity='predict without hook vs with capture hook maxabs exactly0 perbatch',horizon=32,context=128,batch=8,precision='float32,no autocast')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    datasets={}
    for key,source in [('old',old_source),('fresh',fresh_source)]:
        metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'));corpus=np.load(source/'corpus.npz');indices=np.arange(len(metadata)) if key=='old' else np.flatnonzero([m.get('condition')=='mixed' for m in metadata])
        if key=='fresh' and (len(indices)!=512 or len({metadata[i]['seed'] for i in indices})!=128):raise ValueError('Fresh mixture mask not exactly128complete4cellbundles')
        selected=[metadata[i] for i in indices];datasets[key]=(corpus['context'][indices],corpus['clean_future'][indices],selected)
        (out/f'{key}_metadata.json').write_text(json.dumps(selected,indent=2),encoding='utf-8');np.savez_compressed(out/f'{key}_corpus.npz',context=corpus['context'][indices],clean_future=corpus['clean_future'][indices],source_indices=indices)
    provenance={}
    for name,cls,checkpoint,module_name,positions,width in [('bolt',ChronosBoltAdapter,'amazon/chronos-bolt-small','output_patch_embedding',1,512),('timesfm',TimesFMAdapter,'google/timesfm-2.5-200m-pytorch','output_projection_point',4,1280)]:
        adapter=cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded();modules=dict(adapter.module.named_modules())
        if module_name not in modules:raise ValueError(f'Missing exact native head module:{name}:{module_name}')
        source_path=Path(inspect.getsourcefile(type(adapter.module)));provenance[name]=dict(checkpoint=checkpoint,model_class=type(adapter.module).__qualname__,source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),module=module_name,expected_positions=positions,width=width,selected_position=positions-1,datasets={})
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        for key,(context,future,meta) in datasets.items():
            features=[];predictions=[];max_error=0.
            for start in range(0,len(context),8):
                batch=context[start:start+8]
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                    baseline=adapter.predict(batch,32,[.5])['point'];hits=[]
                    def prehook(module,inputs):hits.append(head_tensor(inputs,batch=len(batch),positions=positions,width=width).detach().float().cpu().numpy().copy())
                    handle=modules[module_name].register_forward_pre_hook(prehook)
                    try:prediction=adapter.predict(batch,32,[.5])['point']
                    finally:handle.remove()
                if len(hits)!=1:raise RuntimeError(f'Expected exactly one native-head firing:{name}:{len(hits)}')
                error=float(abs(baseline-prediction).max());max_error=max(max_error,error)
                if error!=0:raise RuntimeError(f'Capture changed native forecast:{name}:{key}:{error}')
                features.append(hits[0]);predictions.append(prediction)
            h=np.concatenate(features);p=np.concatenate(predictions);np.savez_compressed(out/f'{name}_{key}_head.npz',features=h,prediction=p)
            provenance[name]['datasets'][key]=dict(shape=list(h.shape),hook_identity_max_abs=max_error,n_rows=len(h))
            print(json.dumps({name+'_'+key:provenance[name]['datasets'][key]}),flush=True)
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');del adapter,modules;gc.collect();torch.cuda.empty_cache()


def analyze(source,out):
    """Evaluate a single frozen old-trained probe on old test and new128seeds."""
    out.mkdir(parents=True,exist_ok=False);datasets={}
    for key in ['old','fresh']:
        metadata=json.loads((source/f'{key}_metadata.json').read_text(encoding='utf-8'));corpus=np.load(source/f'{key}_corpus.npz');datasets[key]=dict(metadata=metadata,context=corpus['context'].astype(float),truth=corpus['clean_future'],targets=semantic_targets(metadata),seeds=np.array([m['seed'] for m in metadata]),scale=np.array([m['scale'] for m in metadata]))
    old=datasets['old'];split=np.array([m['split'] for m in old['metadata']]);train,validation,test=[split==s for s in ['train','validation','test']];choices={};fits={};stores={}
    for name in ['bolt','timesfm']:
        stores[name]={key:np.load(source/f'{name}_{key}_head.npz') for key in ['old','fresh']};fit=fit_frozen_head(stores[name]['old']['features'],stores[name]['fresh']['features'],old['targets'],train,validation);fits[name]=fit;choices[name]=dict(alpha=fit['alpha'],validation_coefficient_mse=fit['validation_mse'])
        np.savez_compressed(out/f'{name}_frozen_probe.npz',**{k:fit[k] for k in ['scaler_mean','scaler_scale','coefficients','intercept']})
    (out/'frozen_selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8');report=dict(scope='actual native head-input spectral recoverability;supervised extra labels+known basis+externalnormalizer;not causal/general mechanism/zero-shot improvement',probe_precision='float64 scaling/ridge on float32 captured states',models={});saved={}
    for name in ['bolt','timesfm']:
        report['models'][name]=dict(selection=choices[name],datasets={})
        for key,dataset in datasets.items():
            mask=test if key=='old' else np.ones(len(dataset['metadata']),dtype=bool);predicted=fits[name][key+'_prediction'];reconstructed=reconstruct(predicted,dataset['context']);native=stores[name][key]['prediction'];truth=dataset['truth'];scale=dataset['scale'];seeds=dataset['seeds'];targets=dataset['targets'];oracle=input_oracle(dataset['context'])
            native_loss=np.mean((native-truth)**2,axis=1)/scale**2;probe_loss=np.mean((reconstructed-truth)**2,axis=1)/scale**2;oracle_loss=np.mean((oracle-truth)**2,axis=1)/scale**2
            result=dict(native_scaled_mse=seed_summary(native_loss[mask],seeds[mask]),probe_scaled_mse=seed_summary(probe_loss[mask],seeds[mask]),input_oracle_scaled_mse=seed_summary(oracle_loss[mask],seeds[mask]),native_minus_probe_loss=seed_summary((native_loss-probe_loss)[mask],seeds[mask]),coefficients={})
            for phase,columns in [('period32',slice(0,2)),('period16',slice(2,4))]:result['coefficients'][phase]=seed_summary(np.mean((predicted[:,columns]-targets[:,columns])**2,axis=1)[mask],seeds[mask])
            report['models'][name]['datasets'][key]=result;saved[name+'_'+key+'_coefficients']=predicted[mask];saved[name+'_'+key+'_future']=reconstructed[mask]
    np.savez_compressed(out/'predictions.npz',**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);c=sub.add_parser('capture');c.add_argument('--old-source',type=Path,required=True);c.add_argument('--fresh-source',type=Path,required=True);c.add_argument('--out',type=Path,required=True);a=sub.add_parser('analyze');a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2 if args.mode=='capture' else 1)
    if args.mode=='capture':
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
        capture(args.old_source,args.fresh_source,args.out)
    else:analyze(args.source,args.out)
