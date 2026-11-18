"""Conditional repeated-patch homogeneity in the installed TimesFM architecture.

Identical normalized patches and masks yield identical value vectors. Position
may alter Q/K attention weights, but any normalized weighted average of equal V
is equal V. Shared pointwise residual/norm/MLP maps preserve this by induction.
The argument neither predicts the head's error nor general context insensitivity.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np


def token_variation(tokens):
    """Measure within-series token variation and alternating-token separation."""
    tokens=np.asarray(tokens,dtype=float)
    if tokens.ndim!=3 or min(tokens.shape)<1 or not np.isfinite(tokens).all():
        raise ValueError('Need finite nonempty[series,token,channel] activations')
    centered=tokens-tokens.mean(axis=1,keepdims=True)
    energy=np.mean(tokens**2,axis=(1,2));variation=np.mean(centered**2,axis=(1,2))
    relative=np.divide(variation,energy,out=np.zeros_like(variation),where=energy>0)
    result=dict(within_token_variance_fraction=float(variation.sum()/energy.sum()) if energy.sum()>0 else 0.,
                median_relative_token_rms=float(np.median(np.sqrt(relative))),max_relative_token_rms=float(np.sqrt(relative).max()),
                max_inter_token_abs_difference=float(np.ptp(tokens,axis=1).max()),zero_energy_series=int(np.sum(energy==0)))
    if tokens.shape[1]>=2:
        even,odd=tokens[:,::2],tokens[:,1::2]
        within=np.sum((even-even.mean(1,keepdims=True))**2)+np.sum((odd-odd.mean(1,keepdims=True))**2)
        result.update(within_parity_variance_fraction=float(within/np.sum(tokens**2)) if np.sum(tokens**2)>0 else 0.,
                      even_odd_mean_separation_mse=float(np.mean((even.mean(1)-odd.mean(1))**2)))
    return result


def attention_average(logits,values):
    """A normalized attention average, independent of how positional Q/K arose."""
    logits,values=np.asarray(logits,dtype=float),np.asarray(values,dtype=float)
    if logits.ndim!=2 or values.ndim!=2 or logits.shape[1]!=len(values) or not np.isfinite(logits).all() or not np.isfinite(values).all():
        raise ValueError('Need finite[query,key] logits and[key,channel] values')
    weight=np.exp(logits-logits.max(axis=1,keepdims=True));weight/=weight.sum(axis=1,keepdims=True)
    return weight@values


def native_source_audit():
    """Record installed source evidence, failing if the reviewed mechanism changes.

    These checks anchor a manual local-source audit rather than claim that string
    matching proves every architecture property for arbitrary future versions.
    """
    from timesfm.torch.transformer import MultiHeadAttention,Transformer
    from timesfm.torch.dense import ResidualBlock
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    targets=dict(attention=MultiHeadAttention.forward,transformer=Transformer.forward,
                 tokenizer=ResidualBlock.forward,model=TimesFM_2p5_200M_torch_module.forward)
    sources={name:inspect.getsource(value) for name,value in targets.items()}
    attention=sources['attention']
    checks=dict(rotary_query='query = self.rotary_position_embedding(query, position)' in attention,
                rotary_key='key = self.rotary_position_embedding(key, position)' in attention,
                no_rotary_value='self.rotary_position_embedding(value' not in attention,
                value_shared_input='value = self.value(inputs_q)' in attention and 'qkv = self.qkv_proj(inputs_q)' in attention,
                no_position_in_tokenizer='position' not in sources['tokenizer'],
                no_position_in_transformer='position' not in sources['transformer'],
                no_absolute_embedding_in_model_forward='position' not in sources['model'])
    if not all(checks.values()):raise ValueError('Installed TimesFM source differs from the reviewed homogeneity premises')
    return dict(checks=checks,source={name:dict(path=inspect.getsourcefile(targets[name]),first_line=inspect.getsourcelines(targets[name])[1],
                sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()) for name,text in sources.items()},
                scope='conditional on identical normalized patch tokens/masks,nonempty attention rows,shared position-free V and pointwise maps;finite precision only approximate',
                conclusion='repeating identical patches preserves the common hidden value independent of the number of repeats;does not establish head correctness')


def compare_context_outputs(predictions,metadata):
    """Report numerical context differences explicitly instead of declaring exact invariance."""
    if set(predictions)!={128,256,512}:raise ValueError('Need matched128/256/512 native forecasts')
    base=np.asarray(predictions[128],dtype=float)
    if base.ndim!=2 or len(base)!=len(metadata) or not np.isfinite(base).all():raise ValueError('Need finite forecasts aligned to metadata')
    result={}
    for context in (256,512):
        other=np.asarray(predictions[context],dtype=float)
        if other.shape!=base.shape or not np.isfinite(other).all():raise ValueError('Context forecasts must have identical finite shapes')
        entries={}
        for condition in ('mixed','pure_phi','pure_psi'):
            keep=np.array([row['condition']==condition for row in metadata])
            if not keep.any():raise ValueError('Need all pure/mixed conditions')
            delta=other[keep]-base[keep]
            entries[condition]=dict(exact_equal=bool(np.array_equal(other[keep],base[keep])),
                max_abs_difference=float(np.abs(delta).max()),rms_difference=float(np.sqrt(np.mean(delta**2))),
                rms_relative_to_base=float(np.sqrt(np.mean(delta**2))/max(float(np.sqrt(np.mean(base[keep]**2))),1e-12)))
        result[str(context)+'_vs_128']=entries
    return result


def run(activation_source,context_source,out):
    """Audit cached residual tokens and saved native predictions on CPU only."""
    out.mkdir(parents=True,exist_ok=False)
    extraction=json.loads((activation_source/'extraction.json').read_text(encoding='utf-8'))
    report=dict(source_audit=native_source_audit(),activations={},context_outputs={},pure32_behavior={})
    for model in ('timesfm','bolt'):
        data=np.load(activation_source/f'{model}_activations.npz')
        report['activations'][model]={layer:token_variation(data[layer]) for layer in extraction['models'][model]['layers']}
    metadata=json.loads((context_source/'metadata.json').read_text(encoding='utf-8'));truth=np.load(context_source/'corpus_context128.npz')['clean_future']
    for model in ('timesfm','bolt','chronos2'):
        predictions={context:np.load(context_source/f'{model}_context{context}_predictions.npz')['prediction'] for context in (128,256,512)}
        report['context_outputs'][model]=compare_context_outputs(predictions,metadata)
        pure=np.array([row['condition']=='pure_phi' for row in metadata]);mse=np.mean((predictions[128][pure]-truth[pure])**2,axis=1)
        a=np.array([row['a'] for row in metadata])[pure];scale=np.array([row['scale'] for row in metadata])[pure]
        report['pure32_behavior'][model]=dict(raw_mse=float(mse.mean()),mse_common_mixvariance=float(np.mean(mse/scale**2)),mse_over_pure32_variance=float(np.mean(mse/(a*a/2))))
    context=np.load(activation_source/'corpus.npz')['context']
    report['raw_input_patch_variation']=dict(timesfm32=token_variation(context.reshape(len(context),-1,32)),bolt16=token_variation(context.reshape(len(context),-1,16)))
    report['caution']='Pure32 can also be homogeneous and near-perfect;homogeneity explains why repeated-patch extension cannot fix this design,not why a nonlinear head first errs;no general context-insensitivity claim'
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(report,indent=2),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activation-source',type=Path,required=True);p.add_argument('--context-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.activation_source,args.context_source,args.out)
