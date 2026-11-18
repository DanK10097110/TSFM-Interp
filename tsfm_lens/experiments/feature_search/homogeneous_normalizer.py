"""CPU source-exact TimesFM homogeneous-stack and radial-normalizer diagnostic.

Identical normalized32-step patches collapse QK routing to a shared value map.
This explains context invariance, not composition error. Fixed-normalizer inputs
are explicit off-manifold counterfactuals. No fitting, p-values or repair claim.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import resource
import time

import numpy as np


def digest(path):
    """Stream exact checkpoint/source bytes without a second full resident copy."""
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def weighted_values(weights,values):
    """A row-stochastic router cannot distinguish identical V vectors."""
    weights,values=np.asarray(weights),np.asarray(values)
    if weights.ndim!=2 or values.ndim!=2 or weights.shape[1]!=len(values) or np.any(weights<0) or not np.allclose(weights.sum(1),1):
        raise ValueError('Need finite row-stochastic weights and matched values')
    return weights@values


def radial_identity(function,u,v,s,sa,sb):
    """Exact native=fixed+radial decomposition, retaining F(0) and affine bias."""
    combined,first,second,zero,pure_a,pure_b=[np.asarray(function(z)) for z in
        (u+v,u,v,np.zeros_like(u),u*s/sa,v*s/sb)]
    native=s*combined-sa*pure_a-sb*pure_b
    fixed=s*(combined-first-second+zero)
    radial=(s*first-sa*pure_a)+(s*second-sb*pure_b)-s*zero
    return native,fixed,radial


def attribution(native,components,scale,denominator_floor=1e-8):
    """Report exact vector/energy closure and signed shares, with small-R flags."""
    native=np.asarray(native,dtype=float);scale=np.asarray(scale,dtype=float)
    if native.ndim!=2 or scale.shape!=(len(native),) or np.any(scale<=0):raise ValueError('Need waveform rows and positive common scales')
    energies=np.mean(native**2,axis=1)/scale**2
    flags=energies<=denominator_floor
    normalized=native/scale[:,None]
    denominator=float(np.sum(normalized**2))
    parts={key:np.asarray(value,dtype=float)/scale[:,None] for key,value in components.items()}
    output={}
    for key,value in parts.items():
        output[key]=dict(normalized_energy=np.mean(value**2,axis=1).tolist(),
            aggregate_signed_projection_share=None if denominator<=denominator_floor*len(native)*native.shape[1] else float(np.sum(normalized*value)/denominator))
    names=list(parts)
    cross={a+'__'+b:(2*np.mean(parts[a]*parts[b],axis=1)).tolist() for i,a in enumerate(names) for b in names[i+1:]}
    combined=sum(parts.values())
    energy_closure=energies-sum(np.mean(x*x,axis=1) for x in parts.values())-sum(np.asarray(x) for x in cross.values())
    return dict(native_normalized_energy=energies.tolist(),small_native_residual=flags.tolist(),denominator_floor=denominator_floor,
        components=output,cross_normalized_energy=cross,vector_closure_max=float(np.max(abs(normalized-combined))),
        energy_closure_max=float(np.max(abs(energy_closure))),
        scope='Signed projection shares may be negative or above1; descriptive algebra, not causal blame')


def collapsed(model,patch,horizon=64):
    """Evaluate native tokenwise value/residual/FFN path with QK provably inert."""
    import torch
    z=torch.as_tensor(patch,dtype=torch.float32)
    if z.ndim==1:z=z[None]
    if z.shape[1]!=32:raise ValueError('Need one32-step normalized patch per row')
    with torch.inference_mode():
        h=model.tokenizer(torch.cat([z,torch.zeros_like(z)],dim=-1))
        for layer in model.stacked_xf:
            inputs=layer.pre_attn_ln(h)
            attention=layer.attn
            value=attention.qkv_proj(inputs).chunk(3,dim=-1)[2] if attention.fuse_qkv else attention.value(inputs)
            attended=layer.post_attn_ln(attention.out(value))+h
            h=layer.post_ff_ln(layer.ff1(layer.activation(layer.ff0(layer.pre_ff_ln(attended)))))+attended
        output=model.output_projection_point(h).reshape(len(h),128,10)[:,:horizon,model.aridx]
    return output.numpy().astype(float)


def load_model(checkpoint):
    """Assign mmap-backed cached weights into meta construction; never use CUDA."""
    import torch
    from safetensors.torch import load_file
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    torch.set_num_threads(1)
    if torch.cuda.is_available():raise RuntimeError('CPU-only task requires CUDA_VISIBLE_DEVICES empty')
    with torch.device('meta'):model=TimesFM_2p5_200M_torch_module()
    model.load_state_dict(load_file(str(checkpoint),device='cpu'),assign=True,strict=True)
    model.eval()
    return model


def source_data(root):
    """Read only the known fresh64 controlled cohort and its native predictions."""
    prepared=root/'tsfm_lens/runs/feature_search_stage49_crossover3400000_prepared'
    run=root/'tsfm_lens/runs/feature_search_stage49_crossover3400000'
    with np.load(prepared/'p32_16_c128.npz',allow_pickle=False) as source:data={k:source[k] for k in source.files}
    metadata=json.loads((prepared/'p32_16_c128.json').read_text(encoding='utf-8'))
    with np.load(run/'timesfm_p32_16_c128.npz',allow_pickle=False) as source:native={k:source[k] for k in source.files}
    if not np.array_equal(data['row_ids'],native['row_ids']) or str(native['corpus_sha256'].item())!=digest(prepared/'p32_16_c128.npz'):
        raise ValueError('Native row identity/corpus binding mismatch')
    checkpoint=Path(json.loads((root/'examples/feature_search_20261111/native_head_replication_source_data.json').read_text(encoding='utf-8'))['protocol']['manifests']['timesfm']['checkpoint'])
    return data,metadata,native,checkpoint,dict(corpus=digest(prepared/'p32_16_c128.npz'),metadata=digest(prepared/'p32_16_c128.json'),native_forecast=digest(run/'timesfm_p32_16_c128.npz'))


def manifest(root,mode):
    """Freeze source/checkpoint bindings and resource limits before any replay."""
    import timesfm.timesfm_2p5.timesfm_2p5_torch as implementation
    from timesfm.torch import transformer,dense,normalization,util
    data,metadata,native,checkpoint,bindings=source_data(root)
    sources={m.__name__:digest(inspect.getfile(m)) for m in [implementation,transformer,dense,normalization,util]}
    return data,metadata,native,checkpoint,dict(mode=mode,driver_sha256=digest(__file__),source_sha256=sources,
        checkpoint=str(checkpoint),resolved_checkpoint=str(checkpoint.resolve()),checkpoint_weight_sha256=digest(checkpoint),bindings=bindings,
        calibration_ids=[3400000],pilot_ids=[3400000+4*j for j in range(16)],
        horizon=64,point_channel=5,replay_normalized_waveform_RMS=1e-5,
        small_residual_normalized_energy=1e-8,cpu_threads=1,resident_limit_gib=1.5,
        runtime_seconds=60 if mode=='calibrate' else 180,
        available_memory_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if line.startswith('MemAvailable:'))),
        checkpoint_bytes=checkpoint.stat().st_size,
        interpretation='inspected controlled cohort; descriptive source replay only, no statistical replication, no repair',
        counterfactual='Fixed mixture normalizer evaluates lower-RMS u and v off the native unit-RMS manifold; no causal-isolated-feature claim')


def calibration(model,data,metadata,native,ids):
    """Replay all six native cells including background, using actual saved stats."""
    chosen=[i for i,row in enumerate(metadata) if row['generator_seed'] in ids]
    contexts=data['context'][chosen]
    mu=native['native_mu'].reshape(-1)[chosen].astype(float)
    sigma=native['native_sigma'].reshape(-1)[chosen].astype(float)
    patch=(contexts[:,-32:].astype(float)-mu[:,None])/np.where(sigma[:,None]<1e-6,1.,sigma[:,None])
    replay=collapsed(model,patch)*sigma[:,None]+mu[:,None]
    expected=native['prediction'][chosen].astype(float)
    scale=np.asarray([np.sqrt((metadata[i]['a']**2+metadata[i]['b']**2)/2) for i in chosen])
    error=np.sqrt(np.mean(((replay-expected)/scale[:,None])**2,axis=1))
    return dict(maximum_common_scale_waveform_RMS=float(error.max()),passed=bool(error.max()<=1e-5),
                per_cell=[dict(generator_seed=metadata[i]['generator_seed'],condition=metadata[i]['condition'],normalized_RMS=float(e)) for i,e in zip(chosen,error)])


def run(root,out,mode,calibration_source=None):
    """Calibrate once, then separately register IDs before authorized counterfactuals."""
    started=time.monotonic()
    out.mkdir(parents=True,exist_ok=False)
    data,metadata,native,checkpoint,protocol=manifest(root,mode)
    if mode=='pilot':
        if calibration_source is None:raise ValueError('Pilot requires saved calibration')
        previous=json.loads((calibration_source/'results.json').read_text(encoding='utf-8'))
        previous_protocol=json.loads((calibration_source/'protocol.json').read_text(encoding='utf-8'))
        if not previous['calibration']['passed'] or previous_protocol['driver_sha256']!=protocol['driver_sha256'] or previous_protocol['checkpoint_weight_sha256']!=protocol['checkpoint_weight_sha256']:
            raise ValueError('Calibration failed or frozen source/checkpoint changed')
        protocol['calibration_source']=str(calibration_source)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=load_model(checkpoint)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1.5*1024**3:raise RuntimeError('Resident limit exceeded after load')
    ids=protocol['calibration_ids'] if mode=='calibrate' else protocol['pilot_ids']
    verified=calibration(model,data,metadata,native,ids)
    results=dict(calibration=verified)
    if verified['passed'] and mode=='pilot':
        lookup={(row['generator_seed'],row['condition']):i for i,row in enumerate(metadata)}
        stored={key:[] for key in ('native_residual','analytic_native_residual','fixed_normalizer_residual','radial_remainder','native_replay_residual')}
        scales=[]
        for seed in ids:
            if time.monotonic()-started>protocol['runtime_seconds']:raise RuntimeError('Registered CPU runtime cap reached')
            row=metadata[lookup[seed,'baseline']]
            a,b=row['a'],row['b'];s=np.sqrt((a*a+b*b)/2);sa=a/np.sqrt(2);sb=b/np.sqrt(2)
            t=np.arange(-32,0)
            u=a*np.sin(2*np.pi*t/32+row['phi'])/s
            v=b*np.sin(2*np.pi*t/16+row['psi'])/s
            function=lambda z:collapsed(model,z)[0]
            analytical,fixed,radial=radial_identity(function,u,v,s,sa,sb)
            ix={c:lookup[seed,c] for c in ('baseline','pure_phi','pure_psi','background')}
            actual=np.asarray(native['prediction'][ix['baseline']],dtype=float)-native['prediction'][ix['pure_phi']]-native['prediction'][ix['pure_psi']]+native['prediction'][ix['background']]
            for key,value in [('native_residual',actual),('analytic_native_residual',analytical),('fixed_normalizer_residual',fixed),('radial_remainder',radial),('native_replay_residual',actual-analytical)]:stored[key].append(value)
            scales.append(s)
            print(json.dumps(dict(seed=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
        stored={key:np.asarray(value) for key,value in stored.items()};scale=np.asarray(scales)
        results['attribution']=attribution(stored['native_residual'],dict(fixed_normalizer=stored['fixed_normalizer_residual'],radial=stored['radial_remainder'],replay=stored['native_replay_residual']),scale)
        results['mean_normalized_energies']={key:float(np.mean(value**2/scale[:,None]**2)) for key,value in stored.items()}
        np.savez_compressed(out/'residual_waveforms.npz',**stored,scale=scale,generator_seed=np.asarray(ids))
    results.update(wall_seconds=time.monotonic()-started,peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   scope=protocol['interpretation'],counterfactual=protocol['counterfactual'])
    (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['calibrate','pilot'],required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.repo_root,args.out,args.mode,args.calibration)
