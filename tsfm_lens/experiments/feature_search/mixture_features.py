"""Two-frequency factorial corpus and exact target/nuisance phase instrument.

Unlike a single normalized sinusoid this experiment has two independently
varying internal concepts. Selectivity means changing one phase while keeping
the other phase, both amplitudes and the offset fixed. No pretrained model's
normalization state changes between the four noiseless factorial cells.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np


PROTOCOL=dict(seed_base=1104000,n_bundles=192,context=128,horizon=32,periods=[32,16],
              phase_shift=float(np.pi/2),amplitudes=[.5,2],offset=[-2,2],noise=0,slope=0,
              split='bundles0..95train,96..143validation,144..191test',
              resampling_unit='generating_seed_bundle;average both directions and otherphase cells',
              candidate_layers='first,middle,final',candidate_rank=[2,4],
              target_restoration_gate=.5,other_phase_change_gate=.1,
              amplitude_relative_change_gate=.1,offset_change_over_shared_RMS_gate=.1,
              control_advantage_gate_radians=.1,n_random=16,
              inference='4claims:2models*2targetphases;Holmrandom+pbootstrapCI;topPCmandatory',
              selection='fit train;select validation nuisance-eligible donorphase error;freeze test',
              evidence='task-defined factor-selective causal instrument;no unique representation claim')


def make_corpus():
    """Generate balanced noiseless four-cell independent-phase bundles."""
    contexts,future,metadata=[],[],[];t=np.arange(160)
    for seed in range(192):
        rng=np.random.default_rng(1104000+seed)
        a,b=rng.uniform(.5,2,2);offset=rng.uniform(-2,2);phi,psi=rng.uniform(-np.pi,np.pi,2)
        for cphi,cpsi in [(0,0),(0,1),(1,0),(1,1)]:
            p,q=phi+cphi*np.pi/2,psi+cpsi*np.pi/2
            x=offset+a*np.sin(2*np.pi*t/32+p)+b*np.sin(2*np.pi*t/16+q)
            contexts.append(x[:128]);future.append(x[128:])
            metadata.append(dict(seed=seed,generator_seed=1104000+seed,split='train' if seed<96 else 'validation' if seed<144 else 'test',
                                 cell_phi=cphi,cell_psi=cpsi,phi=p,psi=q,a=a,b=b,offset=offset,scale=float(np.sqrt((a*a+b*b)/2))))
    return np.asarray(contexts,dtype=np.float32),np.asarray(future),metadata


def harmonic_fit(values,start=128):
    """Jointly fit both harmonic phases/amplitudes and offset without leakage."""
    t=np.arange(start,start+values.shape[-1])
    design=np.column_stack([np.sin(2*np.pi*t/32),np.cos(2*np.pi*t/32),np.sin(2*np.pi*t/16),np.cos(2*np.pi*t/16),np.ones(len(t))])
    coef=np.linalg.lstsq(design,values.T,rcond=None)[0].T
    return dict(phi=np.arctan2(coef[:,1],coef[:,0]),psi=np.arctan2(coef[:,3],coef[:,2]),
                a=np.hypot(coef[:,0],coef[:,1]),b=np.hypot(coef[:,2],coef[:,3]),offset=coef[:,4],coefficients=coef)


def pairs(metadata,split,target,reverse=False):
    """Return matched rows and bundle IDs, keeping both nuisance phase cells."""
    target_key='cell_phi' if target=='phi' else 'cell_psi';other_key='cell_psi' if target=='phi' else 'cell_phi'
    lookup={(m['seed'],m[target_key],m[other_key]):i for i,m in enumerate(metadata)}
    seeds=sorted({m['seed'] for m in metadata if m['split']==split});receiver=[];donor=[];groups=[]
    for seed in seeds:
        for other in [0,1]:
            receiver.append(lookup[seed,int(reverse),other]);donor.append(lookup[seed,1-int(reverse),other]);groups.append(seed)
    return np.array(receiver),np.array(donor),np.array(groups)


def metrics(edited,receiver,donor,metadata,target):
    """Score target donor-phase restoration and all unchanged latent nuisances."""
    e,r,d=[harmonic_fit(v) for v in [edited,receiver,donor]];other='psi' if target=='phi' else 'phi'
    angle=lambda a,b:abs(np.angle(np.exp(1j*(a-b))))
    return dict(target_error=angle(e[target],d[target]),target_restoration=angle(r[target],d[target])-angle(e[target],d[target]),
                other_phase_change=angle(e[other],r[other]),a_relative_change=abs(e['a']-r['a'])/np.array([m['a'] for m in metadata]),
                b_relative_change=abs(e['b']-r['b'])/np.array([m['b'] for m in metadata]),
                offset_change_normalized=abs(e['offset']-r['offset'])/np.array([m['scale'] for m in metadata]))


def extract(out):
    """Capture canonical and full raw hidden states from two cached TSFMs."""
    import torch
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.extraction.hooks import _primary
    from tsfm_lens.extraction.alignment import run_alignment_gate
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from activation_interchange import patched_predict
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    context,future,meta=make_corpus();np.savez_compressed(out/'corpus.npz',context=context,clean_future=future)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    patch=context.reshape(len(context),4,32);means=patch.mean(2).reshape(192,4,4);std=patch.std(2).reshape(192,4,4)
    results=dict(normalization=dict(patch_mean_within_bundle_max=float(np.ptp(means,axis=1).max()),patch_std_within_bundle_max=float(np.ptp(std,axis=1).max())),models={})
    for name,cls,checkpoint in [('bolt',ChronosBoltAdapter,'amazon/chronos-bolt-small'),('timesfm',TimesFMAdapter,'google/timesfm-2.5-200m-pytorch')]:
        adapter=cls(ModelConfig(name=name,checkpoint=checkpoint),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded()
        all_layers=adapter.all_layer_names();layers=[all_layers[i] for i in [0,len(all_layers)//2,len(all_layers)-1]]
        alignment=run_alignment_gate(adapter,32,layers);collected={k:[] for k in layers};raw={k:[] for k in layers};handles=[];modules=dict(adapter.module.named_modules())
        for layer in layers:
            def capture(m,i,o,key=layer):
                hidden=_primary(o);raw[key].append(hidden.detach().float().cpu().numpy());collected[key].append(adapter.postprocess_tokens(key,hidden).detach().float().cpu().numpy())
            handles.append(modules[layer].register_forward_hook(capture))
        prediction=patched_predict(adapter,context)
        for h in handles:h.remove()
        tokens={k:np.concatenate(v) for k,v in collected.items()};fullraw={'raw__'+k:np.concatenate(v) for k,v in raw.items()}
        repeat_error=float(abs(patched_predict(adapter,context[:8])-prediction[:8]).max())
        if repeat_error!=0:raise RuntimeError(f'determinism failed:{name}:{repeat_error}')
        self_gates={}
        for layer in layers:
            patched=patched_predict(adapter,context[:8],layer,tokens[layer][:8]);err=float(abs(patched-prediction[:8]).max());self_gates[layer]=err
            if err!=0:raise RuntimeError(f'selfpatch failed:{name}:{layer}:{err}')
        np.savez_compressed(out/f'{name}_activations.npz',prediction=prediction,**tokens,**fullraw)
        results['models'][name]=dict(layers=layers,shapes={k:list(v.shape) for k,v in tokens.items()},alignment=alignment,repeat_max_abs=repeat_error,self_max_abs=self_gates)
        (out/'extraction.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps({name:results['models'][name]}),flush=True)
        del adapter,tokens,fullraw,collected,raw,modules;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    extract(args.out)
