"""Right-aligned adjacent-period challenge of conditional patch homogeneity.

Nontiling periods require a full harmonic Gram matrix. Frequency preference,
normalization and task difficulty remain confounds: behavior is not a sole-cause
architecture attribution. No production orthogonality constraint is relaxed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.feature_search.patch_homogeneity import token_variation
from experiments.feature_search.normalization_audit import normalization_state
from tsfm_lens.analysis.stats import mean_ci
from tsfm_lens.experiments.feature_search.statistics import paired_effect

PROTOCOL_PATH=Path(__file__).with_name('patch_context_protocol.json')


def load_protocol():
    """Read the prespecified single-contrast behavioral challenge."""
    return json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def make_corpora(protocol=None):
    """Make matched periods with a common future origin and exact context suffixes."""
    protocol=load_protocol() if protocol is None else protocol
    keys=[(p,c) for p in protocol['target_periods'] for c in protocol['contexts']]
    contexts={key:[] for key in keys};truths={key:[] for key in keys};metadata=[]
    for seed in range(protocol['n_bundles']):
        rng=np.random.default_rng(protocol['seed_base']+seed);a,b=rng.uniform(*protocol['amplitudes'],2);offset=rng.uniform(*protocol['offset']);phi,psi=rng.uniform(-np.pi,np.pi,2)
        cells=[('mixed',p,q) for p in (0,1) for q in (0,1)]+[('pure_phi',p,0) for p in (0,1)]+[('pure_psi',0,q) for q in (0,1)]
        for condition,p,q in cells:
            angle_phi,angle_psi=phi+p*protocol['phase_shift'],psi+q*protocol['phase_shift']
            metadata.append(dict(seed=seed,generator_seed=protocol['seed_base']+seed,split='test',condition=condition,cell_phi=p,cell_psi=q,
                                 phi=angle_phi,psi=angle_psi,a=a,b=b,offset=offset,scale=float(np.sqrt((a*a+b*b)/2))))
            for period,context in keys:
                t=np.arange(-context,protocol['horizon']);values=np.full(len(t),offset)
                if condition!='pure_psi':values+=a*np.sin(2*np.pi*t/period+angle_phi)
                if condition!='pure_phi':values+=b*np.sin(2*np.pi*t/protocol['other_period']+angle_psi)
                contexts[period,context].append(values[:context]);truths[period,context].append(values[context:])
    contexts={key:np.array(value,dtype=np.float32) for key,value in contexts.items()};truths={key:np.array(value) for key,value in truths.items()}
    for period in protocol['target_periods']:
        short,long=sorted(protocol['contexts'])
        if not np.array_equal(contexts[period,short],contexts[period,long][:,-short:]):raise ValueError('Right-aligned short context must equal long-context suffix')
        if not np.array_equal(truths[period,short],truths[period,long]):raise ValueError('Contexts must share exactly the same physical future')
    return contexts,truths,metadata


def harmonic_fit(values,period,*,other_period=16,start=0):
    """Fit both components using the actual full-rank, potentially nondiagonal Gram."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or values.shape[1]<5 or not np.isfinite(values).all() or min(period,other_period)<=2:raise ValueError('Need finite forecasts and identifiable positive periods')
    t=np.arange(start,start+values.shape[1]);design=np.column_stack((np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period),np.sin(2*np.pi*t/other_period),np.cos(2*np.pi*t/other_period),np.ones(len(t))))
    condition=float(np.linalg.cond(design))
    if np.linalg.matrix_rank(design)!=5 or condition>1e6:raise ValueError('Harmonic design is rank-deficient or poorly conditioned')
    coefficients=np.linalg.lstsq(design,values.T,rcond=None)[0].T
    return dict(coefficients=coefficients,gram=design.T@design/len(t),residual=values-coefficients@design.T,condition_number=condition)


def error_decomposition(prediction,truth,period):
    """Reconstruct exact MSE with fullGram cross terms and orthogonal residuals."""
    prediction,truth=np.asarray(prediction,dtype=float),np.asarray(truth,dtype=float)
    if prediction.shape!=truth.shape:raise ValueError('Prediction and truth shapes must match')
    p,t=harmonic_fit(prediction,period),harmonic_fit(truth,period);delta=p['coefficients']-t['coefficients'];gram=p['gram']
    projected=np.einsum('ni,ij,nj->n',delta,gram,delta);diagonal=np.sum(delta**2*np.diag(gram),axis=1)
    residual=np.mean((p['residual']-t['residual'])**2,axis=1);direct=np.mean((prediction-truth)**2,axis=1)
    if not np.allclose(projected+residual,direct,atol=1e-10,rtol=1e-9):raise ValueError('FullGram plus residual failed exact MSE identity')
    return dict(total_mse=direct,projected_mse=projected,residual_mse=residual,gram_cross_terms=projected-diagonal,
                reconstructed_mse=projected+residual,gram=gram,condition_number=p['condition_number'])


def analyze(predictions,truths,metadata,*,protocol=None):
    """Measure the frozen neighboring-period contrast and descriptive context changes."""
    protocol=load_protocol() if protocol is None else protocol;settings=protocol['inference']
    seeds=np.unique([row['seed'] for row in metadata]);groups=np.array([row['seed'] for row in metadata]);condition=np.array([row['condition'] for row in metadata]);energy=np.array([row['scale']**2 for row in metadata])
    if any(not np.isfinite(row['scale']) or row['scale']<=0 or not np.isclose(row['scale']**2,(row['a']**2+row['b']**2)/2) for row in metadata):raise ValueError('Need positive common theoretical component energy')
    def average(values,keep):return np.array([np.asarray(values)[keep&(groups==seed)].mean() for seed in seeds])
    def interval(values):return mean_ci(values,n_boot=settings['n_boot'],seed=settings['seed'],unit='generating_seed_bundle')
    report,benefits={},{}
    for period in protocol['target_periods']:
        entries={}
        for model in protocol['models']:
            measurements={};losses={}
            for context in protocol['contexts']:
                pred=np.asarray(predictions[model,period,context]);truth=np.asarray(truths[period,context])
                if pred.shape!=truth.shape:raise ValueError('Forecasts must align to actual future rows')
                decomposition=error_decomposition(pred,truth,period);loss=decomposition['total_mse']/energy;losses[context]=average(loss,condition=='mixed')
                parts=harmonic_fit(pred,period)['coefficients'];behavior={}
                for target,columns,amp_key in [('phi',slice(0,2),'a'),('psi',slice(2,4),'b')]:
                    present=np.array([row['condition']!='pure_'+('psi' if target=='phi' else 'phi') for row in metadata])
                    amplitudes=np.array([row[amp_key] for row in metadata]);predamp=np.linalg.norm(parts[:,columns],axis=1);phase=np.arctan2(parts[:,columns][:,1],parts[:,columns][:,0]);truephase=np.array([row[target] for row in metadata]);valid=present&(predamp>1e-6*amplitudes)
                    behavior[target]=dict(relative_amplitude_error=float(np.mean(np.abs(predamp[present]/amplitudes[present]-1))),phase_error_radians=None if not valid.any() else float(np.mean(np.abs(np.angle(np.exp(1j*(phase[valid]-truephase[valid])))))),phase_unidentifiable=int(np.sum(present&~valid)))
                measurements[str(context)]=dict(mse_common_theoretical_energy={c:interval(average(loss,condition==c)) for c in ('mixed','pure_phi','pure_psi')},raw_mse=float(decomposition['total_mse'].mean()),
                    gram=decomposition['gram'].tolist(),gram_offdiagonal_max=float(np.max(np.abs(decomposition['gram']-np.diag(np.diag(decomposition['gram']))))),condition_number=decomposition['condition_number'],
                    fullgram_reconstruction_max_abs=float(np.max(np.abs(decomposition['total_mse']-decomposition['reconstructed_mse']))),mean_gram_cross_terms_raw_mse=float(decomposition['gram_cross_terms'].mean()),behavior=behavior)
            short,long=sorted(protocol['contexts']);benefits[model,period]=losses[short]-losses[long]
            difference=np.mean((predictions[model,period,short]-predictions[model,period,long])**2,axis=1)/energy
            entries[model]=dict(contexts=measurements,mixed_context_benefit=interval(benefits[model,period]),
                native_output_change_mse_common_theoretical_energy={c:interval(average(difference,condition==c)) for c in ('mixed','pure_phi','pure_psi')})
        report[str(period)]=entries
    neighbors=.5*(benefits['timesfm',31]+benefits['timesfm',33]);tiled=benefits['timesfm',32]
    primary=paired_effect(neighbors,tiled,seeds,n_boot=settings['n_boot'],n_permutations=settings['n_permutations'],seed=settings['seed'],ci=settings['ci']);primary['passed']=bool(primary['value']>=protocol['minimum_practical'] and primary['lo']>0 and primary['p']<.05)
    primary.update(minimum_practical=protocol['minimum_practical'],family_size=1,unit='common theoretical component-energy MSE;neighbor-mean context benefit minus period32 benefit')
    return dict(primary=primary,conditions=report,n_seeds=len(seeds),
        caution='Adjacent period,finite-window normalizer and task difficulty vary;positive contrast is not sole-cause architecture evidence;fullGram used for nontiling periods',
        units='raw MSE explicitly labeled;all scaled errors use SAME theoretical(a*a+b*b)/2,not realized finite-window variance')


def capture_homogeneity(adapter,contexts,layers):
    """Capture only per-series scalar token variation from representative actual weights."""
    from experiments.feature_search.activation_interchange import patched_predict
    from tsfm_lens.extraction.hooks import _primary
    modules=dict(adapter.module.named_modules());captured={layer:[] for layer in layers};handles=[]
    baseline=patched_predict(adapter,contexts)
    for layer in layers:
        def capture(m,i,o,key=layer):
            hidden=adapter.postprocess_tokens(key,_primary(o)).detach().float().cpu().numpy()
            for row in hidden:captured[key].append(token_variation(row[None]))
        handles.append(modules[layer].register_forward_hook(capture))
    try:prediction=patched_predict(adapter,contexts)
    finally:
        for handle in handles:handle.remove()
    error=float(np.max(np.abs(prediction-baseline)))
    if error!=0:raise RuntimeError('Native hook/nohook forecast identity failed')
    if any(len(value)!=len(contexts) for value in captured.values()):raise ValueError('Unexpected multiple native backbone passes')
    return dict(hook_nohook_max_abs=error,n_series=len(contexts),layers={layer:{key:float(np.mean([row[key] for row in values])) for key in values[0]} for layer,values in captured.items()})


def run(out):
    """Run only cached two-model native forecasts after saving the frozen protocol."""
    import torch
    from experiments.feature_search.activation_interchange import patched_predict
    from tsfm_lens.config import DataConfig,ModelConfig
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
    protocol=load_protocol();out.mkdir(parents=True,exist_ok=False);protocol_bytes=PROTOCOL_PATH.read_bytes();(out/'protocol.json').write_bytes(protocol_bytes)
    contexts,truths,metadata=make_corpora(protocol);(out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for (period,context),value in contexts.items():np.savez_compressed(out/f'corpus_period{period}_context{context}.npz',context=value,clean_future=truths[period,context])
    predictions={};homogeneity={};normalizers={};provenance=dict(protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),models={});started=time.perf_counter()
    sample=np.array([row['seed']<8 and row['condition']=='mixed' for row in metadata]);print(json.dumps(dict(total_native_contexts=3072,estimated_seconds=[60,180],runtime_cap=600)),flush=True)
    for model,cls,surface in [('timesfm',TimesFMAdapter,'timesfm_2p5_decode'),('bolt',ChronosBoltAdapter,'chronos_bolt')]:
        adapter=cls(ModelConfig(name=model,checkpoint=protocol['models'][model]),DataConfig(context_len=128,horizon=32),torch.device('cuda:0'),torch.float32);adapter.ensure_loaded();provenance['models'][model]={}
        for period in protocol['target_periods']:
            for context in protocol['contexts']:
                if time.perf_counter()-started>600:
                    (out/'paused.json').write_text(json.dumps(dict(reason='ten-minute cap',model=model,period=period,context=context)),encoding='utf-8');return
                block=time.perf_counter();adapter.data_cfg=DataConfig(context_len=context,horizon=32);values=contexts[period,context]
                prediction=patched_predict(adapter,values);repeat=patched_predict(adapter,values[:8]);repeat_error=float(np.max(np.abs(repeat-prediction[:8])))
                if repeat_error!=0:raise RuntimeError('Native same-batch repeat failed')
                predictions[model,period,context]=prediction;np.savez_compressed(out/f'{model}_period{period}_context{context}_predictions.npz',prediction=prediction)
                mu,sigma=normalization_state(values,surface);normalizers[model,period,context]=(mu[:,0],sigma[:,0])
                if model=='timesfm':
                    names=adapter.all_layer_names();layers=[names[i] for i in [0,len(names)//2,len(names)-1]];homogeneity[f'period{period}_context{context}']=capture_homogeneity(adapter,values[sample],layers)
                provenance['models'][model][f'{period}_{context}']=dict(repeat_max_abs=repeat_error,seconds=time.perf_counter()-block)
                (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print(json.dumps(dict(model=model,period=period,context=context,seconds=time.perf_counter()-block,elapsed=time.perf_counter()-started)),flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    result=analyze(predictions,truths,metadata,protocol=protocol);scale=np.array([row['scale'] for row in metadata]);groups=np.array([row['seed'] for row in metadata]);mixed=np.array([row['condition']=='mixed' for row in metadata])
    changes={}
    for model in protocol['models']:
        changes[model]={}
        for period in protocol['target_periods']:
            mu_short,s_short=normalizers[model,period,128];mu_long,s_long=normalizers[model,period,512]
            changes[model][str(period)]=dict(mean_absolute_location_change_over_theoretical_scale=float(np.mean(np.abs(mu_long[mixed]-mu_short[mixed])/scale[mixed])),
                mean_absolute_sigma_change_over_theoretical_scale=float(np.mean(np.abs(s_long[mixed]-s_short[mixed])/scale[mixed])),max_absolute_location_change=float(np.max(np.abs(mu_long-mu_short))),max_absolute_sigma_change=float(np.max(np.abs(s_long-s_short))))
    result.update(homogeneity=homogeneity,input_normalizer_context_changes=changes)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(primary=result['primary'],normalizers=changes),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(min(1.,6e9/torch.cuda.get_device_properties(0).total_memory))
    run(args.out)
