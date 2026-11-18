"""Full-token anchored composition restoration with a numerical-anchor stop gate.

The intervention tests suffix compatibility with a declared affine composition
hypothesis, not where harmful layers originate. Actual background states remain
native; a canonical-zero comparison diagnoses numerical anchor ambiguity. The
fresh pilot cannot run without separate authorization and a passing calibration.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import inspect
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import binomtest
from experiments.feature_search import homogeneous_normalizer as replay

CUTS=(0,5,10,15,20)
CANDIDATES=(0,5,10,15)
CONDITIONS=('baseline','pure_phi','pure_psi','background')


def protocol(mode):
    """Freeze identities, contrasts, resource bounds and stop gates before replay."""
    return dict(mode=mode,calibration_ids=[3400000,3400004],pilot_ids=list(range(3600000,3600016)),
        replication_reserved_ids=[3700000,3700015],conditions=CONDITIONS,periods=[32,16],context=128,horizon=64,
        cuts=CUTS,candidates=CANDIDATES,late_control=20,primary_family_size=8,
        random_controls=2,random_seed=3600173,random_geometry_tolerance_float64=1e-8,random_geometry_tolerance_injected_float32=1e-5,
        random_geometry_reference_floor=1.,random_geometry_floor_scope='Relative errors use max(1,absolute reference); below1 this is an explicit absolute numerical tolerance, not scientific loss normalization',random_constraint='One shared feature-space isometry per seed fixes span of all receiver token rows; preserves token edit Gram and every receiver/edit cross dot; no D-by-D matrix',
        minimum_native_gain=.1,minimum_random_advantage=.05,maximum_suffix_energy=.05,
        minimum_ceiling_recovery=.75,minimum_aggregate_ceiling_gap=.1,
        native_replay_RMS=1e-5,anchor_sensitivity_RMS=.01,affine_B_invariance_RMS=1e-5,
        cpu_threads=1,resident_limit_gib=1.5,runtime_seconds=60 if mode=='calibrate' else 180,
        selection='Only earliest fully qualifying candidate earns fresh replication; late/raw/descriptive arms cannot rescue',
        inference='global Holm8 exact two-sided conditional sign tests; positive majority plus empirical mean margin; no mean-superiority claim',
        estimands=['native mixed loss minus anchored edited loss','mean of two fixed random-control losses minus anchored edited loss'],
        units='All losses and suffix energies use the same theoretical (a²+b²)/2 per seed',
        interpretation='Background-anchored amplitude-weighted state hypothesis; off-manifold edit, not canonical component feature or causal-layer localization',
        background='Actual full native four-token states; canonical zero-normalized background is diagnostic only and never replaces actual anchor',
        calibration_output='Numerical checks only; suppress scientific loss/gain selection results')


def anchored(mixed,first,second,background,scale_m,scale_a,scale_b):
    """Affine coefficients sum1, fixing hidden-coordinate translation dependence."""
    shape=(len(scale_m),)+(1,)*(np.asarray(mixed).ndim-1)
    a=(np.asarray(scale_a)/scale_m).reshape(shape)
    b=(np.asarray(scale_b)/scale_m).reshape(shape)
    mixed,first,second,background=[np.asarray(x,dtype=float) for x in (mixed,first,second,background)]
    return background+a*(first-background)+b*(second-background)


def random_edits(receiver,edit,rng,count=2,return_diagnostics=False):
    """Keep edit energy and receiver radial component, randomizing only orientation."""
    h=np.asarray(receiver,dtype=float)
    e=np.asarray(edit,dtype=float)
    if h.shape!=e.shape or h.ndim not in (2,3):raise ValueError('Need matched batch/token/feature arrays')
    original_shape=h.shape
    if h.ndim==2:h=h[:,None,:];e=e[:,None,:]
    samples=np.empty((count,)+h.shape)
    flags=np.zeros(len(h),dtype=bool)
    diagnostics=[[] for _ in h]
    for seed,(rows,edits) in enumerate(zip(h,e)):
        _,singular,right=np.linalg.svd(rows,full_matrices=False)
        tolerance=np.finfo(float).eps*max(rows.shape)*(singular[0] if len(singular) else 0.)
        rank=int(np.sum(singular>tolerance))
        fixed=right[:rank]
        parallel=(edits@fixed.T)@fixed
        orthogonal=edits-parallel
        left,strength,basis=np.linalg.svd(orthogonal,full_matrices=False)
        residual_rank=int(np.sum(strength>max(1e-10,np.finfo(float).eps*max(orthogonal.shape)*(strength[0] if len(strength) else 0.))))
        if residual_rank==0 or rows.shape[1]-rank<residual_rank:
            flags[seed]=True
            samples[:,seed]=edits
            continue
        coefficients=left[:,:residual_rank]*strength[:residual_rank]
        for draw in range(count):
            raw=rng.normal(size=(rows.shape[1],residual_rank))
            raw-=fixed.T@(fixed@raw)
            if np.linalg.matrix_rank(raw)<residual_rank:
                flags[seed]=True
                samples[draw,seed]=edits
                diagnostics[seed].append(dict(reason='rank-deficient QR reference'))
                continue
            rotated,_=np.linalg.qr(raw,mode='reduced')
            proposed=parallel+coefficients@rotated.T
            ideal=geometry_errors(rows,edits,proposed)
            receiver32=rows.astype(np.float32).astype(float)
            desired32=(rows+edits).astype(np.float32).astype(float)-receiver32
            control32=(rows+proposed).astype(np.float32).astype(float)-receiver32
            actual=geometry_errors(receiver32,desired32,control32)
            diagnostics[seed].append(dict(float64=ideal,injected_float32=actual))
            if max(ideal.values())>1e-8 or max(actual.values())>1e-5:
                flags[seed]=True
                samples[draw,seed]=edits
            else:samples[draw,seed]=proposed
    result=(samples.reshape((count,)+original_shape),flags)
    return result+(diagnostics,) if return_diagnostics else result


def geometry_errors(receiver,desired,control):
    """Bound actual edit Gram, receiver/edit cross dots and final token Gram errors."""
    h,e,r=[np.asarray(x,dtype=float) for x in (receiver,desired,control)]
    relative=lambda actual,expected:float(np.max(abs(actual-expected))/max(1.,float(np.max(abs(expected)))))
    per_token=lambda actual,expected:float(np.max(abs(actual-expected)/np.maximum(1.,abs(expected))))
    return dict(edit_gram=relative(r@r.T,e@e.T),receiver_cross=relative(r@h.T,e@h.T),
                final_token_gram=relative((h+r)@(h+r).T,(h+e)@(h+e).T),
                per_token_energy=per_token(np.sum(r*r,-1),np.sum(e*e,-1)),
                per_token_radial=per_token(np.sum(r*h,-1),np.sum(e*h,-1)))


def affine_remainder(scales,means,background_output):
    """Return cut-invariant B, including native bias and actual background scale."""
    sm,sa,sb,s0=[np.asarray(scales[c]) for c in CONDITIONS]
    mm,ma,mb,m0=[np.asarray(means[c]) for c in CONDITIONS]
    return (sm-sa-sb+s0)[:,None]*background_output+(mm-ma-mb+m0)[:,None]


def nonlinear_remainder(function,state,first,second,background,weights,scale):
    """An affine-combination remainder may cancel; no convex Jensen interpretation."""
    a,b=weights
    return scale[:,None]*(function(state)-a[:,None]*function(first)-b[:,None]*function(second)-(1-a-b)[:,None]*function(background))


def make_data(ids):
    """Generate four native branches at a common right-aligned physical origin."""
    contexts,truth,metadata=[],[],[]
    for seed in ids:
        rng=np.random.default_rng(seed)
        a,b=rng.uniform(.5,2.,2);phi,psi=rng.uniform(-np.pi,np.pi,2);offset=rng.uniform(-2.,2.)
        for condition in CONDITIONS:
            include_a=condition not in ('pure_psi','background')
            include_b=condition not in ('pure_phi','background')
            def signal(t):return offset+a*np.sin(2*np.pi*t/32+phi)*include_a+b*np.sin(2*np.pi*t/16+psi)*include_b
            contexts.append(signal(np.arange(-128,0)));truth.append(signal(np.arange(64)))
            metadata.append(dict(generator_seed=seed,condition=condition,a=a,b=b,phi=phi,psi=psi,offset=offset))
    return np.asarray(contexts,dtype=np.float32),np.asarray(truth),metadata


def normalizers(context):
    """Replay installed prefix running statistics with the identical FP32 helpers."""
    import torch
    from timesfm.torch import util
    x=torch.as_tensor(context,dtype=torch.float32)
    n=torch.zeros(len(x));mu=n.clone();sigma=n.clone()
    for patch in x.reshape(len(x),4,32).unbind(1):
        (n,mu,sigma),_=util.update_running_stats(n,mu,sigma,patch,torch.zeros_like(patch,dtype=torch.bool))
    return mu.numpy(),sigma.numpy()


@contextmanager
def capture_hooks(model,caches):
    """Observe actual source decode at the full four-token cutpoints, no editing."""
    handles=[]
    handles.append(model.tokenizer.register_forward_pre_hook(lambda module,args:caches.setdefault('raw',[]).append(args[0][...,:32].detach().clone())))
    handles.append(model.tokenizer.register_forward_hook(lambda module,args,out:caches.setdefault(0,[]).append(out.detach().clone())))
    handles.append(model.output_projection_point.register_forward_hook(lambda module,args,out:caches.setdefault('normalized_head',[]).append(out.detach().clone())))
    for cut in CUTS[1:]:
        def hook(module,args,out,cut=cut):caches.setdefault(cut,[]).append(out[0].detach().clone())
        handles.append(model.stacked_xf[cut-1].register_forward_hook(hook))
    try:yield
    finally:
        for handle in handles:handle.remove()


def suffix(model,state,cut):
    """Resume the exact full native transformer suffix and point head, four tokens."""
    import torch
    h=torch.as_tensor(state,dtype=torch.float32)
    mask=torch.zeros(h.shape[:2],dtype=torch.bool)
    with torch.inference_mode():
        for layer in model.stacked_xf[cut:]:h,_=layer(h,mask,None)
        output=model.output_projection_point(h).reshape(len(h),4,128,10)[:,-1,:64,model.aridx]
    return output.numpy().astype(float)


def native_capture(model,contexts):
    """Use actual decode for natural baselines and actual normalized background."""
    import torch
    cached={};predictions=[]
    for start in range(0,len(contexts),8):
        x=torch.as_tensor(contexts[start:start+8],dtype=torch.float32)
        with torch.inference_mode(),capture_hooks(model,cached):
            output,_,extra=model.decode(64,x,torch.zeros_like(x,dtype=torch.bool))
        if extra is not None and extra.numel():raise ValueError('Unexpected autoregressive rollout')
        predictions.append(output[:,-1,:64,model.aridx].float().numpy())
    states={key:torch.cat(value).numpy() for key,value in cached.items()}
    if any(value.shape[1]!=4 for value in states.values()):raise ValueError('Expected all four native token positions')
    return np.concatenate(predictions).astype(float),states


def canonical_background(model,n):
    """Get zero-normalized states separately; these never replace the actual anchor."""
    import torch
    cached={}
    with torch.inference_mode(),capture_hooks(model,cached):
        model(torch.zeros(n,4,32),torch.zeros(n,4,32,dtype=torch.bool))
    return {key:value[0].numpy() for key,value in cached.items()}


def calibrated_run(model,contexts,metadata):
    """Check source transport/B invariance and every anchor sensitivity before gains."""
    pred,states=native_capture(model,contexts)
    mu,sigma=normalizers(contexts)
    n=len(metadata)//4
    ix={c:np.arange(n)*4+i for i,c in enumerate(CONDITIONS)}
    means={c:mu[index].astype(float) for c,index in ix.items()}
    scales={c:sigma[index].astype(float) for c,index in ix.items()}
    common=np.array([np.sqrt((metadata[i*4]['a']**2+metadata[i*4]['b']**2)/2) for i in range(n)])
    if np.any(scales['baseline']<=0) or np.any(scales['pure_phi']<=0) or np.any(scales['pure_psi']<=0):raise ValueError('Mixture/pure native scales must be positive')
    zero=canonical_background(model,n)
    checks=dict(transport={},normalized_transport={},anchor_sensitivity={},B_invariance={})
    natural_sums=pred[ix['pure_phi']]+pred[ix['pure_psi']]-pred[ix['background']]
    outputs={};Bs={};remainders={}
    for cut in CUTS:
        branch={c:states[cut][index] for c,index in ix.items()}
        transported={c:suffix(model,value,cut) for c,value in branch.items()}
        for condition in CONDITIONS:
            reference=states['normalized_head'][ix[condition]][:,-1].reshape(n,128,10)[:,:64,model.aridx].astype(float)
            checks['normalized_transport'][f'{cut}/{condition}']=float(np.sqrt(np.mean((transported[condition]-reference)**2,axis=1)).max())
            physical=means[condition][:,None]+scales[condition][:,None]*transported[condition]
            checks['transport'][f'{cut}/{condition}']=float(np.sqrt(np.mean(((physical-pred[ix[condition]])/common[:,None])**2,axis=1)).max())
        actual=anchored(branch['baseline'],branch['pure_phi'],branch['pure_psi'],branch['background'],scales['baseline'],scales['pure_phi'],scales['pure_psi'])
        alternate=anchored(branch['baseline'],branch['pure_phi'],branch['pure_psi'],zero[cut],scales['baseline'],scales['pure_phi'],scales['pure_psi'])
        edited=suffix(model,actual,cut)*scales['baseline'][:,None]+means['baseline'][:,None]
        alternate_output=suffix(model,alternate,cut)*scales['baseline'][:,None]+means['baseline'][:,None]
        checks['anchor_sensitivity'][str(cut)]=np.sqrt(np.mean(((edited-alternate_output)/common[:,None])**2,axis=1)).tolist()
        B=affine_remainder(scales,means,transported['background'])
        Bs[cut]=B
        a=scales['pure_phi']/scales['baseline'];b=scales['pure_psi']/scales['baseline']
        nonlinear=scales['baseline'][:,None]*((edited-means['baseline'][:,None])/scales['baseline'][:,None]-a[:,None]*transported['pure_phi']-b[:,None]*transported['pure_psi']-(1-a-b)[:,None]*transported['background'])
        remainders[cut]=nonlinear
        discrepancy=edited-natural_sums-B-nonlinear
        checks.setdefault('identity_closure',{})[str(cut)]=float(np.sqrt(np.mean((discrepancy/common[:,None])**2,axis=1)).max())
        identity=anchored(branch['baseline'],branch['baseline'],branch['background'],branch['background'],scales['baseline'],scales['baseline'],scales['background'])
        checks.setdefault('self_patch_state',{})[str(cut)]=float(np.max(abs(identity-branch['baseline'])))
        outputs[cut]=dict(anchored=edited,state=actual,receiver=branch['baseline'])
    for cut in CUTS:checks['B_invariance'][str(cut)]=float(np.sqrt(np.mean(((Bs[cut]-Bs[0])/common[:,None])**2,axis=1)).max())
    # Raw normalized input reconstruction is a negative control, not a candidate.
    raw={c:states['raw'][index] for c,index in ix.items()}
    raw_anchor=anchored(raw['baseline'],raw['pure_phi'],raw['pure_psi'],raw['background'],scales['baseline'],scales['pure_phi'],scales['pure_psi'])
    checks['raw_input_state_RMS']=np.sqrt(np.mean((raw_anchor-raw['baseline'])**2,axis=(1,2))).tolist()
    import torch
    with torch.inference_mode():
        raw_hidden=model.tokenizer(torch.cat([torch.as_tensor(raw_anchor,dtype=torch.float32),torch.zeros_like(torch.as_tensor(raw_anchor,dtype=torch.float32))],dim=-1))
    raw_pred=suffix(model,raw_hidden,0)*scales['baseline'][:,None]+means['baseline'][:,None]
    checks['raw_input_waveform_RMS']=np.sqrt(np.mean(((raw_pred-pred[ix['baseline']])/common[:,None])**2,axis=1)).tolist()
    checks['maximum_anchor_sensitivity']=max(max(v) for v in checks['anchor_sensitivity'].values())
    checks['passed']=numerical_gate(checks)
    return checks,dict(prediction=pred,states=states,means=means,scales=scales,common=common,outputs=outputs,B=Bs,nonlinear=remainders,component_sum=natural_sums)


def numerical_gate(checks):
    """Every numerical candidate and late cut must pass; never average away failure."""
    return bool(max(checks['transport'].values())<=1e-5 and max(checks['normalized_transport'].values())<=1e-5 and max(checks['B_invariance'].values())<=1e-5 and
        max(checks['identity_closure'].values())<=1e-5 and max(checks['raw_input_waveform_RMS'])<=1e-5 and
        max(checks['raw_input_state_RMS'])<=1e-5 and max(checks['self_patch_state'].values())<=1e-8 and
        max(max(values) for values in checks['anchor_sensitivity'].values())<=.01)


def qualify(native,component,anchored_loss,random_loss,suffix_energy,random_identifiable=None):
    """Complete Holm8 precedes fixed size/recovery gates and earliest-cut selection."""
    from tsfm_lens.analysis.feature_interference import sign_comparison
    from tsfm_lens.analysis.stats import holm
    native,component=np.asarray(native),np.asarray(component)
    gap=float(np.mean(native-component))
    claims={}
    for cut in CANDIDATES:
        for endpoint,difference in [('native_gain',native-anchored_loss[cut]),('random_advantage',random_loss[cut]-anchored_loss[cut])]:
            valid=random_identifiable is None or bool(np.all(random_identifiable[cut]))
            value=sign_comparison(difference if endpoint!='random_advantage' or valid else np.full(len(difference),np.nan))
            if endpoint=='random_advantage' and not valid:
                value['untestable']='Unidentifiable orthogonal edit or random geometry for at least one registered seed; no seed removal; cut cannot qualify'
                value['unidentifiable_seed_indices']=np.flatnonzero(~np.asarray(random_identifiable[cut])).tolist()
            value['mean']=float(np.mean(difference))
            claims[f'{cut}/{endpoint}']=value
    adjusted=holm({key:value['p'] for key,value in claims.items()})
    gates={}
    for cut in CANDIDATES:
        for endpoint,minimum in [('native_gain',.1),('random_advantage',.05)]:
            row=claims[f'{cut}/{endpoint}']
            row.update(p_holm=adjusted[f'{cut}/{endpoint}'],minimum_mean=minimum,
                passed=bool(row['value'] is not None and row['value']>.5 and adjusted[f'{cut}/{endpoint}']<.05 and row['mean']>=minimum))
        recovery=None if gap<.1 else float(np.mean(native-anchored_loss[cut])/gap)
        gates[str(cut)]=dict(suffix_energy=float(np.mean(suffix_energy[cut])),ceiling_gap=gap,recovery=recovery,
            passed=bool(all(claims[f'{cut}/{endpoint}']['passed'] for endpoint in ('native_gain','random_advantage')) and
                np.mean(suffix_energy[cut])<=.05 and recovery is not None and recovery>=.75))
    eligible=[cut for cut in CANDIDATES if gates[str(cut)]['passed']]
    return dict(claims=claims,gates=gates,earliest_qualifying_cut=eligible[0] if eligible else None,
                family_size=len(claims),scope='Exact conditional majority plus empirical mean gates; not mean-superiority inference')


def pilot_forecasts(model,truth,metadata,bundle):
    """Evaluate the frozen interventions only after numerical checks pass on all seeds."""
    n=len(metadata)//4
    native=bundle['prediction'][np.arange(n)*4]
    target=truth[np.arange(n)*4]
    scale=bundle['common']
    energy=scale**2
    mse=lambda value:np.mean((value-target)**2,axis=1)/energy
    natural=mse(native);component=mse(bundle['component_sum'])
    anchored_loss={};random_loss={};suffix_energy={};descriptive={};arrays={};identifiable={}
    for cut in CUTS:
        edited=bundle['outputs'][cut]['anchored']
        h=bundle['outputs'][cut]['receiver']
        candidate=bundle['outputs'][cut]['state']
        edits,degenerate,geometry=random_edits(h,candidate-h,np.random.default_rng(3600173+cut),2,return_diagnostics=True)
        identifiable[cut]=~degenerate
        physical=lambda state:suffix(model,state,cut)*bundle['scales']['baseline'][:,None]+bundle['means']['baseline'][:,None]
        random_forecasts=[physical(h+edit) for edit in edits]
        anchored_loss[cut]=mse(edited)
        random_loss[cut]=np.mean([mse(value) for value in random_forecasts],axis=0)
        suffix_energy[cut]=np.mean(bundle['nonlinear'][cut]**2,axis=1)/energy
        ix_a=np.arange(n)*4+1;ix_b=np.arange(n)*4+2;ix_0=np.arange(n)*4+3
        unweighted=bundle['states'][cut][ix_a]+bundle['states'][cut][ix_b]-bundle['states'][cut][ix_0]
        descriptive[str(cut)]=dict(anchor_loss=anchored_loss[cut].tolist(),random_mean_loss=random_loss[cut].tolist(),
            suffix_energy=suffix_energy[cut].tolist(),sign_reverse_loss=mse(physical(2*h-candidate)).tolist(),
            unweighted_loss=mse(physical(unweighted)).tolist(),random_control_unidentifiable=degenerate.tolist(),random_geometry=geometry,
            role='late descriptive control' if cut==20 else 'registered candidate')
        arrays[f'cut{cut}_anchored']=edited
        arrays[f'cut{cut}_B']=bundle['B'][cut]
        arrays[f'cut{cut}_suffix_remainder']=bundle['nonlinear'][cut]
        for index,value in enumerate(random_forecasts):arrays[f'cut{cut}_random{index}']=value
    arrays.update(native=native,actual_component_sum=bundle['component_sum'],truth=target,common_scale=scale)
    statistical_inputs=dict(native=natural,component=component,anchored_loss=anchored_loss,random_loss=random_loss,suffix_energy=suffix_energy,random_identifiable=identifiable)
    return dict(descriptive=descriptive,native_loss=natural.tolist(),component_sum_loss=component.tolist(),
                ceiling='Actual native physical forecast sum; late head-state composition is not an exact ceiling',
                interpretation='Earliest tested suffix compatibility only; invertible nonlinear encodings can change the curve without native errors'),arrays,statistical_inputs


def effective_cut(result):
    """A resource/numerical failure overrides any conditional scientific selection."""
    if not result['numerical_checks']['passed'] or result.get('resource_limit_exceeded',False):return None
    return result.get('pilot',{}).get('selection',{}).get('earliest_qualifying_cut')


def memory_snapshot():
    """Record current RSS and process high water, never resetting either measure."""
    rows=Path('/proc/self/status').read_text(encoding='utf-8').splitlines()
    return {key:int(next(row.split()[1] for row in rows if row.startswith(key+':')))*1024 for key in ('VmRSS','VmHWM')}


def reanalyze_saved(source,out):
    """Recompute original statistics from immutable saved waveforms without a model."""
    original=json.loads((source/'results.json').read_text(encoding='utf-8'))
    declared=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if replay.digest(source/'corpus.npz')!=declared['corpus_sha256'] or replay.digest(source/'metadata.json')!=declared['metadata_sha256']:
        raise ValueError('Saved source corpus/metadata binding changed')
    with np.load(source/'forecast_waveforms.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    with np.load(source/'corpus.npz',allow_pickle=False) as corpus:
        np.testing.assert_array_equal(arrays['truth'],corpus['clean_future'][::4])
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    if [row['generator_seed'] for row in metadata[::4]]!=list(range(3600000,3600016)):
        raise ValueError('Saved seed membership/order differs from the frozen cohort')
    energy=arrays['common_scale']**2
    mse=lambda value:np.mean((value-arrays['truth'])**2,axis=1)/energy
    native=mse(arrays['native']);component=mse(arrays['actual_component_sum'])
    np.testing.assert_array_equal(native,np.asarray(original['pilot']['native_loss']))
    np.testing.assert_array_equal(component,np.asarray(original['pilot']['component_sum_loss']))
    anchored_loss={};random_loss={};suffix_energy={};identifiable={}
    for cut in CUTS:
        desc=original['pilot']['descriptive'][str(cut)]
        anchored_loss[cut]=mse(arrays[f'cut{cut}_anchored'])
        random_loss[cut]=np.mean([mse(arrays[f'cut{cut}_random{r}']) for r in range(2)],axis=0)
        suffix_energy[cut]=np.mean(arrays[f'cut{cut}_suffix_remainder']**2,axis=1)/energy
        identifiable[cut]=~np.asarray(desc['random_control_unidentifiable'])
        for key,value in [('anchor_loss',anchored_loss[cut]),('random_mean_loss',random_loss[cut]),('suffix_energy',suffix_energy[cut])]:
            np.testing.assert_array_equal(value,np.asarray(desc[key]))
    selected=qualify(native,component,anchored_loss,random_loss,suffix_energy,identifiable)
    if selected!=original['pilot']['selection']:
        raise AssertionError('Saved forecast reanalysis changed original claims/gates')
    out.mkdir(parents=True,exist_ok=False)
    result=dict(exact_all8_claims_and_gates=True,conditional_selection=selected,
        effective_earliest_cut=None if not original['numerical_checks']['passed'] or original.get('resource_limit_exceeded',False) else selected['earliest_qualifying_cut'],
        original_resource_failure_preserved=bool(original.get('resource_limit_exceeded',False)),
        original_cut10_unidentifiable_seed_indices=selected['claims']['10/random_advantage'].get('unidentifiable_seed_indices'),
        source_bindings={name:replay.digest(source/name) for name in ['protocol.json','results.json','corpus.npz','metadata.json','forecast_waveforms.npz']},
        scope='Execution verification of the same inspected16, not independent scientific replication',memory=memory_snapshot())
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def run(root,out,mode,calibration=None,equivalence_source=None):
    """Calibration suppresses loss/gain selection; fresh mode remains separately gated."""
    started=time.monotonic()
    memory={'start':memory_snapshot()}
    out.mkdir(parents=True,exist_ok=False)
    config=protocol(mode)
    _,_,_,checkpoint,manifest=replay.manifest(root,'calibrate')
    config.update(driver_sha256=replay.digest(__file__),helper_sha256=replay.digest(replay.__file__),checkpoint=manifest['checkpoint'],
                  checkpoint_weight_sha256=manifest['checkpoint_weight_sha256'],source_sha256=manifest['source_sha256'],available_memory_kib=manifest['available_memory_kib'])
    if mode=='pilot':
        if calibration is None:raise ValueError('Fresh pilot requires a passing saved calibration')
        previous=json.loads((calibration/'results.json').read_text(encoding='utf-8'))
        old_protocol=json.loads((calibration/'protocol.json').read_text(encoding='utf-8'))
        if not previous['numerical_checks']['passed'] or old_protocol['driver_sha256']!=config['driver_sha256'] or old_protocol['checkpoint_weight_sha256']!=config['checkpoint_weight_sha256'] or old_protocol['helper_sha256']!=config['helper_sha256'] or old_protocol['source_sha256']!=config['source_sha256']:
            raise ValueError('Calibration did not pass or frozen source/checkpoint changed')
        config['calibration_source']=str(calibration)
        config['calibration_protocol_sha256']=replay.digest(calibration/'protocol.json')
        config['calibration_results_sha256']=replay.digest(calibration/'results.json')
    if equivalence_source is not None:
        if mode!='pilot':raise ValueError('Equivalence source applies only to the same16 pilot execution')
        config['execution_equivalence_source']=str(equivalence_source)
        config['execution_equivalence_bindings']={name:replay.digest(equivalence_source/name) for name in ['protocol.json','results.json','corpus.npz','metadata.json','forecast_waveforms.npz']}
        config['execution_scope']='Same16 resource correction, not independent scientific replication'
    contexts,truth,metadata=make_data(config['calibration_ids'] if mode=='calibrate' else config['pilot_ids'])
    if equivalence_source is not None:
        with np.load(equivalence_source/'corpus.npz',allow_pickle=False) as previous_input:
            np.testing.assert_array_equal(contexts,previous_input['context'])
            np.testing.assert_array_equal(truth,previous_input['clean_future'])
        if metadata!=json.loads((equivalence_source/'metadata.json').read_text(encoding='utf-8')):raise AssertionError('Same16 metadata changed')
    np.savez_compressed(out/'corpus.npz',context=contexts,clean_future=truth)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    config.update(corpus_sha256=replay.digest(out/'corpus.npz'),metadata_sha256=replay.digest(out/'metadata.json'))
    (out/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    memory['before_model_load']=memory_snapshot()
    model=replay.load_model(checkpoint)
    memory['after_model_load']=memory_snapshot()
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1.5*1024**3:raise RuntimeError('Resident limit exceeded after checkpoint load')
    checks,bundle=calibrated_run(model,contexts,metadata)
    memory['after_numerical_model_phase']=memory_snapshot()
    checks['resource_passed']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=1.5*1024**3
    checks['passed']=bool(checks['passed'] and checks['resource_passed'])
    result=dict(numerical_checks=checks,wall_seconds=time.monotonic()-started,
                peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                decision='ready for separate fresh-pilot authorization' if checks['passed'] else 'STOP: numerical ambiguity; no fresh pilot or anchor replacement',
                scope='Calibration only; no scientific loss/gain selection reported')
    statistical_inputs=None
    equivalent_arrays=None
    if mode=='pilot' and checks['passed']:
        scientific,arrays,statistical_inputs=pilot_forecasts(model,truth,metadata,bundle)
        result['pilot']=scientific
        result['scope']='Frozen fresh-seed discovery screen, no automatic replication or layer-blame claim'
        np.savez_compressed(out/'forecast_waveforms.npz',**arrays)
        if equivalence_source is not None:
            with np.load(equivalence_source/'forecast_waveforms.npz',allow_pickle=False) as previous_arrays:
                if set(arrays)!=set(previous_arrays.files):raise AssertionError('Saved forecast array family changed')
                for key,value in arrays.items():
                    if value.dtype!=previous_arrays[key].dtype or value.shape!=previous_arrays[key].shape or value.tobytes()!=previous_arrays[key].tobytes():raise AssertionError(f'Forecast array changed: {key}')
            equivalent_arrays=dict(all_arrays_bitwise_equal=True,array_count=len(arrays))
        memory['after_forecast_model_phase']=memory_snapshot()
    del model,bundle
    gc.collect()
    memory['after_model_and_states_released']=memory_snapshot()
    if statistical_inputs is not None:
        result['pilot']['selection']=qualify(**statistical_inputs)
        memory['after_statistics']=memory_snapshot()
        if equivalence_source is not None:
            previous_science=json.loads((equivalence_source/'results.json').read_text(encoding='utf-8'))['pilot']
            if result['pilot']!=previous_science:raise AssertionError('Scientific losses, geometry flags, claims or gates changed')
            result['execution_equivalence']=dict(**equivalent_arrays,all_scientific_fields_exact=True,source=str(equivalence_source))
            result['scope']='Same16 execution validation only; not independent scientific replication'
    result['wall_seconds']=time.monotonic()-started
    result['peak_resident_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if result['peak_resident_bytes']>1.5*1024**3:
        result['resource_limit_exceeded']=True
        result['numerical_checks']['passed']=False
        result['decision']='STOP: resident limit exceeded; no qualification'
    result['effective_earliest_cut']=effective_cut(result)
    if mode=='pilot' and not result.get('resource_limit_exceeded',False) and result['numerical_checks']['passed']:
        result['decision']='Execution complete; any effective candidate requires separate fresh replication authorization'
    result['memory_boundaries']=memory
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['calibrate','pilot','reanalyze-saved'],required=True)
    parser.add_argument('--equivalence-source-run',type=Path)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='reanalyze-saved':
        if args.source_run is None:parser.error('Saved reanalysis requires --source-run')
        print(json.dumps(reanalyze_saved(args.source_run,args.out),indent=2))
    else:run(args.repo_root,args.out,args.mode,args.calibration,args.equivalence_source_run)
