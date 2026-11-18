"""Frozen cut15 replication: serial16-seed workers, then model-free Holm2.

Top-level imports are deliberately stdlib-only: the controller must not retain
scientific/model imports alongside a worker. No shard effect-size selection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import numbers
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

SEEDS=tuple(range(3700000,3700064))
SHARDS=tuple(SEEDS[i:i+16] for i in range(0,64,16))
CUTS=(15,20)
CAP=int(1.5*2**30)
POINT_GATES=dict(native_gain=.1,random_advantage=.05,suffix_energy_max=.05,component_reference_gap_min=.1,recovery_min=.75)
NUMERICAL_GATES=dict(physical_transport=1e-5,normalized_transport=1e-5,B_invariance=1e-5,identity_closure=1e-5,
    raw_state=1e-5,raw_waveform=1e-5,self_state_max_abs=1e-8,canonical_anchor_sensitivity=.01)
RANDOM_KEY='SeedSequence([3700173,generator_seed,15]);unchanged random_edits count2 perseed'
RANDOM_SCOPE='ONLY primary15 has2random/reverse/unweighted arms;20hasanchored+source/numerical/anchor checks,no random gate'
RESOURCES=dict(cpu_threads=1,CUDA_VISIBLE_DEVICES='',offline=True,resident_limit_bytes=CAP,worker_timeout_seconds=180,total_timeout_seconds=180)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def memory(pid=None):
    rows=Path(f'/proc/{pid or "self"}/status').read_text(encoding='utf-8').splitlines()
    return {key:int(next(r.split()[1] for r in rows if r.startswith(key+':')))*1024 for key in ('VmRSS','VmHWM')}


def keyed_random(receiver,edit,seed,cut):
    """One seed/cut key produces the unchanged helper's TWO ordered draws."""
    import numpy as np
    from experiments.feature_search import composition_restoration as source
    if cut not in CUTS:raise ValueError('Only frozen cut15 and control20')
    rng=np.random.default_rng(np.random.SeedSequence([3700173,int(seed),int(cut)]))
    return source.random_edits(receiver[None],edit[None],rng,2,return_diagnostics=True)


def exact_ids(seed_ids,expected):
    try:return len(seed_ids)==len(expected) and all(isinstance(x,numbers.Integral) and not isinstance(x,bool) for x in seed_ids) and tuple(seed_ids)==tuple(expected)
    except TypeError:return False


def validate_membership(seed_ids):return exact_ids(seed_ids,SEEDS)


def declarations_match(p):
    expected=dict(seed_ids=list(SEEDS),candidate=15,cuts=list(CUTS),late_control=20,forbidden_candidate_search=[0,5,10],
        periods=[32,16],context=128,horizon=64,conditions=['baseline','pure_phi','pure_psi','background'],family_size=2,
        random_count=2,random_seed_key=RANDOM_KEY,random_scope=RANDOM_SCOPE,random_geometry_FP64=1e-8,
        random_geometry_injected_FP32=1e-5,geometry_reference_floor=1.,point_gates=POINT_GATES,numerical_gates=NUMERICAL_GATES,resources=RESOURCES)
    return all(p.get(key)==value for key,value in expected.items())


def shard_record_valid(record,index):
    return bool(type(index) is int and 0<=index<4 and record.get('valid',False)
        and type(record.get('shard_index')) is int and record['shard_index']==index and exact_ids(record.get('seed_ids',()),SHARDS[index])
        and record.get('numerical_checks',{}).get('passed',False) and record.get('primary15_random_geometry_pass',False)
        and record.get('model_released_before_return',False) and record.get('peak_resident_bytes',CAP+1)<=CAP
        and record.get('wall_seconds',181)<=180)


def resource_override(result,passed,reason):
    if passed:return result
    failed=dict(status='failed_incomplete',qualified=False,claims={},family_size=2,reason=reason,
        planned_n=64,no_partial_confirmatory_inference=True,
        peak_resident_bytes=result.get('peak_resident_bytes'),wall_seconds=result.get('wall_seconds'),
        protocol_sha256=result.get('protocol_sha256'))
    for key in ('total_wall_seconds','controller_peak_resident_bytes','maximum_simultaneous_task_RSS','model_free_analysis'):
        if key in result:failed[key]=result[key]
    return failed


def pooled_decision(arrays,validity):
    """No confirmatory tests unless the FULL64 validity/resource gate passed."""
    import numpy as np
    failed=dict(status='failed_incomplete',qualified=False,claims={},family_size=2,
                scope='Full64 planned gate failed/incomplete; no partial-sample confirmatory p-values')
    if not validity or len(validity)!=4 or not all(shard_record_valid(r,i) for i,r in enumerate(validity)):return failed
    if 'seed_ids' not in arrays or not validate_membership(arrays['seed_ids']):return failed
    required=('native','actual_component_sum','truth','common_scale','cut15_anchored','cut15_random0','cut15_random1','cut15_suffix_remainder')
    if any(k not in arrays for k in required):return failed
    if any(np.shape(arrays[k])!=((64,) if k=='common_scale' else (64,64)) or not np.isfinite(arrays[k]).all() for k in required):return failed
    if np.any(arrays['common_scale']<=0):return failed
    from tsfm_lens.analysis.feature_interference import sign_comparison
    from tsfm_lens.analysis.stats import holm
    scale=arrays['common_scale']**2
    mse=lambda y:np.mean((y-arrays['truth'])**2,axis=1)/scale
    native=mse(arrays['native']);component=mse(arrays['actual_component_sum']);edited=mse(arrays['cut15_anchored'])
    random=(mse(arrays['cut15_random0'])+mse(arrays['cut15_random1']))/2
    vectors=dict(native_gain=native-edited,random_advantage=random-edited)
    claims={key:sign_comparison(value) for key,value in vectors.items()}
    adjusted=holm({key:value['p'] for key,value in claims.items()})
    for key,minimum in [('native_gain',.1),('random_advantage',.05)]:
        value=claims[key];value.update(mean=float(vectors[key].mean()),p_holm=adjusted[key],minimum_mean=minimum)
        value['passed']=bool(value['value'] is not None and value['value']>.5 and adjusted[key]<.05 and value['mean']>=minimum)
    gap=float(np.mean(native-component));recovery=None if gap<.1 else float(np.mean(native-edited)/gap)
    suffix=float(np.mean(np.mean(arrays['cut15_suffix_remainder']**2,axis=1)/scale))
    gates=dict(reference_gap=gap,recovery_unclipped=recovery,mean_suffix_energy=suffix,
        passed=bool(all(r['passed'] for r in claims.values()) and suffix<=.05 and recovery is not None and recovery>=.75))
    return dict(status='complete',qualified=gates['passed'],candidate_cut=15,late_control_cut=20,
        claims=claims,family_size=2,gates=gates,
        mean_losses=dict(native=float(native.mean()),component_reference=float(component.mean()),anchored=float(edited.mean()),random_mean=float(random.mean())),
        per_seed=dict(seed_ids=arrays['seed_ids'].tolist(),native_loss=native.tolist(),component_loss=component.tolist(),edited_loss=edited.tolist(),random_mean_loss=random.tolist(),**{k:v.tolist() for k,v in vectors.items()}),
        scope='Exact two-sided conditional-majority Holm2 plus empirical point gates; not population mean-superiority; component sum is a reference,not a bound')


@contextmanager
def partial_capture(model,cache):
    """Only four-token states15/20; no tokenizer/cut0 state cache."""
    handles=[]
    handles.append(model.tokenizer.register_forward_pre_hook(lambda _,args:cache.setdefault('raw',[]).append(args[0][...,:32].detach().clone())))
    def head(_,args,out):
        cache.setdefault('head',[]).append(out.reshape(len(out),4,128,10)[:,-1,:64,model.aridx].detach().clone())
    handles.append(model.output_projection_point.register_forward_hook(head))
    for cut in CUTS:
        def block(_,args,out,cut=cut):cache.setdefault(cut,[]).append(out[0].detach().clone())
        handles.append(model.stacked_xf[cut-1].register_forward_hook(block))
    try:yield
    finally:
        for handle in handles:handle.remove()


def capture(model,contexts):
    import numpy as np
    import torch
    cache={};pred=[]
    for start in range(0,len(contexts),8):
        x=torch.as_tensor(contexts[start:start+8],dtype=torch.float32)
        with torch.inference_mode(),partial_capture(model,cache):
            output,_,extra=model.decode(64,x,torch.zeros_like(x,dtype=torch.bool))
        if extra is not None and extra.numel():raise ValueError('H64 must not autoregress')
        pred.append(output[:,-1,:64,model.aridx].float().numpy())
    state={key:torch.cat(value).numpy() for key,value in cache.items()}
    if set(state)!={15,20,'raw','head'} or any(state[c].shape!=(len(contexts),4,1280) for c in CUTS):
        raise ValueError('Unexpected selected full-token state geometry')
    return np.concatenate(pred).astype(float),state


def native_stage(model,context,truth,metadata):
    """Return waveforms/validity only; no truth-loss or effect-size computation."""
    import numpy as np
    import torch
    from experiments.feature_search import composition_restoration as source
    pred,states=capture(model,context);mu,sigma=source.normalizers(context)
    n=len(metadata)//4;ix={c:np.arange(n)*4+i for i,c in enumerate(source.CONDITIONS)}
    means={c:mu[index].astype(float) for c,index in ix.items()};scales={c:sigma[index].astype(float) for c,index in ix.items()}
    common=np.asarray([np.sqrt((metadata[i*4]['a']**2+metadata[i*4]['b']**2)/2) for i in range(n)])
    if any(np.any(scales[c]<=0) for c in ('baseline','pure_phi','pure_psi')):raise ValueError('Native mixture/pure scale must be positive')
    zero_cache={}
    with torch.inference_mode(),partial_capture(model,zero_cache):model(torch.zeros(n,4,32),torch.zeros(n,4,32,dtype=torch.bool))
    zeros={c:zero_cache[c][0].numpy() for c in CUTS};del zero_cache
    checks=dict(transport={},normalized_transport={},anchor_sensitivity={},B_invariance={},identity_closure={},self_patch_state={})
    arrays=dict(native=pred[ix['baseline']],actual_component_sum=pred[ix['pure_phi']]+pred[ix['pure_psi']]-pred[ix['background']],
        truth=truth[ix['baseline']],common_scale=common,native_all_conditions=pred)
    B_reference=source.affine_remainder(scales,means,states['head'][ix['background']].astype(float))
    geometry={};rng_hashes={};all_geometry=True;pending={}
    for cut in CUTS:
        branch={c:states[cut][index] for c,index in ix.items()};transport={c:source.suffix(model,value,cut) for c,value in branch.items()}
        for condition in source.CONDITIONS:
            reference=states['head'][ix[condition]].astype(float)
            checks['normalized_transport'][f'{cut}/{condition}']=float(np.sqrt(np.mean((transport[condition]-reference)**2,axis=1)).max())
            physical=means[condition][:,None]+scales[condition][:,None]*transport[condition]
            checks['transport'][f'{cut}/{condition}']=float(np.sqrt(np.mean(((physical-pred[ix[condition]])/common[:,None])**2,axis=1)).max())
        h=branch['baseline'];candidate=source.anchored(h,branch['pure_phi'],branch['pure_psi'],branch['background'],scales['baseline'],scales['pure_phi'],scales['pure_psi'])
        physical=lambda value:source.suffix(model,value,cut)*scales['baseline'][:,None]+means['baseline'][:,None]
        edited=physical(candidate)
        alternate=source.anchored(h,branch['pure_phi'],branch['pure_psi'],zeros[cut],scales['baseline'],scales['pure_phi'],scales['pure_psi'])
        alternate_out=physical(alternate)
        checks['anchor_sensitivity'][str(cut)]=np.sqrt(np.mean(((edited-alternate_out)/common[:,None])**2,axis=1)).tolist()
        B=source.affine_remainder(scales,means,transport['background'])
        checks['B_invariance'][str(cut)]=float(np.sqrt(np.mean(((B-B_reference)/common[:,None])**2,axis=1)).max())
        a=scales['pure_phi']/scales['baseline'];b=scales['pure_psi']/scales['baseline']
        nonlinear=scales['baseline'][:,None]*((edited-means['baseline'][:,None])/scales['baseline'][:,None]-a[:,None]*transport['pure_phi']-b[:,None]*transport['pure_psi']-(1-a-b)[:,None]*transport['background'])
        discrepancy=edited-arrays['actual_component_sum']-B-nonlinear
        checks['identity_closure'][str(cut)]=float(np.sqrt(np.mean((discrepancy/common[:,None])**2,axis=1)).max())
        identity=source.anchored(h,h,branch['background'],branch['background'],scales['baseline'],scales['baseline'],scales['background'])
        checks['self_patch_state'][str(cut)]=float(np.max(abs(identity-h)))
        pending[cut]=(h,candidate)
        arrays[f'cut{cut}_anchored']=edited;arrays[f'cut{cut}_B']=B;arrays[f'cut{cut}_suffix_remainder']=nonlinear
    checks['B_invariance']['15_vs20']=float(np.sqrt(np.mean(((arrays['cut15_B']-arrays['cut20_B'])/common[:,None])**2,axis=1)).max())
    raw={c:states['raw'][index] for c,index in ix.items()}
    raw_edit=source.anchored(raw['baseline'],raw['pure_phi'],raw['pure_psi'],raw['background'],scales['baseline'],scales['pure_phi'],scales['pure_psi'])
    checks['raw_input_state_RMS']=np.sqrt(np.mean((raw_edit-raw['baseline'])**2,axis=(1,2))).tolist()
    with torch.inference_mode():raw_hidden=model.tokenizer(torch.cat([torch.as_tensor(raw_edit,dtype=torch.float32),torch.zeros_like(torch.as_tensor(raw_edit,dtype=torch.float32))],dim=-1))
    raw_pred=source.suffix(model,raw_hidden,0)*scales['baseline'][:,None]+means['baseline'][:,None]
    checks['raw_input_waveform_RMS']=np.sqrt(np.mean(((raw_pred-arrays['native'])/common[:,None])**2,axis=1)).tolist()
    arrays['raw_input_control']=raw_pred
    checks['maximum_anchor_sensitivity']=max(max(values) for values in checks['anchor_sensitivity'].values())
    checks['passed']=source.numerical_gate(checks)
    if not checks['passed']:return checks,arrays,geometry,rng_hashes,False
    for cut in (15,):
        h,candidate=pending[cut]
        control_states=[np.empty_like(candidate) for _ in range(2)];geometry[str(cut)]=[];rng_hashes[str(cut)]=[]
        for index in range(n):
            seed=metadata[index*4]['generator_seed']
            edits,flags,diag=keyed_random(h[index],candidate[index]-h[index],seed,cut)
            geometry[str(cut)].append(dict(seed_id=seed,unidentifiable=bool(flags[0]),diagnostics=diag[0]))
            all_geometry=all_geometry and not flags[0]
            hashes=[]
            for draw in range(2):
                control_states[draw][index]=h[index]+edits[draw,0]
                hashes.append(hashlib.sha256(edits[draw,0].tobytes()).hexdigest())
            rng_hashes[str(cut)].append(dict(seed_id=seed,key=[3700173,seed,cut],draw_sha256=hashes))
        if not all_geometry:return checks,arrays,geometry,rng_hashes,False
        for draw in range(2):
            arrays[f'cut{cut}_random{draw}']=source.suffix(model,control_states[draw],cut)*scales['baseline'][:,None]+means['baseline'][:,None]
        arrays['cut15_reverse']=source.suffix(model,2*h-candidate,15)*scales['baseline'][:,None]+means['baseline'][:,None]
        unweighted=states[15][ix['pure_phi']]+states[15][ix['pure_psi']]-states[15][ix['background']]
        arrays['cut15_unweighted']=source.suffix(model,unweighted,15)*scales['baseline'][:,None]+means['baseline'][:,None]
    return checks,arrays,geometry,rng_hashes,bool(all_geometry)


def prepare(root,corrected,original,out):
    """Prospective corpus/bindings only: no model constructor or native forward."""
    import numpy as np
    from experiments.feature_search import composition_restoration as source
    from experiments.feature_search import homogeneous_normalizer as helper
    if out.exists():raise ValueError('Use fresh prospective output')
    good=read(corrected/'results.json');bad=read(original/'results.json');declared=read(corrected/'protocol.json')
    if good.get('effective_earliest_cut')!=15 or not good['numerical_checks']['passed'] or good['peak_resident_bytes']>CAP:
        raise ValueError('Corrected discovery did not qualify cut15')
    if not bad.get('resource_limit_exceeded') or good['pilot']!=bad['pilot']:
        raise ValueError('Retain failed original and exact scientific equivalence')
    with np.load(corrected/'forecast_waveforms.npz',allow_pickle=False) as a,np.load(original/'forecast_waveforms.npz',allow_pickle=False) as b:
        if set(a.files)!=set(b.files) or len(a.files)!=29 or any(not np.array_equal(a[k],b[k]) for k in a.files):
            raise ValueError('Corrected same16 waveforms differ')
    if declared['driver_sha256']!=digest(source.__file__) or declared['helper_sha256']!=digest(helper.__file__):
        raise ValueError('Validated scientific helpers changed')
    if digest(declared['checkpoint'])!=declared['checkpoint_weight_sha256']:raise ValueError('Weights changed')
    context,truth,metadata=source.make_data(SEEDS)
    row_ids=np.asarray([f"{m['generator_seed']}:{m['condition']}" for m in metadata])
    out.mkdir(parents=True)
    np.savez_compressed(out/'corpus.npz',context=context,clean_future=truth,row_ids=row_ids)
    write(out/'metadata.json',metadata)
    shards=[]
    for index,seed_ids in enumerate(SHARDS):
        sl=slice(index*64,(index+1)*64);name=f'shard_{index}'
        np.savez_compressed(out/f'{name}_corpus.npz',context=context[sl],clean_future=truth[sl],row_ids=row_ids[sl])
        write(out/f'{name}_metadata.json',metadata[sl])
        shards.append(dict(shard_index=index,seed_ids=list(seed_ids),corpus=f'{name}_corpus.npz',metadata=f'{name}_metadata.json',
            corpus_sha256=digest(out/f'{name}_corpus.npz'),metadata_sha256=digest(out/f'{name}_metadata.json'),row_ids=row_ids[sl].tolist()))
    protocol=dict(stage='49.16',design='Fresh64 selected-cut replication; no native output at preparation',
        seed_ids=SEEDS,shards=shards,candidate=15,cuts=CUTS,late_control=20,forbidden_candidate_search=[0,5,10],
        cut_meaning='After15of20 transformer blocks/index14;20isheadINPUT,notoutput',
        periods=[32,16],context=128,horizon=64,conditions=source.CONDITIONS,
        family_size=2,random_count=2,random_seed_key=RANDOM_KEY,random_scope=RANDOM_SCOPE,
        random_geometry_FP64=1e-8,random_geometry_injected_FP32=1e-5,geometry_reference_floor=1.,
        geometry_interpretation='max(1,absolute reference) is numerical tolerance,not a scientific loss denominator',
        primary_family='ONE pooled Holm2,ALL64 seeds:nativegain and mean2randomadvantage;two-sided exact conditional signs,positive majority',
        point_gates=POINT_GATES,
        recovery='mean(native-edited)/mean(native-component_reference),UNCLIPPED;reference is not a bound;gap<.1 => no pass',
        numerical_gates=NUMERICAL_GATES,
        B_reference='Actual native normalized background head plus direct15vs20 comparison;no cut0 storage',
        serial_execution='Four serial16-seed workers;release model/states andexit before nextworker. Model-free pooled stats only afterall4valid.',
        fail_fast='Any required numerical/primary15geometry/resource/membership failure stops remaining shards;full64failed/incomplete;no partial confirmatory p-values or shard effects inspected.',
        units='ONE generatingseed;all losses/suffix energy use SAME theoretical(a²+b²)/2 perseed;no MASE/no variance floors',
        no_rescue='20/raw controls cannot select/rescue;no cohort enlargement/replacement or additional random draws.',
        resources=RESOURCES,
        memory_scope='All process HWM retained;lightstdlib controller+oneworker simultaneousRSS and conservative sumofHWM <=1.5GiB;models absent beforestats.',
        checkpoint=declared['checkpoint'],checkpoint_weight_sha256=declared['checkpoint_weight_sha256'],source_sha256=declared['source_sha256'],
        helper_source=str(Path(source.__file__).resolve()),helper_sha256=digest(source.__file__),
        load_helper_source=str(Path(helper.__file__).resolve()),load_helper_sha256=digest(helper.__file__),runner_sha256=digest(__file__),
        corrected_discovery=str(corrected),failed_original=str(original),
        discovery_bindings={f'{label}/{name}':digest(path/name) for label,path in [('corrected',corrected),('failed_original',original)] for name in ('results.json','protocol.json','forecast_waveforms.npz')},
        corpus_sha256=digest(out/'corpus.npz'),metadata_sha256=digest(out/'metadata.json'),
        interpretation='Background-anchored off-manifold suffix compatibility,not canonical feature identity/harmful layer origin;no population mean superiority from signs.',
        public_report='No positive report update before separatelyreviewed native replication qualifies')
    write(out/'protocol.json',protocol)


def verify_protocol(out):
    p=read(out/'protocol.json')
    if p['runner_sha256']!=digest(__file__) or p['helper_sha256']!=digest(p['helper_source']) or p['load_helper_sha256']!=digest(p['load_helper_source']):
        raise ValueError('Prospective source binding changed')
    if not declarations_match(p) or len(p['shards'])!=4 or any(r.get('shard_index')!=i or not exact_ids(r.get('seed_ids',()),SHARDS[i]) for i,r in enumerate(p['shards'])):
        raise ValueError('Frozen candidate/seed plan changed')
    if digest(out/'corpus.npz')!=p['corpus_sha256'] or digest(out/'metadata.json')!=p['metadata_sha256']:raise ValueError('Global corpus changed')
    for label,path in [('corrected',Path(p['corrected_discovery'])),('failed_original',Path(p['failed_original']))]:
        for name in ('results.json','protocol.json','forecast_waveforms.npz'):
            if digest(path/name)!=p['discovery_bindings'][f'{label}/{name}']:raise ValueError('Discovery anchor changed')
    return p


def worker(out,index):
    """One native16 block; output validity only, never shard losses/signs."""
    if type(index) is not int or not 0<=index<4:raise ValueError('Shard index must be0..3')
    entered=time.monotonic()
    import importlib
    import numpy as np
    p=verify_protocol(out);plan=p['shards'][index]
    if tuple(plan['seed_ids'])!=SHARDS[index]:raise ValueError('Shard membership changed')
    result_path=out/f'shard_{index}_result.json'
    if result_path.exists():raise ValueError('Never overwrite/restart a completed or failed shard')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Model worker must hide CUDA')
    for name,value in p['source_sha256'].items():
        module=importlib.import_module(name)
        if digest(module.__file__)!=value:raise ValueError('Installed native source changed')
    from experiments.feature_search import homogeneous_normalizer as helper
    from experiments.feature_search import composition_restoration as source
    if digest(p['checkpoint'])!=p['checkpoint_weight_sha256']:raise ValueError('Checkpoint changed')
    for key in ('corpus','metadata'):
        if digest(out/plan[key])!=plan[f'{key}_sha256']:raise ValueError('Shard source changed')
    with np.load(out/plan['corpus'],allow_pickle=False) as f:context=f['context'].copy();truth=f['clean_future'].copy();row_ids=f['row_ids'].copy()
    metadata=read(out/plan['metadata'])
    if row_ids.tolist()!=plan['row_ids'] or [r['generator_seed'] for r in metadata[::4]]!=plan['seed_ids']:
        raise ValueError('Ordered shard row IDs changed')
    expected_context,expected_truth,expected_metadata=source.make_data(plan['seed_ids'])
    if metadata!=expected_metadata or not np.array_equal(context,expected_context) or not np.array_equal(truth,expected_truth):
        raise ValueError('Shard physical generator mismatch')
    import torch
    torch.set_num_threads(1);boundaries={'before_load':memory()};model=helper.load_model(Path(p['checkpoint']))
    boundaries['after_load']=memory()
    if boundaries['after_load']['VmHWM']>CAP:
        del model;gc.collect();write(result_path,dict(valid=False,shard_index=index,seed_ids=plan['seed_ids'],reason='Resource failure after model load',memory=boundaries));return
    checks,arrays,geometry,keys,geom_pass=native_stage(model,context,truth,metadata)
    boundaries['after_native']=memory();del model;gc.collect();boundaries['after_model_released']=memory()
    seed_ids=np.asarray(plan['seed_ids'],dtype=np.int64)
    waveform=out/f'shard_{index}_waveforms.npz'
    np.savez_compressed(waveform,**arrays,seed_ids=seed_ids,row_ids=row_ids,shard_index=np.asarray(index,dtype=np.int64),
        protocol_sha256=np.asarray(digest(out/'protocol.json')),corpus_sha256=np.asarray(plan['corpus_sha256']),metadata_sha256=np.asarray(plan['metadata_sha256']))
    elapsed=time.monotonic()-entered;peak=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,memory()['VmHWM'])
    valid=bool(checks['passed'] and geom_pass and peak<=CAP and elapsed<=180)
    write(result_path,dict(valid=valid,shard_index=index,seed_ids=plan['seed_ids'],numerical_checks=checks,
        primary15_random_geometry_pass=geom_pass,random_geometry=geometry,random_identity=keys,
        geometry_status='not_evaluated_numerical_failure' if not checks['passed'] else 'passed' if geom_pass else 'failed',
        peak_resident_bytes=peak,wall_seconds=elapsed,memory=boundaries,model_released_before_return=True,
        waveform_sha256=digest(waveform),protocol_sha256=digest(out/'protocol.json'),
        helper_sha256=p['helper_sha256'],checkpoint_weight_sha256=p['checkpoint_weight_sha256'],runner_sha256=p['runner_sha256'],
        scope='Validity/resource only;no shard effect-size or sign analysis'))
    print(json.dumps(dict(shard_index=index,valid=valid,peak_resident_bytes=peak,wall_seconds=elapsed)),flush=True)


def load_shard(out,index,p):
    """Exact IDs, source/protocol/waveform hashes; never infer identity by shape."""
    if type(index) is not int or not 0<=index<4:raise ValueError('Shard index must be0..3')
    import numpy as np
    plan=p['shards'][index];result=read(out/f'shard_{index}_result.json')
    if not shard_record_valid(result,index):
        raise ValueError('Invalid/missing planned shard')
    for key in ('corpus','metadata'):
        if digest(out/plan[key])!=plan[f'{key}_sha256']:raise ValueError('Current shard corpus/metadata changed')
    if result['protocol_sha256']!=digest(out/'protocol.json') or result['peak_resident_bytes']>CAP or not result['model_released_before_return']:
        raise ValueError('Shard protocol/resource/release failure')
    for key in ('helper_sha256','checkpoint_weight_sha256','runner_sha256'):
        if result.get(key)!=p[key]:raise ValueError('Shard helper/checkpoint/source binding failed')
    path=out/f'shard_{index}_waveforms.npz'
    if digest(path)!=result['waveform_sha256']:raise ValueError('Shard waveform changed')
    with np.load(path,allow_pickle=False) as file:
        for key,value in [('protocol_sha256',digest(out/'protocol.json')),('corpus_sha256',plan['corpus_sha256']),('metadata_sha256',plan['metadata_sha256'])]:
            if file[key].item()!=value:raise ValueError('Shard output source binding failed')
        if not exact_ids(file['seed_ids'],SHARDS[index]) or file['row_ids'].tolist()!=plan['row_ids'] or file['shard_index'].item()!=index:
            raise ValueError('Shard output ordered IDs changed')
        arrays={k:file[k].copy() for k in file.files if k not in ('row_ids','shard_index','protocol_sha256','corpus_sha256','metadata_sha256')}
    metadata=read(out/plan['metadata'])
    with np.load(out/plan['corpus'],allow_pickle=False) as corpus:
        if not np.array_equal(arrays['truth'],corpus['clean_future'][::4]):raise ValueError('Shard physical target mismatch')
    common=np.asarray([np.sqrt((m['a']**2+m['b']**2)/2) for m in metadata[::4]])
    if not np.array_equal(common,arrays['common_scale']):raise ValueError('Common theoretical scale changed')
    native=arrays['native_all_conditions']
    if not np.array_equal(native[::4],arrays['native']) or not np.array_equal(native[1::4]+native[2::4]-native[3::4],arrays['actual_component_sum']):
        raise ValueError('Actual native component/background reference changed')
    return arrays,result


def analyze(out):
    """Separate model-free process, after every model worker has exited."""
    import numpy as np
    entered=time.monotonic();p=verify_protocol(out);controller=read(out/'controller_validity.json')
    if not controller.get('all_four_valid'):
        write(out/'results.json',dict(status='failed_incomplete',qualified=False,claims={},family_size=2,reason='Controller validity failure'));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Analysis must hide CUDA')
    loaded=[load_shard(out,index,p) for index in range(4)]
    keys=set(loaded[0][0])
    if any(set(arrays)!=keys for arrays,_ in loaded):raise ValueError('Incomplete shard array schemas')
    arrays={key:np.concatenate([pair[0][key] for pair in loaded]) for key in keys}
    result=pooled_decision(arrays,[pair[1] for pair in loaded])
    result.update(protocol_sha256=digest(out/'protocol.json'),model_free_analysis=True,
        shard_output_hashes={str(i):digest(out/f'shard_{i}_waveforms.npz') for i in range(4)},
        old_resource_failure_retained=True,conditional_same16_discovery_not_replication=True)
    if result['status']=='complete':
        scale=arrays['common_scale']**2;truth=arrays['truth'];mse=lambda value:np.mean((value-truth)**2,axis=1)/scale
        result['descriptive_controls']={key:dict(mean_loss=float(mse(arrays[key]).mean()),losses=mse(arrays[key]).tolist(),role='Not a primary claim or selection arm')
            for key in ('cut20_anchored','raw_input_control','cut15_reverse','cut15_unweighted')}
    result.update(peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-entered)
    result=resource_override(result,result['peak_resident_bytes']<=CAP and result['wall_seconds']<=180,'Model-free analysis resource failure')
    write(out/'results.json',result)
    print(json.dumps(dict(status=result['status'],qualified=result['qualified'],model_free_analysis=True)),flush=True)


def controller_validity(records,controller_peak):
    """Decide validity only; no effect-size fields are read between shards."""
    return bool(len(records)==4 and all(shard_record_valid(record,i) and record.get('peak_resident_bytes',CAP+1)+controller_peak<=CAP
        for i,record in enumerate(records)))


def run(out):
    """Stdlib-only controller. Four serial workers, then a separate reducer."""
    p=verify_protocol(out)
    if (out/'results.json').exists() or any(out.glob('shard_*_result.json')):raise ValueError('Never restart/drop/replace a completed or failed planned cohort')
    started=time.monotonic();records=[];controller_peak=memory()['VmHWM'];maximum_simultaneous=0
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    def child(mode,index=None):
        nonlocal controller_peak,maximum_simultaneous
        args=[sys.executable,str(Path(__file__).resolve()),'--out',str(out),'--mode',mode]
        if index is not None:args+=['--shard',str(index)]
        with (out/f'{mode}_{index if index is not None else "pooled"}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(args,env=environment,stdout=log,stderr=subprocess.STDOUT)
            reason=None
            while process.poll() is None:
                controller_peak=max(controller_peak,memory()['VmHWM'])
                try:combined=memory()['VmRSS']+memory(process.pid)['VmRSS']
                except (FileNotFoundError,StopIteration):combined=0
                maximum_simultaneous=max(maximum_simultaneous,combined)
                if combined>CAP:reason='Simultaneous task RSS exceeded1.5GiB'
                if time.monotonic()-started>180:reason='Whole-study180second deadline'
                if reason:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    return False,reason
                time.sleep(.05)
        return process.returncode==0,None if process.returncode==0 else f'{mode} process exit{process.returncode}'
    for index in range(4):
        success,reason=child('worker',index)
        path=out/f'shard_{index}_result.json'
        record=read(path) if success and path.exists() else dict(valid=False,shard_index=index,seed_ids=list(SHARDS[index]),reason=reason or 'Missing shard result')
        records.append(record)
        valid=bool(shard_record_valid(record,index) and record.get('peak_resident_bytes',CAP+1)+controller_peak<=CAP)
        if not valid:
            write(out/'controller_validity.json',dict(all_four_valid=False,completed_shards=len(records),planned_seed_ids=SEEDS,
                validity=records,controller_peak_bytes=controller_peak,maximum_simultaneous_RSS=maximum_simultaneous,reason=reason or 'Shard validity/resource failure'))
            write(out/'results.json',dict(status='failed_incomplete',qualified=False,claims={},family_size=2,failed_shard=index,
                planned_n=64,evaluated_shards=len(records),no_partial_confirmatory_inference=True,reason=reason or 'Shard validity/resource failure'))
            print(json.dumps(dict(status='failed_incomplete',failed_shard=index)),flush=True);return
    all_valid=controller_validity(records,controller_peak)
    write(out/'controller_validity.json',dict(all_four_valid=all_valid,completed_shards=4,planned_seed_ids=SEEDS,validity=records,
        controller_peak_bytes=controller_peak,maximum_simultaneous_RSS=maximum_simultaneous,models_exited_before_analysis=True))
    if not all_valid:
        write(out/'results.json',dict(status='failed_incomplete',qualified=False,claims={},family_size=2,reason='Pooled validity incomplete'));return
    success,reason=child('analyze')
    if not success or not (out/'results.json').exists():
        write(out/'results.json',dict(status='failed_incomplete',qualified=False,claims={},family_size=2,reason=reason or 'Missing pooled result'));return
    result=read(out/'results.json');result['total_wall_seconds']=time.monotonic()-started
    result['controller_peak_resident_bytes']=controller_peak;result['maximum_simultaneous_task_RSS']=maximum_simultaneous
    result=resource_override(result,result.get('peak_resident_bytes',CAP+1)+controller_peak<=CAP and result['total_wall_seconds']<=180,'Authoritative total resource failure')
    write(out/'results.json',result)
    print(json.dumps(dict(status=result['status'],qualified=result['qualified'],total_wall_seconds=result['total_wall_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('prepare','run','worker','analyze'),required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--corrected',type=Path);parser.add_argument('--original',type=Path);parser.add_argument('--shard',type=int)
    a=parser.parse_args()
    if a.mode=='prepare':prepare(a.root,a.corrected,a.original,a.out)
    elif a.mode=='worker':worker(a.out,a.shard)
    elif a.mode=='analyze':analyze(a.out)
    else:run(a.out)
