"""Control-only fixed-B8 native neighbor diagnostic; no feature analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiments.feature_search import chronos2_batch_control_design as design
from experiments.feature_search.phase_orbit_native import digest


def prepare(design_dir,pilot,out):
    if out.exists():raise ValueError('Preserve existing controls')
    declared=json.loads((design_dir/'protocol.json').read_text(encoding='utf-8'))
    if declared['design_source_sha256']!=digest(design.__file__):raise ValueError('Design source changed')
    old=json.loads((pilot/'protocol.json').read_text(encoding='utf-8'))
    prospective=dict(design_dir=str(design_dir),design_protocol_sha256=digest(design_dir/'protocol.json'),
        runner_sha256=digest(__file__),checkpoint=old['models']['chronos2'],
        original_failed_result_sha256=digest(pilot/'results.json'),original_pilot=str(pilot),
        conditions='42B8 forwards perpair plus2 sharedgroup positivecontrol forwards;2pairs=88forwards=704contexts',
        shared_group_positive='allzero IDs,base versus FIRSTfixed neighborcase(target0);mustdetect nonzero targetresponse in eachpair',
        guards='repeat/default-vs-explicit unique IDs/fixed-target neighbor/permutation exact0 each;sharegroups positive separately;no singletonretest,no thresholdrelaxation',
        limitations='Source masks mathematicallyindependent is distinct from empirical exactnumeric guards;oldsingletonfailure never relabeled;no Fourier/orbit/feature analysis',
        resources=dict(gpu=3,cpu_threads=2,batch_size=8,cap_bytes=6*2**30,timeout_seconds=180))
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(prospective,indent=2),encoding='utf-8')


def run(out):
    import torch
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models.chronos2_adapter import Chronos2Adapter
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    dd=Path(p['design_dir']);d=json.loads((dd/'protocol.json').read_text(encoding='utf-8'))
    if p['runner_sha256']!=digest(__file__) or p['design_protocol_sha256']!=digest(dd/'protocol.json') or d['design_source_sha256']!=digest(design.__file__):
        raise ValueError('Frozen source/protocol changed')
    if (out/'results.json').exists():raise ValueError('Never overwrite completed control')
    for path,value in d['sources'].items():
        if digest(path)!=value:raise ValueError('Installed source changed')
    checkpoint=Path(p['checkpoint']['checkpoint'])
    if digest(checkpoint)!=p['checkpoint']['weight_sha256'] or digest(checkpoint.parent/'config.json')!=p['checkpoint']['config_sha256']:
        raise ValueError('Checkpoint changed')
    source=Path(d['corpus_source'])
    cases={}
    for stem,bindings in d['bindings'].items():
        for suffix,value in bindings.items():
            if digest(source/f'{stem}.{suffix}')!=value:raise ValueError('Corpus changed')
        with np.load(source/f'{stem}.npz') as data:cases[stem]=design.fixed_batches(data['context'])
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(1.,6*2**30/torch.cuda.get_device_properties(0).total_memory))
    started=time.monotonic()
    adapter=Chronos2Adapter(ModelConfig(name='chronos2',checkpoint=str(checkpoint.parent)),DataConfig(context_len=128,horizon=64),torch.device('cuda:0'),torch.float32)
    adapter.ensure_loaded();results={};arrays={};calls=0
    def predict(batch,groups=None):
        nonlocal calls
        if time.monotonic()-started>180:raise RuntimeError('Three-minute cap reached')
        if batch.shape!=(8,128) or not np.isfinite(batch).all():raise ValueError('Every call fixed finiteB8×128')
        handle=None;hits=[]
        if groups is not None:
            def inject(_,args,kwargs):
                hits.append(1);kwargs=dict(kwargs);kwargs['group_ids']=torch.as_tensor(groups,device=adapter.device,dtype=torch.long)
                return args,kwargs
            handle=adapter._inner.register_forward_pre_hook(inject,with_kwargs=True)
        try:
            with torch.inference_mode():value=adapter.predict(batch,64,[.5])['point']
        finally:
            if handle is not None:handle.remove()
        if groups is not None and len(hits)!=1:raise ValueError('Expected one native model/group injection call')
        if value.shape!=(8,64) or not np.isfinite(value).all():raise ValueError('Native output shape/nonfinite')
        calls+=1;return value
    for stem,(base,counterfactuals) in cases.items():
        baseline=predict(base);repeat=predict(base);explicit=predict(base,np.arange(8))
        record=dict(repeat_max_abs=float(abs(repeat-baseline).max()),default_vs_explicit_max_abs=float(abs(explicit-baseline).max()),neighbors=[],permutations=[])
        arrays[f'{stem}_baseline']=baseline
        for index,case in enumerate(counterfactuals):
            observed=predict(case['batch']);unique=predict(case['batch'],np.arange(8));target=case['target']
            record['neighbors'].append(dict(target=target,neighbor_bank=case['neighbor_bank'],
                target_max_abs=float(abs(observed[target]-baseline[target]).max()),default_vs_explicit_max_abs=float(abs(observed-unique).max())))
            arrays[f'{stem}_neighbor_{index}']=observed
        for shift in range(1,8):
            permutation=np.roll(np.arange(8),shift)
            observed=predict(base[permutation])[np.argsort(permutation)]
            record['permutations'].append(dict(shift=shift,max_abs=float(abs(observed-baseline).max())))
        shared=predict(base,np.zeros(8,dtype=int));shared_changed=predict(counterfactuals[0]['batch'],np.zeros(8,dtype=int))
        record['shared_group_positive_target0_max_abs']=float(abs(shared_changed[0]-shared[0]).max())
        record['fixed_neighbor_independence_pass']=bool(all(r['target_max_abs']==0 for r in record['neighbors']))
        record['default_explicit_pass']=bool(record['default_vs_explicit_max_abs']==0 and all(r['default_vs_explicit_max_abs']==0 for r in record['neighbors']))
        record['position_permutation_pass']=bool(all(r['max_abs']==0 for r in record['permutations']))
        record['repeat_pass']=record['repeat_max_abs']==0
        record['shared_positive_pass']=record['shared_group_positive_target0_max_abs']>0
        record['all_pass']=all(record[k] for k in ('fixed_neighbor_independence_pass','default_explicit_pass','position_permutation_pass','repeat_pass','shared_positive_pass'))
        results[stem]=record
    if calls!=88:raise ValueError('Frozen native call budget changed')
    result=dict(cases=results,all_new_controls_pass=bool(all(r['all_pass'] for r in results.values())),
        native_calls=calls,context_forecasts=calls*8,total_wall_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        original_control_failure_retained=True,scope='Measurement diagnosis only;no oldglobalpass,no singletonthreshold relaxation,no phasefeatureclaim')
    np.savez_compressed(out/'control_forecasts.npz',**arrays)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--design-dir',type=Path);parser.add_argument('--pilot',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:prepare(args.design_dir,args.pilot,args.out)
    else:run(args.out)
