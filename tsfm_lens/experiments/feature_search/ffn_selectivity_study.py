"""Direct native FFN perturbation selectivity, without an SAE or semantic claim.

Preparation is model-free. First/half FFN projection-input prehooks and residual
outputs share each pass. A per-background locked-normalizer context reuses its
reference pass, keeping the entire first-stage pilot at480 native cells/model.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import contextlib
import os
import resource
import struct
import sys
import sysconfig
import time
import json

import numpy as np

from experiments.feature_search.coherence_study import read, write, sha, load_bank
from experiments.feature_search.natural_validation_study import normalize_history, THREAD_VARS

MODELS = ('bolt', 'timesfm')
TYPES = ('sin24', 'sin19.2', 'steprecent32', 'ramprecent64', 'noise')
BUDGETS = (1, 4, 16)
SEED = 4961000
MEMORY, SECONDS, CELL_CAP = 3*2**30, 900, 500


def knowledge(out, phase, status, findings, next_decision):
    """Persist the precise selectivity scope and all failed instrumentation."""
    path = out/'knowledge.json'
    events = read(path).get('events', []) if path.exists() else []
    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), phase=phase, status=status,
        question='Do native post-activation FFN neuron axes support held-out perturbation selectivity at single/few-feature budgets beyond orthogonal rotations, PCA, residual coordinates and actual input?',
        design='16 fresh synthetic backgrounds8/8 plus8 reused real4/4 when past-only artifacts exist; five RMS.1 perturbation types with sign/phase variants; background-locked normalization; first/half FFN and residual captures',
        exact_findings=findings,
        controls=['FIT-only feature/PCA/readout fitting', 'Mean and RMS token-delta views separate',
            'Actual input perturbation baseline', 'Three fixed orthogonal rotations',
            'Exact normalizer and FFN self-replay; within-FFN reach', 'Background/seed-group bootstrap, never tokens or variants'],
        limitations=['Perturbation selectivity only, NOT general monosemanticity', 'No SAE or sparse dictionary training',
            'Reused real sources support no new real-generalization claim', 'Structured orthogonal rotations are not Haar samples',
            'PCA control is limited to FIT-supported directions; candidate counts and retained energy are reported',
            'FFN native gates are pending until an explicitly granted run; causal unit-delta removal is a separate optional stage'],
        evidence_class='descriptive held-out perturbation selectivity' if phase=='analysis' and status=='complete' else 'instrumentation',
        next_decision=next_decision)
    events.append(entry)
    write(path,dict(schema_version=1,**entry,events=events))


def perturbations(seed):
    """Four phase/sign variants per sinusoid; two signs for each other type."""
    t = np.arange(-128, 0)
    specifications = []
    for label,period in (('sin24',24.),('sin19.2',19.2)):
        for phase in (0.,np.pi/2):
            wave = np.sin(2*np.pi*t/period+phase)
            for sign in (-1,1):
                specifications.append((dict(kind=label,phase=phase,sign=sign),sign*wave))
    step = np.r_[np.zeros(96),np.ones(32)]
    ramp = np.r_[np.zeros(64),np.linspace(0,1,64)]
    noise = np.random.default_rng(seed+100000).normal(size=128)
    for label,wave in (('steprecent32',step),('ramprecent64',ramp),('noise',noise)):
        for sign in (-1,1):
            specifications.append((dict(kind=label,phase=None,sign=sign),sign*wave))
    values = np.asarray([.1*x/np.sqrt(np.mean(x*x)) for _,x in specifications])
    return values,[m for m,_ in specifications]


def synthetic_background(seed, family):
    """Fresh AR backgrounds with balanced slow structure, not forecast targets."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=128)
    for i in range(1,128):
        x[i] += .65*x[i-1]
    t = np.arange(128)
    if family==1:
        x += rng.uniform(-2,2)*np.linspace(-1,1,128)
    elif family==2:
        x += rng.uniform(.5,2)*np.sin(2*np.pi*t/rng.uniform(30,70)+rng.uniform(-np.pi,np.pi))
    elif family==3:
        x[64:] += rng.uniform(-2,2)
    return normalize_history(x)[0]


def reused_real(pilot):
    """Read only verified past roots, never target archives or future scores."""
    names = ('protocol.json','bank.npz','rows.json','plans.json')
    if not all((pilot/name).exists() for name in names):
        return [],dict(status='omitted',reason='Past-only pilot artifacts unavailable')
    p = read(pilot/'protocol.json')
    for name in names[1:]:
        if sha(pilot/name)!=p['hashes'][name]:
            raise ValueError('Reused past-source hash changed')
    rows,plans,bank = read(pilot/'rows.json'),read(pilot/'plans.json'),load_bank(pilot/'bank.npz')
    domains = ('monash_electricity_hourly','m4_hourly','monash_pedestrian_counts','monash_traffic')
    selected = []
    for domain in domains:
        for split in ('FIT','TEST'):
            ids = [i for i,r in enumerate(rows) if r['domain']==domain and r['split']==split]
            if not ids:
                return [],dict(status='omitted',reason='Need one existing FIT and TEST past source per domain')
            i=ids[0]
            if plans[i][2]['origin']!=256:
                raise ValueError('Expected main-origin past root')
            x=bank[plans[i][2]['levels'][0][0]]
            selected.append((dict(source=rows[i]['source'],kind='real_reused',domain=domain,split=split,
                seed=SEED+100+len(selected)),normalize_history(x)[0]))
    return selected,dict(status='reused_past_only',count=8,new_real_generalization=False,
        bindings={str(pilot/name):sha(pilot/name) for name in names})


def architecture(checkpoints):
    """Verify installed module paths and cached tensor headers without loading weights."""
    site=Path(sysconfig.get_paths()['purelib'])
    files=[site/'transformers/models/t5/modeling_t5.py',site/'chronos/chronos_bolt.py',
           site/'timesfm/torch/transformer.py',site/'timesfm/timesfm_2p5/timesfm_2p5_base.py',
           site/'timesfm/timesfm_2p5/timesfm_2p5_torch.py',site/'timesfm/torch/util.py']
    if 'hidden_states = self.wo(hidden_states)' not in files[0].read_text() or 'self.ff1(self.activation(self.ff0(' not in files[2].read_text():
        raise ValueError('Installed FFN path changed; explicit source review required')
    result={}
    for model in MODELS:
        config_path=Path(checkpoints[model]['path'])/'config.json'
        config=read(config_path)
        files.append(config_path)
        if model=='bolt' and (config.get('dense_act_fn')!='relu' or config.get('is_gated_act')
                or config['chronos_config']['input_patch_size']!=16 or not config['chronos_config']['use_reg_token']):
            raise ValueError('Bolt activation/patch geometry changed')
        weight=Path(checkpoints[model]['path'])/next(iter(checkpoints[model]['weights']))
        with weight.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0]
            header=json.loads(f.read(length))
        indices=(0,3) if model=='bolt' else (0,10)
        sites=[]
        for i in indices:
            down=f'encoder.block.{i}.layer.1.DenseReluDense.wo' if model=='bolt' else f'stacked_xf.{i}.ff1'
            residual=f'encoder.block.{i}' if model=='bolt' else f'stacked_xf.{i}'
            shape=header[down+'.weight']['shape']
            if shape!=([512,2048] if model=='bolt' else [1280,1280]):
                raise ValueError('Checkpoint FFN dimensions changed')
            sites.append(dict(layer=i,down_projection=down,residual=residual,ffn_width=shape[1],residual_width=shape[0],
                tokens=9 if model=='bolt' else 4,
                capture='forward_pre_hook args[0]',activation='ReLU' if model=='bolt' else 'SiLU'))
        result[model]=dict(sites=sites,scope='encoder FFNs only; decoder excluded' if model=='bolt' else 'decoder transformer FFNs',
            functional_wrapper_required=False,width_expansion=4 if model=='bolt' else 1)
    return dict(models=result,source_hashes={str(p):sha(p) for p in files},runtime_validated=False)


def prepare(out):
    """Freeze480-cell pilot and optional causal plan; never instantiate native models."""
    if out.exists():
        raise FileExistsError('Fresh FFN run required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        data=[(dict(source=f'syn:{seed}',kind='synthetic',domain='synthetic',seed=seed,
                    family=i%4,split='FIT' if i<8 else 'TEST'),synthetic_background(seed,i%4))
              for i,seed in enumerate(range(SEED,SEED+16))]
        real,reuse=reused_real(out.parent/'natural_validation_pilot4951001')
        data+=real
        if len(set(m['source'] for m,_ in data))!=len(data):
            raise ValueError('Duplicate source groups')
        rows,queries,conditions=[],[],None
        for meta,x in data:
            delta,conditions=perturbations(meta['seed'])
            q=np.concatenate([x[None],x[None]+delta]).astype(np.float32)
            rms=np.sqrt(np.mean((q[1:].astype(float)-q[0])**2,axis=1))
            if np.max(abs(rms-.1))>1e-7:
                raise ValueError('Input RMS equalization failed')
            rows.append(dict(meta,index=len(rows),measured_rms=rms.tolist()))
            queries.append(q)
        cells=len(rows)*(1+1+14+2+2)
        if cells>CELL_CAP:
            raise ValueError('Native cell budget exceeds500')
        natural=out.parent/'natural_validation_replication4955000'
        reference=read(natural/'protocol.json')
        execution=read(natural/'execution.json')
        capability=architecture(reference['checkpoints'])
        estimate={m:dict(reference_cells=reference['nodes'],reference_seconds=execution[m]['seconds'],
            native_cells=cells,batch1_vs_batch4_allowance_factor=4,
            estimated_seconds=execution[m]['seconds']*cells/reference['nodes']*4,
            conservative_seconds=execution[m]['seconds']*cells/reference['nodes']*8,
            peak_memory_basis='Native model <=1.64GB observed; captures <100MB/model, capped at3GiB') for m in MODELS}
        if any(v['conservative_seconds']>SECONDS for v in estimate.values()):
            raise ValueError('Expected worker budget exceeds900seconds')
        np.savez_compressed(out/'inputs.npz',queries=np.asarray(queries))
        write(out/'rows.json',rows)
        write(out/'conditions.json',conditions)
        write(out/'capability.json',capability)
        write(out/'reuse.json',reuse)
        write(out/'budget.json',estimate)
        causal=dict(status='optional second stage only after selectivity pilot review and new grant',
            native_cells_stage1=0,budgets=[1,4],scope='Native FFN axes only',
            operation='For FIT-selected units, replace perturbed postactivation coordinate by its background value: subtract the perturbation-induced unit delta, all tokens at that site',
            control='Activity-matched random native units at equal budgets, matched using FIT activity; eight fixed random draws',
            outcomes='Change in native forecast response to the input perturbation; no true-future accuracy claim',
            uncertainty='Seed/source groups; no token bootstrap',automatic_launch=False)
        write(out/'causal_plan.json',causal)
        from experiments.feature_search import ffn_selectivity_analysis,ffn_selectivity_run,real_background_poc,homogeneous_normalizer
        paths=[Path(__file__)]+[Path(m.__file__) for m in (ffn_selectivity_analysis,ffn_selectivity_run,real_background_poc,homogeneous_normalizer)]
        files=['inputs.npz','rows.json','conditions.json','capability.json','reuse.json','budget.json','causal_plan.json']
        protocol=dict(version=1,run=out.name,hold_native=True,requires_new_slot=True,no_SAE=True,
            label='Held-out controlled perturbation selectivity; NOT general monosemanticity',
            synthetic_seeds=list(range(SEED,SEED+16)),backgrounds=len(rows),synthetic=16,reused_real=len(real),
            split='8 FIT/8 TEST synthetic;4 FIT/4 TEST real if available; all variants/tokens stay in their source group',
            types=list(TYPES),variants=14,input_RMS=.1,history=128,horizon=32,
            native_point_forecast='Median: Bolt quantile.5, TimesFM aridx5',normalization='Fixed background mean/scale, including matching inverse; one reference native pass reused per background',
            capture=capability['models'],models=list(MODELS),checkpoints=reference['checkpoints'],
            views=['absolute pooled signed-token-mean delta','pooled token-RMS delta'],
            representations=['native axes','orthogonal DCT rotations seeds4961101/2/3','FIT source/type-balanced token-delta PCA supported directions'],
            view_order='Transform tokenwise signed activation deltas before pooling. Average variants within each source/type; mean and RMS views analyzed separately.',
            feature_budgets=list(BUDGETS),fitting='FIT-only ANOVA axis selection, standardization and nearest-class-centroid readout. PCA weights each source/type equally despite variant counts. FIT class preferences fixed for held-out unit response-purity measurement.',
            input_baseline='Actual float32 query-minus-background128-lag vector, same FIT selection and readout; mean/RMS views coincide with one token',
            uncertainty='1999 source-group bootstrap of held-out accuracy/purity and paired method differences; synthetic and reused-real strata separate; exploratory, no multiplicity-confirmed monosemanticity claims',
            cells_per_model=cells,cell_breakdown_per_background=dict(native_reference=1,locked_reference=1,perturbations=14,self_replays=2,reach_checks=2),
            cell_definition='One128-point context passed through native predict; normalizer-only arithmetic is not an extra model pass',
            native_batch=1,cpu_threads=1,worker_cap=MEMORY,seconds_per_model=SECONDS,
            gates=dict(normalizer_prediction_identity_exact=True,normalizer_capture_identity_exact=True,
                FFN_self_replay_exact=True,hook_firings_per_site_per_pass=1,within_FFN_reach_relative_gt=1e-6,
                finite=True,input_RMS_tolerance=1e-7,all_groups_disjoint=True),
            causal_stage='Registered only; not part of this native execution',
            worker_reads=['protocol.json','inputs.npz','rows.json','conditions.json'],
            sources={**capability['source_hashes'],**{str(p.resolve()):sha(p) for p in paths}},
            natural_reference_bindings={str(natural/name):sha(natural/name) for name in ('protocol.json','execution.json')},
            hashes={name:sha(out/name) for name in files})
        write(out/'protocol.json',protocol)
    except Exception as exc:
        knowledge(out,'prepare','failed',dict(error_type=type(exc).__name__,error=str(exc),native_launched=False),
                  'Resolve recorded preparation failure. HOLD native.')
        raise
    knowledge(out,'prepare','complete',dict(backgrounds=len(rows),native_cells_per_model=cells,budget=estimate,
        reused_real=reuse,capability=capability,native_launched=False,native_gates_pending=True,
        protocol_sha256=sha(out/'protocol.json')),'HOLD native pending a new slot grant. No SAE and no causal second-stage execution.')
    print(dict(backgrounds=len(rows),native_cells=cells,budget=estimate,status='HOLD native'),flush=True)


def sites(native):
    """Resolve real projection modules; residuals are captured from the same blocks."""
    sequence=native.model.encoder.block if native.name=='bolt' else native.model.stacked_xf
    result={}
    for tag,i in (('first',0),('half',len(sequence)//2)):
        block=sequence[i]
        down=block.layer[1].DenseReluDense.wo if native.name=='bolt' else block.ff1
        result[tag]=(down,block)
    return result


def capture(native,x,selected,replacement=None):
    """Capture postactivation projection inputs, with optional within-FFN replacement."""
    import torch
    from tsfm_lens.extraction.hooks import _primary
    observed={f'{tag}_{space}':[] for tag in selected for space in ('ffn','residual')}
    handles=[]
    for tag,(down,block) in selected.items():
        def before(module,args,tag=tag):
            value=args[0]
            changed=replacement is not None and replacement[0]==tag
            if changed:
                patched=torch.as_tensor(replacement[1],device=value.device,dtype=value.dtype)
                if patched.shape!=value.shape:
                    raise ValueError('FFN replacement geometry mismatch')
                value=patched
            observed[f'{tag}_ffn'].append(value.detach().cpu().numpy().copy())
            if changed:
                return (value,)+args[1:]
        def after(module,args,output,tag=tag):
            observed[f'{tag}_residual'].append(_primary(output).detach().cpu().numpy().copy())
        handles.append(down.register_forward_pre_hook(before))
        handles.append(block.register_forward_hook(after))
    try:
        forecast=native.predict(x)[0]
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values)!=1 for values in observed.values()):
        raise ValueError('Expected one hook firing/site/pass')
    return forecast,{key:values[0] for key,values in observed.items()}


@contextlib.contextmanager
def locked_background(native,anchor,invoke):
    """Yield the already-captured native reference, then lock its exact normalizer."""
    import torch
    if native.name=='bolt':
        reference=invoke(anchor)
        module=native.model.instance_norm
        original=module.forward
        with torch.no_grad():
            _,stats=original(torch.as_tensor(anchor,dtype=torch.float32))
        def fixed(x,loc_scale=None):
            return original(x,stats)
        module.forward=fixed
        try:
            yield reference
        finally:
            module.forward=original
    else:
        module=sys.modules[type(native.model).__module__]
        original=module.revin
        stats=[]
        def trace(x,mu,sigma,reverse=False):
            if not reverse:
                stats.append((mu.detach().clone(),sigma.detach().clone()))
            return original(x,mu,sigma,reverse=reverse)
        module.revin=trace
        try:
            reference=invoke(anchor)
        finally:
            module.revin=original
        if len(stats)!=1:
            raise ValueError('Expected one normalization prefill per128-point reference')
        mu,sigma=stats[0]
        def fixed(x,unused_mu,unused_sigma,reverse=False):
            return original(x,mu,sigma,reverse=reverse)
        module.revin=fixed
        try:
            yield reference
        finally:
            module.revin=original


def exact_gate(reference,actual,label):
    """A failed numerical identity is not a usable selectivity instrument."""
    if not np.array_equal(reference,actual):
        raise ValueError(f'{label}: exact replay failed')


def reach_gate(reference,changed):
    """Full within-FFN zeroing must reach the native forecast on each background/site."""
    value=float(np.max(abs(np.asarray(changed)-reference))/max(1.,float(np.max(abs(reference)))))
    if value<=1e-6:
        raise ValueError('Within-FFN reach gate failed')
    return value


def worker(out,model,native_factory=None):
    """Native stage1 only;500-cell cap includes all replay/reach/normalizer passes."""
    if os.environ.get('FFN_SELECTIVITY_MONITORED')!='1':
        raise RuntimeError('HOLD native until a new granted monitored run')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in THREAD_VARS):
        raise RuntimeError('Requires one-thread CPU worker')
    p=read(out/'protocol.json')
    for path,digest in p['sources'].items():
        if sha(path)!=digest:
            raise ValueError('Frozen source changed')
    for name in ('inputs.npz','rows.json','conditions.json'):
        if sha(out/name)!=p['hashes'][name]:
            raise ValueError('Frozen worker input changed')
    if (out/f'{model}_captures.npz').exists():
        raise FileExistsError('Refuse capture overwrite')
    checkpoint=p['checkpoints'][model]
    for name,digest in checkpoint['weights'].items():
        if sha(Path(checkpoint['path'])/name)!=digest:
            raise ValueError('Checkpoint changed')
    if native_factory is None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from experiments.feature_search.real_background_poc import Native
        native_factory=Native
    started=time.monotonic()
    native=native_factory(model,checkpoint['path'])
    selected=sites(native)
    with np.load(out/'inputs.npz',allow_pickle=False) as f:
        queries=f['queries']
    rows=read(out/'rows.json')
    if queries.shape!=(p['backgrounds'],15,128) or len(rows)!=len(queries) or not np.isfinite(queries).all():
        raise ValueError('Frozen background/input geometry invalid')
    count=0
    def invoke(x,replacement=None):
        nonlocal count
        count+=len(x)
        if count>CELL_CAP or count>p['cells_per_model']:
            raise RuntimeError('Native cell cap exceeded')
        if time.monotonic()-started>SECONDS or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>MEMORY:
            raise RuntimeError('Worker resource cap exceeded')
        return capture(native,x,selected,replacement)
    arrays={f'{tag}_{space}':[] for tag in selected for space in ('ffn','residual')}
    predictions=[]
    reach=[]
    for i,q in enumerate(queries):
        anchor=q[0:1]
        with locked_background(native,anchor,invoke) as (reference,original):
            baseline,states=invoke(anchor)
            exact_gate(reference,baseline,'normalizer prediction')
            for spec,tag in zip(p['capture'][model]['sites'],selected):
                if (states[f'{tag}_ffn'].shape[1:]!=(spec['tokens'],spec['ffn_width'])
                        or states[f'{tag}_residual'].shape[1:]!=(spec['tokens'],spec['residual_width'])):
                    raise ValueError('FFN/residual capture geometry differs from frozen architecture')
            for key in states:
                exact_gate(original[key],states[key],'normalizer '+key)
            for tag in selected:
                replay,_=invoke(anchor,(tag,states[f'{tag}_ffn']))
                exact_gate(baseline,replay,'FFN '+tag)
                changed,_=invoke(anchor,(tag,np.zeros_like(states[f'{tag}_ffn'])))
                reach.append(dict(background=i,site=tag,relative_change=reach_gate(baseline,changed)))
            group={key:[value[0]] for key,value in states.items()}
            forecasts=[baseline[0]]
            for query in q[1:]:
                pred,values=invoke(query[None])
                forecasts.append(pred[0])
                for key,value in values.items():
                    group[key].append(value[0])
            predictions.append(forecasts)
            for key in arrays:
                arrays[key].append(np.asarray(group[key]))
    if count!=p['cells_per_model']:
        raise ValueError('Unexpected native pass count')
    arrays={key:np.asarray(value) for key,value in arrays.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()) or not np.isfinite(predictions).all():
        raise ValueError('Nonfinite captures/forecasts')
    np.savez_compressed(out/f'{model}_captures.npz',**arrays,forecasts=np.asarray(predictions),
                        source_ids=np.asarray([r['source'] for r in rows]))
    write(out/f'{model}_worker.json',dict(valid=True,normalizer_identity_exact=True,self_replay_exact=True,
        all_reach_passed=True,reach=reach,cells=count,seconds=time.monotonic()-started,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        protocol_sha256=sha(out/'protocol.json'),sha256=sha(out/f'{model}_captures.npz')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','worker'])
    parser.add_argument('--out',type=Path,default=Path('runs/ffn_selectivity_pilot4961000'))
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.out)
    else:
        if args.model is None:
            parser.error('worker requires --model')
        worker(args.out,args.model)
