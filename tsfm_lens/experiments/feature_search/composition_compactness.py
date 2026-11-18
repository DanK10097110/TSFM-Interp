"""Exploratory cross-fit compactness of an oracle composition edit at cut15.

Discovery16 is already inspected. Fixed ranks1/2/4 are screened without p-values,
using opposite generating-seed parity folds and no future labels in fitting.
Separate processes isolate capture, fit, suffix evaluation and summarization.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

IDS=tuple(range(3600000,3600016))
RANKS=(1,2,4)
CAP=int(1.5*2**30)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def memory(pid='self'):
    rows=Path(f'/proc/{pid}/status').read_text(encoding='utf-8').splitlines()
    return {k:int(next(row.split()[1] for row in rows if row.startswith(k+':')))*1024 for k in ('VmRSS','VmHWM')}


def fit_crossfold(delta,seed_ids,ranks=RANKS):
    """Only edit arrays/IDs enter fitting; a shared channel projector acts per token."""
    import numpy as np
    value=np.asarray(delta,dtype=float);ids=np.asarray(seed_ids)
    if value.ndim!=3 or value.shape[1]!=4 or len(value)!=16 or ids.shape!=(16,) or ids.dtype.kind not in 'iu' or set(ids.tolist())!=set(IDS):
        raise ValueError('Need all16 distinct generating IDs and four-token channel deltas')
    projected={rank:np.zeros_like(value) for rank in ranks};mean=np.zeros_like(value)
    validity={rank:True for rank in ranks};records=[]
    for parity in (0,1):
        evaluation=np.flatnonzero(ids%2==parity);training=np.flatnonzero(ids%2!=parity)
        if len(evaluation)!=8 or len(training)!=8:raise ValueError('Need8training/8evaluation seeds')
        matrix=value[training].reshape(-1,value.shape[-1])/np.sqrt(4.)
        _,singular,right=np.linalg.svd(matrix,full_matrices=False)
        tolerance=max(matrix.shape)*np.finfo(np.float64).eps*(singular[0] if len(singular) else 0.)
        numerical_rank=int(np.sum(singular>tolerance))
        channel_mean=value[training].mean(axis=(0,1))
        mean[evaluation]=channel_mean
        record=dict(evaluation_parity=parity,training_ids=ids[training].tolist(),evaluation_ids=ids[evaluation].tolist(),
            training_seed_count=8,dependent_token_rows=32,numerical_rank=numerical_rank,numerical_rank_tolerance=float(tolerance),
            singular_values=singular.tolist(),bases={})
        for rank in ranks:
            if rank>numerical_rank:
                validity[rank]=False;continue
            basis=right[:rank]
            projected[rank][evaluation]=(value[evaluation]@basis.T)@basis
            record['bases'][str(rank)]=basis.tolist()
        records.append(record)
    return projected,mean,validity,records


def decide(native,full,mean_only,rank_losses,random_losses,validity):
    """Retain all ranks; empirical screening only, no statistical confirmation."""
    import numpy as np
    native,full,mean_only=[np.asarray(x,dtype=float) for x in (native,full,mean_only)]
    if any(x.shape!=(16,) or not np.isfinite(x).all() for x in (native,full,mean_only)):raise ValueError('Need all16 finite seed losses')
    reference=float(np.mean(native-full));rows={}
    for rank in RANKS:
        loss=np.asarray(rank_losses[rank]);random=np.asarray(random_losses[rank])
        if loss.shape!=(16,) or random.shape!=(16,):raise ValueError('No missing seed rows permitted')
        valid=bool(validity[rank] and np.isfinite(loss).all() and np.isfinite(random).all())
        recovery=None if reference<=0 or not valid else float(np.mean(native-loss)/reference)
        rows[str(rank)]=dict(valid_all16=valid,recovery=recovery,mean_loss=float(loss.mean()) if valid else None,
            random_advantage=float(np.mean(random-loss)) if valid else None,
            mean_only_advantage=float(np.mean(mean_only-loss)) if valid else None)
        row=rows[str(rank)]
        row['passed']=bool(valid and recovery is not None and recovery>=.8 and row['mean_loss']<=.06 and
            row['random_advantage']>=.05 and row['mean_only_advantage']>=.05)
    selected=[rank for rank in RANKS if rows[str(rank)]['passed']]
    return dict(ranks=rows,smallest_passing_rank=selected[0] if selected else None,full_edit_mean_improvement=reference,
                mean_only_mean_loss=float(mean_only.mean()),scope='Exploratory reused16 screen only; no p-values, confirmation, deployable repair or canonical feature')


def prepare(root,out):
    """Freeze prospective screen declarations and old-data/reference bindings only."""
    from experiments.feature_search import composition_restoration as source
    from experiments.feature_search import homogeneous_normalizer as helper
    out.mkdir(parents=True,exist_ok=False)
    previous=root/'tsfm_lens/runs/feature_search_stage49_restoration3600000_memorysafe'
    protocol=read(previous/'protocol.json');result=read(previous/'results.json')
    if result['effective_earliest_cut']!=15 or not result['numerical_checks']['passed']:raise ValueError('Validated cut15 discovery required')
    if sha(source.__file__)!=protocol['driver_sha256'] or sha(helper.__file__)!=protocol['helper_sha256']:raise ValueError('Frozen scientific helper changed')
    spec=dict(seed_ids=IDS,ranks=RANKS,cut=15,cut_index=14,forbidden_depth_search=True,
        folds='Generatingseed parity:fit8opposite/apply8;all4tokenrows weighted1/4 perseed,not independent tokens',
        projector='UNCENTERED FP64 SVD;shared channel basis,per-token application;no truth/forecast in fit',
        numerical_rank='max(training_matrix.shape)*eps64*s_max;null-vector ranks unscorable;not semantic stability',
        mean_control='Trainingfold channel mean delta,broadcast to all tokens;does not exclude every position template',
        random_controls=2,random_key='SeedSequence([3800173,generating_seed,rank])',
        random_scope='Existing norm/radial/editGram-matched helper;not matched-rank/capacity equivalent',
        gates=dict(recovery=.8,mean_loss=.06,random_advantage=.05,mean_only_advantage=.05),
        no_statistics='No p-values/confirmation on inspected16;smallestpassingrank only nominates separatefreshcheck',
        identity_RMS=1e-5,all16_geometry_per_rank=True,cpu_threads=1,runtime_seconds=180,resident_limit_bytes=CAP,
        checkpoint=protocol['checkpoint'],checkpoint_sha256=protocol['checkpoint_weight_sha256'],installed_sources=protocol['source_sha256'],
        helper=str(Path(source.__file__).resolve()),helper_sha256=sha(source.__file__),loader=str(Path(helper.__file__).resolve()),loader_sha256=sha(helper.__file__),
        discovery=str(previous),discovery_bindings={name:sha(previous/name) for name in ['protocol.json','results.json','corpus.npz','metadata.json','forecast_waveforms.npz']},
        driver_sha256=sha(__file__))
    write(out/'protocol.json',spec)


def verified(out):
    p=read(out/'protocol.json')
    if p['driver_sha256']!=sha(__file__) or p['helper_sha256']!=sha(p['helper']) or p['loader_sha256']!=sha(p['loader']):raise ValueError('Source binding changed')
    if p['seed_ids']!=list(IDS) or p['ranks']!=list(RANKS) or p['cut']!=15 or p['cut_index']!=14:raise ValueError('Fixed screen membership changed')
    if p['gates']!=dict(recovery=.8,mean_loss=.06,random_advantage=.05,mean_only_advantage=.05) or p['random_controls']!=2 or p['cpu_threads']!=1 or p['runtime_seconds']!=180 or p['resident_limit_bytes']!=CAP or p['identity_RMS']!=1e-5:raise ValueError('Frozen scientific/resource declarations changed')
    for name,value in p['discovery_bindings'].items():
        if sha(Path(p['discovery'])/name)!=value:raise ValueError('Inspected discovery source changed')
    return p


def verify_native(p):
    """Native imports/weight hashing belong only to isolated model workers."""
    for name,value in p['installed_sources'].items():
        module=importlib.import_module(name)
        if sha(module.__file__)!=value:raise ValueError('Installed native source changed')
    if sha(p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Checkpoint changed')


def failed_execution(result,reason):
    """Resource/phase failure overrides any conditional compactness selection."""
    result=dict(result or {})
    if result.get('smallest_passing_rank') is not None:result['conditional_smallest_rank']=result['smallest_passing_rank']
    result.update(execution_valid=False,smallest_passing_rank=None,status='failed',failure_reason=reason)
    return result


def capture(out):
    """Capture only selected15, then release weights; check old native/full edit."""
    import numpy as np
    import torch
    from experiments.feature_search import composition_restoration as source
    from experiments.feature_search import homogeneous_normalizer as helper
    p=verified(out)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only capture must hide CUDA')
    verify_native(p)
    contexts,truth,metadata=source.make_data(IDS);cache=[]
    model=helper.load_model(Path(p['checkpoint']))
    def hook(_,args,value):cache.append(value[0].detach().clone())
    handle=model.stacked_xf[14].register_forward_hook(hook);prediction=[]
    with torch.inference_mode():
        for start in range(0,len(contexts),8):
            x=torch.as_tensor(contexts[start:start+8]);value,_,extra=model.decode(64,x,torch.zeros_like(x,dtype=torch.bool))
            if extra is not None and extra.numel():raise ValueError('Unexpected rollout')
            prediction.append(value[:,-1,:64,model.aridx].numpy())
    handle.remove();state=torch.cat(cache).numpy();prediction=np.concatenate(prediction).astype(float)
    mu,sigma=source.normalizers(contexts);ix={c:np.arange(16)*4+i for i,c in enumerate(source.CONDITIONS)}
    sm=sigma[ix['baseline']].astype(float);mm=mu[ix['baseline']].astype(float)
    if state.shape!=(64,4,1280):raise ValueError('Cut15 must retain native4x1280 state')
    h=state[ix['baseline']]
    anchor=source.anchored(h,state[ix['pure_phi']],state[ix['pure_psi']],state[ix['background']],sm,sigma[ix['pure_phi']],sigma[ix['pure_psi']])
    full=source.suffix(model,anchor,15)*sm[:,None]+mm[:,None]
    scale=np.asarray([np.sqrt((m['a']**2+m['b']**2)/2) for m in metadata[::4]])
    with np.load(Path(p['discovery'])/'forecast_waveforms.npz',allow_pickle=False) as old:
        error=lambda a,b:float(np.sqrt(np.mean(((a-b)/scale[:,None])**2,axis=1)).max())
        checks=dict(native_RMS=error(prediction[ix['baseline']],old['native']),full_edit_RMS=error(full,old['cut15_anchored']))
    del model;import gc;gc.collect()
    np.savez_compressed(out/'captured.npz',receiver=h,delta=anchor-h,seed_ids=np.asarray(IDS),mu=mm,sigma=sm,
        native=prediction[ix['baseline']],full_edit=full,truth=truth[::4],common_scale=scale)
    checks['passed']=max(checks.values())<=1e-5
    checks['captured_sha256']=sha(out/'captured.npz')
    write(out/'capture_checks.json',checks)
    if not checks['passed']:raise ValueError('Original native/full-edit numerical identity failed')


def fit(out):
    """Read ONLY receiver/delta/IDs, never future/forecast arrays from capture file."""
    import numpy as np
    from experiments.feature_search import composition_restoration as source
    verified(out)
    capture_record=read(out/'capture_checks.json')
    if not capture_record['passed'] or sha(out/'captured.npz')!=capture_record['captured_sha256']:raise ValueError('Captured state binding or numerical gate changed')
    with np.load(out/'captured.npz',allow_pickle=False) as data:h=data['receiver'];delta=data['delta'];ids=data['seed_ids']
    projected,mean,valid,folds=fit_crossfold(delta,ids)
    arrays=dict(mean_only_delta=mean);geometry={}
    for rank in RANKS:
        arrays[f'rank{rank}_delta']=projected[rank];geometry[str(rank)]=[]
        controls=[np.empty_like(delta) for _ in range(2)]
        for index,seed in enumerate(ids):
            rng=np.random.default_rng(np.random.SeedSequence([3800173,int(seed),rank]))
            edits,flags,diagnostics=source.random_edits(h[index:index+1],projected[rank][index:index+1],rng,2,return_diagnostics=True)
            if flags[0]:valid[rank]=False
            geometry[str(rank)].append(dict(seed=int(seed),unidentifiable=bool(flags[0]),diagnostics=diagnostics[0]))
            for draw in range(2):controls[draw][index]=edits[draw,0]
        for draw,value in enumerate(controls):arrays[f'rank{rank}_random{draw}_delta']=value
    np.savez_compressed(out/'projections.npz',**arrays)
    write(out/'fit.json',dict(captured_sha256=sha(out/'captured.npz'),projections_sha256=sha(out/'projections.npz'),folds=folds,validity={str(k):bool(v) for k,v in valid.items()},geometry=geometry,
        scope='Fit used receiver/edit/seed only;no truth/future/forecast labels;all ranks retained'))


def evaluate(out):
    """Suffix-only worker; no SVD/statistical/model pipelines share its lifetime."""
    import numpy as np
    from experiments.feature_search import composition_restoration as source
    from experiments.feature_search import homogeneous_normalizer as helper
    p=verified(out)
    if not read(out/'capture_checks.json')['passed']:raise ValueError('Captured numerical gate failed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only suffix must hide CUDA')
    verify_native(p)
    fitted=read(out/'fit.json')
    if sha(out/'captured.npz')!=fitted['captured_sha256'] or sha(out/'projections.npz')!=fitted['projections_sha256']:raise ValueError('Fitted state/projection binding changed')
    with np.load(out/'captured.npz',allow_pickle=False) as f:h=f['receiver'];mu=f['mu'];sigma=f['sigma']
    with np.load(out/'projections.npz',allow_pickle=False) as f:edits={key:f[key] for key in f.files}
    model=helper.load_model(Path(p['checkpoint']))
    waveforms={key:source.suffix(model,h+value,15)*sigma[:,None]+mu[:,None] for key,value in edits.items()}
    del model;import gc;gc.collect()
    np.savez_compressed(out/'projected_forecasts.npz',**waveforms)
    write(out/'evaluation.json',dict(forecasts_sha256=sha(out/'projected_forecasts.npz'),captured_sha256=sha(out/'captured.npz'),projections_sha256=sha(out/'projections.npz')))


def summarize(out):
    """Score once every held-out projection is frozen; no p-values or rank rescue."""
    import numpy as np
    verified(out)
    if not read(out/'capture_checks.json')['passed']:raise ValueError('Captured numerical gate failed')
    evaluated=read(out/'evaluation.json');fitted=read(out/'fit.json')
    if sha(out/'projected_forecasts.npz')!=evaluated['forecasts_sha256'] or sha(out/'captured.npz')!=evaluated['captured_sha256'] or sha(out/'projections.npz')!=evaluated['projections_sha256'] or sha(out/'projections.npz')!=fitted['projections_sha256']:raise ValueError('Scoring phase binding changed')
    with np.load(out/'captured.npz',allow_pickle=False) as f:data={key:f[key] for key in ('native','full_edit','truth','common_scale')}
    with np.load(out/'projected_forecasts.npz',allow_pickle=False) as f:forecasts={key:f[key] for key in f.files}
    mse=lambda y:np.mean((y-data['truth'])**2,axis=1)/data['common_scale']**2
    fit_record=read(out/'fit.json');valid={r:fit_record['validity'][str(r)] for r in RANKS}
    losses={r:mse(forecasts[f'rank{r}_delta']) for r in RANKS}
    random={r:np.mean([mse(forecasts[f'rank{r}_random{i}_delta']) for i in range(2)],axis=0) for r in RANKS}
    result=decide(mse(data['native']),mse(data['full_edit']),mse(forecasts['mean_only_delta']),losses,random,valid)
    result['source_hashes']={name:sha(out/name) for name in ['protocol.json','captured.npz','capture_checks.json','projections.npz','fit.json','projected_forecasts.npz']}
    write(out/'results.json',result)


def run(out):
    """One lightweight controller, serial phases, full-process RSS/HWM accounting."""
    verified(out)
    if (out/'results.json').exists() or any(out.glob('*_exit_resource.json')):raise ValueError('No overwriting/restarting a completed or failed screen')
    started=time.monotonic();controller_peak=memory()['VmHWM'];phases={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for mode in ('capture','fit','evaluate','summarize'):
        with (out/f'{mode}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
            maximum=0;peak=0;reason=None
            while child.poll() is None:
                controller_peak=max(controller_peak,memory()['VmHWM'])
                try:status=memory(child.pid);peak=max(peak,status['VmHWM']);maximum=max(maximum,memory()['VmRSS']+status['VmRSS'])
                except (FileNotFoundError,StopIteration):pass
                if maximum>CAP or peak+controller_peak>CAP:reason='Task resident limit exceeded'
                if time.monotonic()-started>180:reason='Whole screen runtime limit exceeded'
                if reason:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                time.sleep(.05)
        exit_path=out/f'{mode}_exit_resource.json'
        if exit_path.exists():peak=max(peak,read(exit_path)['VmHWM'])
        else:reason=reason or 'Missing worker final resource record'
        if peak+controller_peak>CAP:reason=reason or 'Final worker high water exceeded task cap'
        if time.monotonic()-started>180:reason=reason or 'Final whole-screen runtime exceeded180s'
        phases[mode]=dict(returncode=child.returncode,maximum_simultaneous_RSS=maximum,conservative_peak_sum=peak+controller_peak)
        if reason or child.returncode!=0:
            why=reason or f'{mode} failed'
            write(out/'failure.json',dict(status='failed',reason=why,phases=phases))
            existing=read(out/'results.json') if (out/'results.json').exists() else {}
            failure=failed_execution(existing,why);failure['execution_phases']=phases
            write(out/'results.json',failure);return
    result=read(out/'results.json');result.update(execution_phases=phases,total_wall_seconds=time.monotonic()-started,
        controller_peak_bytes=controller_peak,execution_valid=True,status='complete',execution_scope='Reused16 exploratory screen;not new scientific confirmation')
    write(out/'results.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','run','capture','fit','evaluate','summarize'],required=True)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.out)
    else:
        began=time.monotonic()
        try:globals()[a.mode](a.out)
        finally:
            if a.mode!='run':write(a.out/f'{a.mode}_exit_resource.json',dict(**memory(),wall_seconds=time.monotonic()-began))
