"""Fresh synthetic replication with frozen old-FIT probes and bounded native workers.

Pilot modules remain immutable. All new seeds are TEST-only. The added
input+forecast+hidden arm is an explicitly exploratory old-FIT-only extension.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from scipy.linalg import solve
from experiments.feature_search.comparability_core import (ALPHAS,descriptors,folds,group_values,interval,kernel_bank,ridge,stable_argmin,weights)
from experiments.feature_search.comparability_pilot import ROOT,read,write,sha,load_data,verify as verify_pilot

PILOT=ROOT/'runs/comparability_pilot4951000'
DEFAULT=ROOT/'runs/comparability_replication4952000'
PRIMARY=(('bolt','timesfm','forecast'),('timesfm','bolt','forecast'),('timesfm','bolt','fusion_response'))
ARMS=('input','input+A_hidden','input+A_forecast','input+B_hidden','input+A_forecast+A_hidden')


def knowledge(out,event):
    """Persist every preparation, worker outcome and final interpretation without losing history."""
    path=out/'knowledge.json'
    data=read(path) if path.exists() else dict(schema_version=1,run=out.name,question='Replicate cross-model observable prediction on64 fresh sources with frozen old-FIT probes.',evidence_class='Pending fresh source-held-out predictive replication; no mechanistic or semantic identity claim.',events=[])
    data['events'].append(dict(unix_time=time.time(),**event))
    tmp=out/'knowledge.tmp'
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)


def generate(seeds):
    """Repeat the original five-condition generator exactly, sharing noise/phase within seed."""
    rows=[];histories=[];futures=[]
    t=np.arange(-128,32)
    cases=[('on_grid_far',5.,17.,8.),('off_grid_far',5.35,17.35,8.),('off_grid_near',5.35,6.35,8.),('chirp',5.35,17.35,8.),('balanced_offgrid',5.35,17.35,.5)]
    for seed in seeds:
        rng=np.random.default_rng(seed);phi,psi=rng.uniform(-np.pi,np.pi,2);noise=rng.normal(0,.05,len(t))
        for case,a,b,amp in cases:
            phase=2*np.pi*a*t/128+phi
            if case=='chirp':phase+=2*np.pi*2*(t/128)**2
            value=.5*np.sin(phase)+amp*np.sin(2*np.pi*b*t/128+psi)+noise
            rows.append(dict(source=f'syn:{seed}',seed=int(seed),domain='synthetic',case=case,split='test',run='comparability_replication4952000',index=len(rows)))
            histories.append(value[:128]);futures.append(value[128:])
    return rows,np.asarray(histories),np.asarray(futures)


def gram_features(features):
    """Keep models in their own coordinates, comparing only predictive kernels."""
    return {name:features[name]@features[name].T for name in ('A_hidden','B_hidden','A_forecast')}


def extension_kernel(bank,spec):
    """Equal-weight input, A forecast and A hidden; no TEST-dependent mixing."""
    return (2*bank['input+A_forecast'][spec['base_kernel']]+bank['A_hidden'][spec['hidden_kernel']])/3


def selected_kernel(bank,arm,spec):
    """Resolve only a frozen candidate; never select using replication outcomes."""
    return extension_kernel(bank,spec) if arm=='input+A_forecast+A_hidden' else bank[arm][spec['kernel']]


def choose_extension(banks,y,rows,base_kernel):
    """Tune the exploratory addition on old FIT sources only, keeping the base features fixed."""
    specs=[dict(base_kernel=base_kernel,hidden_kernel=f'pca{rank}',alpha=alpha) for rank in (8,16) for alpha in ALPHAS]
    scores=[]
    for spec in specs:
        all_values=[];all_ids=[]
        for tr,va,bank in banks:
            pred=ridge(extension_kernel(bank,spec),y,tr,va,weights(rows,tr),spec['alpha'])
            all_values.extend(np.mean((pred-y[va])**2,axis=1));all_ids.extend(va)
        scores.append(float(np.mean(list(group_values(all_values,rows,all_ids).values()))))
    chosen=stable_argmin(scores)
    return dict(specs[chosen],cv_loss=scores[chosen],status='exploratory old-FIT-only follow-up')


def source_files():
    """Bind native helper implementations as well as all new replication code and tests."""
    folder=ROOT/'experiments/feature_search'
    names=['comparability_replication.py','comparability_core.py','comparability_pilot.py','selective_study.py','selective_tree.py','coherence_study.py','real_background_poc.py','homogeneous_normalizer.py']
    return [folder/n for n in names]+[ROOT/'tests/test_comparability_replication.py',ROOT/'tsfm_lens/extraction/hooks.py']


def prepare(out):
    """Generate fresh inputs and serialize frozen old-FIT support, coefficients and choices."""
    from experiments.feature_search.selective_tree import decompose
    verify_pilot(PILOT)
    out.mkdir(parents=True,exist_ok=False)
    newrows,x,y=generate(range(4952000,4952064))
    parts=[];plans=[]
    for xx in x:
        levels,details=decompose(xx)
        parts.append(levels[1] if len(levels)>1 else [xx,np.zeros_like(xx)])
        plans.append(dict(details=details,levels=[None]*len(levels),accepted=len(levels)>1))
    np.savez_compressed(out/'inputs.npz',history=x.astype(np.float32),parts=np.asarray(parts,np.float32))
    np.savez_compressed(out/'targets.npz',future=y)
    write(out/'rows.json',newrows);write(out/'plans.json',plans)
    oldrows,ox,oh,oy,_,_=load_data()
    fit=np.array([i for i,r in enumerate(oldrows) if r['domain']=='synthetic' and r['split']=='fit'])
    rs=[oldrows[i] for i in fit]
    assert not ({r['source'] for r in rs}&{r['source'] for r in newrows})
    support=dict(input=ox[fit],bolt_hidden=oh['bolt'][fit],timesfm_hidden=oh['timesfm'][fit],bolt_forecast=oy['bolt']['forecast'][fit],timesfm_forecast=oy['timesfm']['forecast'][fit])
    np.savez_compressed(out/'fit_support.npz',**support);write(out/'fit_rows.json',rs)
    original=read(PILOT/'selections.json');specs=[];coefs={};replay=[]
    tr=np.arange(len(rs));w=weights(rs,tr);sw=np.sqrt(w)
    for a,b,ep in PRIMARY:
        features=dict(input=support['input'],A_hidden=support[a+'_hidden'],B_hidden=support[b+'_hidden'],A_forecast=support[a+'_forecast'])
        final=kernel_bank(features['input'],gram_features(features),rs,tr)
        banks=[(t,v,kernel_bank(features['input'],gram_features(features),rs,t)) for t,v in folds(rs,tr)]
        lookup={r['arm']:r for r in original if r['cohort']=='synthetic' and r['A']==a and r['B']==b and r['endpoint']==ep}
        yy=oy[b][ep][fit]
        extra=choose_extension(banks,yy,rs,lookup['input+A_forecast']['kernel'])
        for arm in ARMS:
            spec=extra if arm==ARMS[-1] else {k:lookup[arm][k] for k in ('kernel','alpha','cv_loss')}
            key=f'{a}_to_{b}__{ep}__{arm}'
            kk=selected_kernel(final,arm,spec)
            mean=w@yy
            coef=sw[:,None]*solve(kk*sw[:,None]*sw[None,:]+spec['alpha']*np.eye(len(tr)),(yy-mean)*sw[:,None],assume_a='pos')
            coefs[key+'__coef']=coef;coefs[key+'__mean']=mean;coefs[key+'__train_kernel']=kk
            reference=ridge(kk,yy,tr,tr,w,spec['alpha'])
            error=float(abs(kk@coef+mean-reference).max());assert error<1e-10
            replay.append(dict(key=key,max_prediction_error=error))
            specs.append(dict(key=key,A=a,B=b,endpoint=ep,arm=arm,**spec))
    np.savez_compressed(out/'frozen_probes.npz',**coefs)
    write(out/'probe_specs.json',specs);write(out/'probe_replay.json',replay)
    checkpoint=read(ROOT/'runs/selective_replication4942000/protocol.json')['checkpoints']
    protocol=dict(version=1,seeds=list(range(4952000,4952064)),rows=320,source_count=64,cases=[r['case'] for r in newrows[:5]],
        status='Frozen before native inference; slot2 granted by user after pilot review.',
        primary=[dict(A=a,B=b,endpoint=e) for a,b,e in PRIMARY],
        discovery='Synthetic sources from original pilot and replication FIT only;20 source groups,96 rows. Original selected kernels/alphas frozen from pilot selections.json. Coefficients and training support serialized; no pilot TEST target used in fitting or extension selection.',
        exploratory='input+Aforecast+Ahidden fixes original input+Aforecast feature choice, then selects added hidden rank8/16 and alpha on oldFIT source folds only. This arm was not a pilot primary. Fresh TEST cannot tune any arm.',
        preprocessing='Exactly the frozen pilot weighted centering, global hidden scale and dual PCA; input includes full history ridge, descriptor ridge, PCA and RBF selection. Targets and inverse use identical history mean/SD in every arm.',
        capture=dict(bolt=dict(first='0',half='3',shape=[9,512]),timesfm=dict(first='0',half='10',shape=[4,1280]),batch=1,normalization='Original native baseline; fusion leaves/zero use original full-context native normalizer and matching inverse.',fusion='First block; depth1; equal leaf weights; affine zero correction; pure blend1. No random edits, deeper trees, stability sweeps, or other native endpoints. Tree abstention yields exact native forecast.'),
        gates='Exact repeat, full-context locked-normalizer identity, first-block self patch and half-depth self patch on every row; input leaf closure <=2e-6 relative max(1,maxabs(history)). Native cuts/shapes/checkpoint hashes must match.',
        uncertainty='64 seed groups,5 dependent conditions each. Equal source means and4000 paired source bootstrap draws. Primary3 gain-over-input intervals98.33333333333333%;95% descriptive intervals otherwise. No tuning/selection on fresh TEST.',
        source_null='99 fixed evaluation-source permutations (seed4952100+j), matching each of five conditions within whole seed bundles. Frozen probes and input/Aforecast stay fixed; only Ahidden is source-shuffled. Report alignment tail rank as diagnostic, not conditional-independence or confirmatory p-value. TEST sources never enter training.',
        success='Each primary: >=5% loss reduction vs frozen strong input, Bonferroni3 interval lower>0, gain above95th percentile of99 alignment-shuffle gains. Report input+Aforecast comparison and same-model comparator regardless of success. Extra hidden beyond input+Aforecast is exploratory and cannot rescue failed primary.',
        behavior_vs_semantics='Response target is B fusion-minus-native, not known component semantics. Separately report native/fused error against true future, response energy, zero-response floor and tree eligibility. Predicting a model response or failure does not establish that response is correct or semantically shared.',
        resources=dict(cpu_threads=1,model_seconds=900,resident_bytes=3*2**30,serial_native_workers=True,gpu=False),
        checkpoints=checkpoint,slot='slot2; user granted explicitly; slot1 belongs to natural pilot',
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files()},
        pilot_binding=dict(protocol_sha256=sha(PILOT/'protocol.json'),amendment_sha256=sha(PILOT/'amendment_01.json'),selections_sha256=sha(PILOT/'selections.json')),
        artifact_hashes={f:sha(out/f) for f in ['inputs.npz','targets.npz','rows.json','plans.json','fit_support.npz','fit_rows.json','frozen_probes.npz','probe_specs.json','probe_replay.json']})
    write(out/'protocol.json',protocol);write(out/'freeze.json',dict(sha256=sha(out/'protocol.json'),unix_time=time.time()))
    knowledge(out,dict(experiment='preparation',status='complete; no native inference yet',design=protocol,exact_findings=dict(probes=len(specs),max_fit_replay_error=max(r['max_prediction_error'] for r in replay),fit_sources=len({r['source'] for r in rs}),fresh_sources=64),controls=['Frozen original FIT only','all fresh seeds TEST-only','normalization matched'],limitations=['Synthetic only','exploratory extra arm'],next_decision='Run tests, then use granted slot2 for sequential model workers.'))
    print('Prepared',out,flush=True)


def verify(out,include_targets=True):
    """Verify immutable code/input bindings; workers never read future target contents."""
    p=read(out/'protocol.json');assert sha(out/'protocol.json')==read(out/'freeze.json')['sha256']
    for file,digest in p['code_hashes'].items():
        assert sha(ROOT/file)==digest,'Code binding changed: '+file
    for file,digest in p['artifact_hashes'].items():
        if not include_targets and file=='targets.npz':continue
        assert sha(out/file)==digest,'Artifact binding changed: '+file
    return p


def require_grant(out):
    """Native launch requires the explicit slot grant bound to this protocol."""
    path=out/'slot_grant.json'
    if not path.exists():raise RuntimeError('Native launch held: no slot grant')
    grant=read(path)
    if not grant.get('granted') or grant.get('slot')!=2 or grant.get('protocol_sha256')!=sha(out/'protocol.json'):
        raise RuntimeError('Native launch held: invalid slot grant')


def validate(out):
    """Run native-free known-answer and freeze/split regression tests before launch."""
    verify(out)
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_comparability_replication.py','-q'],cwd=ROOT,env=dict(os.environ,TSFM_TEST_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True)
    (out/'validation.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    write(out/'validation.json',dict(passed=result.returncode==0,returncode=result.returncode,code_hashes=read(out/'protocol.json')['code_hashes']))
    knowledge(out,dict(experiment='prelaunch_tests',status='passed' if result.returncode==0 else 'failed',artifact='validation.log',next_decision='Launch only if passed and slot grant valid.'))
    print(result.stdout,flush=True)
    if result.returncode:raise RuntimeError('Validation failed')


def worker(out,model):
    """Capture only declared native observables, keeping every future inaccessible to fitting."""
    require_grant(out);p=verify(out,include_targets=False)
    if not read(out/'validation.json')['passed']:raise RuntimeError('Validation missing')
    checkpoint=p['checkpoints'][model]
    for file,digest in checkpoint['weights'].items():assert sha(Path(checkpoint['path'])/file)==digest,'Checkpoint mismatch'
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import capture,patched
    from experiments.feature_search.coherence_study import normalization
    from experiments.feature_search.selective_tree import fuse
    started=time.monotonic();native=Native(model,checkpoint['path'])
    import torch
    torch.set_num_interop_threads(1)
    blocks=native.model.stacked_xf if model=='timesfm' else native.model.encoder.block
    cfg=p['capture'][model];sites={cfg['first']:blocks[int(cfg['first'])],cfg['half']:blocks[int(cfg['half'])]}
    with np.load(out/'inputs.npz') as f:x=f['history'];parts=f['parts']
    plans=read(out/'plans.json');hidden=[];forecasts=[];fused=[]
    gates=dict(repeat=0.,normalizer_identity=0.,first_self=0.,half_self=0.,input_closure=0.)
    for i,xx in enumerate(x):
        xx=xx[None];forecast,h=capture(native,xx,sites)
        assert list(h[cfg['half']][0].shape)==cfg['shape']
        repeat=native.predict(xx)[0]
        gates['repeat']=max(gates['repeat'],float(abs(repeat-forecast).max()))
        for label,cut in [('first_self',cfg['first']),('half_self',cfg['half'])]:
            replay=patched(native,xx,sites[cut],h[cut])
            gates[label]=max(gates[label],float(abs(replay-forecast).max()))
        with normalization(native,xx,'background'):
            locked=native.predict(xx)[0]
            _,zero=capture(native,np.zeros_like(xx),{cfg['first']:sites[cfg['first']]})
        gates['normalizer_identity']=max(gates['normalizer_identity'],float(abs(locked-forecast).max()))
        check_gates(gates)
        if plans[i]['accepted']:
            leaf=parts[i]
            closure=float(abs(leaf.sum(0)-xx[0]).max()/max(1,float(abs(xx).max())))
            gates['input_closure']=max(gates['input_closure'],closure);assert closure<=2e-6
            with normalization(native,np.repeat(xx,2,axis=0),'background'):
                _,leafh=capture(native,leaf,{cfg['first']:sites[cfg['first']]})
            state=fuse(leafh[cfg['first']],zero[cfg['first']][0],[1.,1.])
            response=patched(native,xx,sites[cfg['first']],state[None])
        else:response=forecast
        hidden.append(h[cfg['half']][0]);forecasts.append(forecast[0]);fused.append(response[0])
        if i%20==0:print(model,i,'/',len(x),round(time.monotonic()-started,1),flush=True)
        if time.monotonic()-started>900 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3*2**30:raise RuntimeError('Native resource cap')
    dest=out/f'{model}_native.npz'
    np.savez_compressed(dest,hidden=np.asarray(hidden),native=np.asarray(forecasts),fusion=np.asarray(fused))
    result=dict(model=model,rows=len(x),gates=gates,seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,artifact=dest.name,sha256=sha(dest))
    write(out/f'{model}_worker.json',result)
    knowledge(out,dict(experiment='native_'+model,status='complete',evidence_class='Instrumentation validity; outcomes not interpreted',exact_findings=result,next_decision='Continue sequential workers, then score all frozen probes.'))


def check_gates(gates):
    """Reject even tiny replay/normalizer errors rather than hiding them in a tolerance."""
    if any(gates[k]!=0 for k in ('repeat','normalizer_identity','first_self','half_self')):
        raise ValueError('Exact identity gate failed')
    if gates['input_closure']>2e-6:
        raise ValueError('Input closure failed')


def run(out):
    """Launch exactly one worker at a time with externally monitored RSS/time limits."""
    require_grant(out);verify(out)
    if not read(out/'validation.json')['passed']:raise RuntimeError('Validation gate')
    results={}
    for model in ('bolt','timesfm'):
        if (out/f'{model}_worker.json').exists():raise ValueError('Refuse repeated native run')
        started=time.monotonic();peak=0;reason=None
        with (out/f'{model}.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.comparability_replication','worker','--out',str(out),'--model',model],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text()
                    peak=max(peak,int(next(v.split()[1] for v in status.splitlines() if v.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>3*2**30 or time.monotonic()-started>900:
                    reason='3GiB/900-second cap';proc.kill();proc.wait();break
                time.sleep(.2)
        results[model]=dict(returncode=proc.returncode,seconds=time.monotonic()-started,peak_rss_bytes=peak,failure=reason)
        write(out/f'{model}_execution.json',results[model])
        if proc.returncode or reason:
            knowledge(out,dict(experiment='native_'+model,status='failed',exact_findings=results[model],artifact=model+'.log',next_decision='Stop; diagnose without loosening gates or silently rerunning.'))
            raise RuntimeError('Worker failed: '+model)
    write(out/'execution.json',results)


def source_permutation(rows,seed):
    """Permute entire fresh source bundles while matching all condition labels."""
    sources=sorted({r['source'] for r in rows});by={(r['source'],r['case']):i for i,r in enumerate(rows)}
    mapping=dict(zip(sources,np.random.default_rng(seed).permutation(sources)))
    return np.array([by[mapping[r['source']],r['case']] for r in rows])


def analyze(out):
    """Apply frozen coefficients to fresh sources; separate response prediction from correctness."""
    p=verify(out);rows=read(out/'rows.json');rs=read(out/'fit_rows.json');specs=read(out/'probe_specs.json')
    with np.load(out/'inputs.npz') as f:x=f['history'].astype(float)
    with np.load(out/'targets.npz') as f:future=f['future']
    plans=read(out/'plans.json');newinput=descriptors(x,plans)
    saved={}
    for m in ('bolt','timesfm'):
        assert sha(out/f'{m}_native.npz')==read(out/f'{m}_worker.json')['sha256']
        with np.load(out/f'{m}_native.npz') as f:saved[m]={k:f[k].astype(float) for k in f.files}
    support=dict(np.load(out/'fit_support.npz'));coef=dict(np.load(out/'frozen_probes.npz'))
    n=len(rs);tr=np.arange(n);te=np.arange(n,n+len(rows));w=weights(rs,tr)
    mu,sd=x.mean(1),x.std(1)
    results=[];allpred={};behavior=[]
    for a,b,ep in PRIMARY:
        features=dict(input=np.concatenate([support['input'],newinput]),
            A_hidden=np.concatenate([support[a+'_hidden'],saved[a]['hidden'].reshape(len(rows),-1)]),
            B_hidden=np.concatenate([support[b+'_hidden'],saved[b]['hidden'].reshape(len(rows),-1)]),
            A_forecast=np.concatenate([support[a+'_forecast'],(saved[a]['native']-mu[:,None])/sd[:,None]]))
        bank=kernel_bank(features['input'],gram_features(features),rs+rows,tr)
        yy=(saved[b]['native']-mu[:,None])/sd[:,None] if ep=='forecast' else (saved[b]['fusion']-saved[b]['native'])/sd[:,None]
        preds={};losses={};selected={}
        for spec in [s for s in specs if (s['A'],s['B'],s['endpoint'])==(a,b,ep)]:
            arm,key=spec['arm'],spec['key'];kk=selected_kernel(bank,arm,spec)
            np.testing.assert_allclose(kk[np.ix_(tr,tr)],coef[key+'__train_kernel'],atol=1e-9,rtol=1e-9)
            pred=kk[np.ix_(te,tr)]@coef[key+'__coef']+coef[key+'__mean']
            preds[arm]=pred;selected[arm]=spec;allpred[key]=pred
            losses[arm]=np.mean((pred-yy)**2,axis=1)
        contrast={}
        for label,base,extra in [('primary_gain','input','input+A_hidden'),('vs_Aforecast','input+A_forecast','input+A_hidden'),('exploratory_hidden_beyond_Aforecast','input+A_forecast',ARMS[-1])]:
            vals=group_values(losses[base]-losses[extra],rows,np.arange(len(rows)))
            contrast[label]=interval(list(vals.values()),coverage=1-.05/3 if label=='primary_gain' else .95)
        null={arm:[] for arm in ('input+A_hidden',ARMS[-1])}
        for seed in range(4952100,4952199):
            perm=source_permutation(rows,seed)+n
            for arm in null:
                spec=selected[arm];key=spec['key']
                if arm==ARMS[-1]:
                    cross=(2*bank['input+A_forecast'][spec['base_kernel']][np.ix_(te,tr)]+bank['A_hidden'][spec['hidden_kernel']][np.ix_(perm,tr)])/3
                    base='input+A_forecast'
                else:
                    inp,hid=spec['kernel'].split('+')
                    cross=(bank['input'][inp][np.ix_(te,tr)]+bank['A_hidden'][hid][np.ix_(perm,tr)])/2
                    base='input'
                pred=cross@coef[key+'__coef']+coef[key+'__mean']
                gain=losses[base]-np.mean((pred-yy)**2,axis=1)
                null[arm].append(float(np.mean(list(group_values(gain,rows,np.arange(len(rows))).values()))))
        avgloss={arm:interval(list(group_values(v,rows,np.arange(len(rows))).values())) for arm,v in losses.items()}
        gain=contrast['primary_gain'];nullvals=null['input+A_hidden']
        success=bool(gain['low']>0 and gain['mean']>=.05*avgloss['input']['mean'] and gain['mean']>np.quantile(nullvals,.95))
        result=dict(A=a,B=b,endpoint=ep,losses=avgloss,contrasts=contrast,relative_gain=gain['mean']/avgloss['input']['mean'],primary_replicates=success,
            shuffled_gains=null,shuffled_tail_rank=(1+sum(v>=gain['mean'] for v in nullvals))/100,target_energy=float(np.mean(yy**2)),n_sources=64)
        results.append(result)
    for b in ('bolt','timesfm'):
        native=np.mean((saved[b]['native']-future)**2,axis=1)/sd**2
        fused=np.mean((saved[b]['fusion']-future)**2,axis=1)/sd**2
        for case in ['ALL']+p['cases']:
            ids=np.array([i for i,r in enumerate(rows) if case=='ALL' or r['case']==case])
            behavior.append(dict(model=b,case=case,native_loss=interval(list(group_values(native[ids],rows,ids).values())),fusion_loss=interval(list(group_values(fused[ids],rows,ids).values())),fusion_gain=interval(list(group_values((native-fused)[ids],rows,ids).values())),eligible=sum(plans[i]['accepted'] for i in ids)))
    write(out/'results.json',results);write(out/'behavior_vs_semantics.json',behavior);np.savez_compressed(out/'predictions.npz',**allpred)
    knowledge(out,dict(experiment='fresh_replication',status='complete',evidence_class='Fresh-seed frozen-probe predictive replication; not causal cross-model or semantic agreement',design=dict(primary=p['primary'],source_count=64,resources=p['resources']),exact_findings=results,controls=dict(native_gates={m:read(out/f'{m}_worker.json')['gates'] for m in ('bolt','timesfm')},null=p['source_null'],behavior=behavior),limitations=['Same5 synthetic conditions; no real-data generalization.','Extra input+Aforecast+Ahidden arm is exploratory.','Predicting fusion response is distinct from semantic correctness.','Conditional bootstrap omits old FIT uncertainty; source shuffle is alignment diagnostic.'],next_decision='Interpret all primary contrasts and observable-only controls before proposing further data.'))
    print(json.dumps([{k:r[k] for k in ('A','B','endpoint','relative_gain','primary_replicates','shuffled_tail_rank')} for r in results],indent=2),flush=True)


def main():
    """One-core affinity; native memory is externally monitored, CPU analyses hard-capped."""
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','validate','run','worker','analyze']);parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--model',choices=['bolt','timesfm']);a=parser.parse_args()
    if a.command not in ('worker','run'):resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
    if a.command=='worker':worker(a.out,a.model)
    else:{'prepare':prepare,'validate':validate,'run':run,'analyze':analyze}[a.command](a.out)


if __name__=='__main__':main()
