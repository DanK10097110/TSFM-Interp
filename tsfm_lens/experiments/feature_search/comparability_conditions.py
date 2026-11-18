"""Preregistered condition-shift study: prepare now, native capture only on a new grant.

Two separate estimands: old frozen-probe extrapolation and equal-budget local-FIT
feature efficiency. The primary is TimesFM->Bolt fusion response versus BOTH
untrained families; native response energy and correctness are separate outcomes.
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
from experiments.feature_search.comparability_core import (ALPHAS,descriptors,folds,group_values,interval,kernel_bank,pca_kernel,ridge,stable_argmin,weights)
from experiments.feature_search.comparability_pilot import ROOT,read,write,sha
from experiments.feature_search.comparability_random_features import random_features,SEEDS,FAMILIES

DEFAULT=ROOT/'runs/comparability_conditions4962000'
OLD=ROOT/'runs/comparability_replication4952000'
RANDOM=ROOT/'runs/comparability_random4953000'
TARGETS=(('bolt','timesfm','forecast'),('timesfm','bolt','forecast'),('timesfm','bolt','fusion_response'))
CONDITIONS=(('anchor',5.35,17.35,8.),('weaker',5.35,17.35,2.),('shifted_frequency',7.35,23.35,8.),('shifted_weaker',7.35,23.35,2.),('shifted_balanced',7.35,23.35,.5))
COVERAGE=.975
MIN_REDUCTION=.05


def knowledge(out,event):
    """Preserve preparation, audit, capture and analysis history in this new run only."""
    path=out/'knowledge.json'
    data=read(path) if path.exists() else dict(question='Does trained TimesFM hidden improve Bolt fusion-response prediction beyond both random families under matched local FIT?',evidence_class='Prepared prospective synthetic predictive study; native capture held.',events=[])
    data['events'].append(dict(unix_time=time.time(),**event))
    temporary=out/'knowledge.tmp';temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temporary,path)


def generate():
    """Use only4962000..4962051, sharing phases/noise within each five-condition source."""
    rows=[];histories=[];futures=[];t=np.arange(-128,32)
    for seed in range(4962000,4962052):
        rng=np.random.default_rng(seed);phi,psi=rng.uniform(-np.pi,np.pi,2);noise=rng.normal(0,.05,len(t))
        split='fit' if seed<4962020 else 'test'
        for case,a,b,amplitude in CONDITIONS:
            value=.5*np.sin(2*np.pi*a*t/128+phi)+amplitude*np.sin(2*np.pi*b*t/128+psi)+noise
            eligible=split=='fit' and not (seed<4962004 and case=='shifted_balanced')
            rows.append(dict(source=f'syn:{seed}',seed=seed,domain='synthetic',case=case,split=split,fit_eligible=eligible,index=len(rows),run='comparability_conditions4962000'))
            histories.append(value[:128]);futures.append(value[128:])
    return rows,np.asarray(histories),np.asarray(futures)


def fit_ids(rows):
    """Require disjoint whole sources and the exact20-source/96-row local budget."""
    fit=np.array([i for i,r in enumerate(rows) if r['fit_eligible']])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    assert len(fit)==96 and len({rows[i]['source'] for i in fit})==20
    assert len(test)==160 and len({rows[i]['source'] for i in test})==32
    assert not ({rows[i]['source'] for i in fit}&{rows[i]['source'] for i in test})
    assert all(rows[i]['split']=='fit' for i in fit)
    return fit,test


def primary_rule(trained_loss,random_losses,rows):
    """Both families must pass97.5% paired source CI and>=5% point reduction."""
    results={}
    for family in FAMILIES:
        loss=np.mean(np.asarray(random_losses[family]),axis=0)
        gain=interval(list(group_values(loss-trained_loss,rows,np.arange(len(rows))).values()),coverage=COVERAGE)
        random_mean=float(np.mean(list(group_values(loss,rows,np.arange(len(rows))).values())))
        relative=gain['mean']/random_mean if random_mean>0 else None
        passed=bool(random_mean>0 and gain['low']>0 and relative>=MIN_REDUCTION)
        results[family]=dict(gain=gain,random_mean_loss=random_mean,relative_reduction=relative,passed=passed)
    return dict(passed=all(v['passed'] for v in results.values()),families=results,coverage=COVERAGE,minimum_relative_reduction=MIN_REDUCTION)


def behavior_metrics(history,future,native,fusion):
    """Separate response energy from correctness via gain=2*correction_dot-energy."""
    variance=np.var(history.astype(float),axis=1);delta=fusion-native
    native_loss=np.mean((native-future)**2,axis=1)/variance
    fusion_loss=np.mean((fusion-future)**2,axis=1)/variance
    energy=np.mean(delta**2,axis=1)/variance
    dot=np.mean(delta*(future-native),axis=1)/variance
    gain=native_loss-fusion_loss
    np.testing.assert_allclose(gain,2*dot-energy,atol=1e-10,rtol=1e-10)
    return dict(native_loss=native_loss,fusion_loss=fusion_loss,correctness_gain=gain,response_energy=energy,correction_dot=dot)


def owned_sources():
    """Bind all live dependencies, tests and the isolated mutation harness."""
    folder=ROOT/'experiments/feature_search'
    names=['comparability_conditions.py','comparability_conditions_audit.py','comparability_core.py','comparability_pilot.py','comparability_random_features.py','comparability_replication.py','selective_study.py','selective_tree.py','coherence_study.py','real_background_poc.py','homogeneous_normalizer.py']
    return [folder/n for n in names]+[ROOT/'tests/test_comparability_conditions.py',ROOT/'tsfm_lens/extraction/hooks.py']


def upstream_files():
    """Bind old frozen trained/random probes and support without refitting them."""
    return [OLD/f for f in ('protocol.json','fit_support.npz','fit_rows.json','frozen_probes.npz','probe_specs.json')]+[RANDOM/f for f in ('protocol.json','frozen_random_probes.npz','probe_specs.json','plant_audit.json','arithmetic_replay.json')]


def prepare(out):
    """Freeze260 inputs, strict primary thresholds, source/checkpoint bindings and native HOLD."""
    from experiments.feature_search.selective_tree import decompose
    out.mkdir(parents=True,exist_ok=False);rows,x,y=generate();fit,test=fit_ids(rows)
    parts=[];plans=[]
    for xx in x:
        levels,details=decompose(xx);parts.append(levels[1] if len(levels)>1 else [xx,np.zeros_like(xx)])
        plans.append(dict(details=details,levels=[None]*len(levels),accepted=len(levels)>1))
    np.savez_compressed(out/'inputs.npz',history=x.astype(np.float32),parts=np.asarray(parts,np.float32))
    np.savez_compressed(out/'targets.npz',future=y)
    write(out/'rows.json',rows);write(out/'plans.json',plans)
    old=read(OLD/'protocol.json');specs=read(OLD/'probe_specs.json')
    bases=[s for s in specs if s['arm'] in ('input','input+A_forecast')]
    protocol=dict(version=1,status='PREPARED; HOLD NATIVE until a NEW slot grant for this protocol',namespace='4962000..4962051;4954000 namespace belongs to Carson and is not used',
        fit_seeds=list(range(4962000,4962020)),test_seeds=list(range(4962020,4962052)),rows=260,fit_rows=96,test_rows=160,
        conditions=[dict(name=n,frequencies=[a,b],amplitudes=[.5,c]) for n,a,b,c in CONDITIONS],
        omitted_fit='First4 FIT sources omit shifted_balanced from fitting only; capture all260 rows. The omitted4 FIT rows are not TEST or used for selection.',
        primary=dict(axis='local_FIT',A='timesfm',B='bolt',endpoint='fusion_response',aggregate='Within each of32 TEST seeds average all5 conditions equally, then average sources equally.',
            trained_arm='input+Aforecast+Ahidden',random_arms='input+Aforecast+random: RFF and ReLU-MLP, each average squared loss over all3 frozen initialization seeds; no best seed/family selection and no prediction ensemble.',
            family_size=2,paired_source_CI_coverage=COVERAGE,bootstrap_draws=4000,bootstrap_seed=4951000,
            required='BOTH random-minus-trained paired-source intervals must have lower>0 AND trained mean loss must be>=5% lower than EACH random-family mean loss.',
            minimum_relative_reduction=MIN_REDUCTION,relative_denominator='Each random-family mean loss; zero denominator is not scorable and cannot pass.',
            no_rescue='A pass in one family, a descriptive condition, a frozen-extrapolation arm, or a forecast control cannot rescue failure of this primary.'),
        axes=dict(local_FIT='Recompute preprocessing and fit readouts on exactly20 newFIT sources/96 rows; four whole-source folds choose only added rank/alpha. NewTEST labels never enter fitting.',
            frozen_extrapolation='Apply original oldFIT frozen trained/random coefficients and preprocessing support without updates; summarize ALL and each condition on the same32 new TEST sources. Descriptive axis, no success claim or tuning.',
            controls='Forecast targets Bolt->TimesFM and TimesFM->Bolt are descriptive controls. Local input-only and fixed input+Aforecast baseline are retained.'),
        local_fixed_base='Source model Aforecast values are native outputs, not a trainable model. Input+Aforecast kernel recipe is fixed to prior selected choices (rbf2.0+pca16 for forecast, rbf0.5+pca16 for fusion response). Refit centers/PCA/support on localFIT only. Baseline ridge alpha remains0.001. All learned/random additions use equal block weight1/3.',
        base_specs=bases,budget=dict(widths=dict(bolt=4608,timesfm=5120),ranks=[8,16],alphas=list(ALPHAS),initialization_seeds=list(SEEDS),random_families=list(FAMILIES),candidates_per_added_arm=8,fit_sources=20,fit_rows=96),
        maps='Identical frozen random_features implementation from4953000: full normalized128 history plus2 normalizer values, FIT standardization; RFF gamma1/median distance or untrained affine/ReLU; same widths. Readout budget matched, not pretraining FLOPs/parameters.',
        normalization='Identical history mean/SD targets and inverse for all arms. No physical-scale disadvantage to hidden baselines.',
        behavior='Report source means/CIs by ALL and condition for normalized response energy, exact-zero response fraction, tree eligibility, native/fused true-future MSE, gain, and correction dot. Verify gain=2*dot-energy. Emulating a response does not establish its correctness or semantic equivalence.',
        alignment_null='Omitted from this bounded primary; optional future reuse-only alignment audit cannot change the frozen primary decision.',
        resources=dict(cpu_threads=1,model_seconds=900,resident_bytes=3*2**30,serial_workers=True,gpu=False),capture=old['capture'],checkpoints=old['checkpoints'],gates=old['gates'],
        natural_leg='NONE; no natural-context expansion authorized or prepared.',
        evidence='Prospectively frozen synthetic predictive feature-efficiency comparison; no causal or equivalent-semantics claim, no information beyond deterministic full input.',
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in owned_sources()},upstream_hashes={str(p.relative_to(ROOT)):sha(p) for p in upstream_files()},
        artifact_hashes={f:sha(out/f) for f in ('inputs.npz','targets.npz','rows.json','plans.json')})
    write(out/'protocol.json',protocol);write(out/'freeze.json',dict(sha256=sha(out/'protocol.json'),unix_time=time.time()))
    knowledge(out,dict(experiment='prepare',status='complete; native held',design=protocol,exact_findings=dict(rows=len(rows),fit_rows=len(fit),test_rows=len(test),fit_sources=20,test_sources=32),controls=['Source disjointness','Frozen thresholds/bases/maps','No natural leg'],limitations=['No native outcomes captured yet','Local and extrapolation estimands remain separate'],next_decision='Run tests and isolated plants, then request a native slot.'))
    data=read(out/'knowledge.json');data.update(status='prepared_native_hold',design=protocol,exact_findings={'native_rows_captured':0,'fit_sources':20,'fit_rows':96,'test_sources':32,'test_rows':160},controls=['Pending tests/plants'],limitations=['Synthetic conditions only','Matched readout budget is not matched pretraining compute'],next_decision='Validate and request slot; never infer grant from prior slot2 authorization.')
    (out/'knowledge.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Prepared',out,flush=True)


def verify(out,targets=True):
    """Reject changed code, inputs, checkpoints metadata or original frozen probes."""
    p=read(out/'protocol.json');assert sha(out/'protocol.json')==read(out/'freeze.json')['sha256']
    for field in ('code_hashes','upstream_hashes'):
        for file,digest in p[field].items():assert sha(ROOT/file)==digest,file
    for file,digest in p['artifact_hashes'].items():
        if targets or file!='targets.npz':assert sha(out/file)==digest,file
    return p


def require_grant(out):
    """Reject missing, prior-study, invalid-slot or stale-protocol native grants."""
    path=out/'slot_grant.json'
    if not path.exists():raise RuntimeError('Native HOLD: new slot grant required')
    grant=read(path)
    if not grant.get('granted') or grant.get('slot') not in (1,2) or grant.get('protocol_sha256')!=sha(out/'protocol.json'):
        raise RuntimeError('Native HOLD: invalid or inherited grant')


def require_ready(out):
    """Both direct workers and the serial launcher require passed tests and isolated plants."""
    for name in ('validation.json','plant_audit.json'):
        record=read(out/name)
        if not record['passed'] or record.get('protocol_sha256')!=sha(out/'protocol.json'):raise RuntimeError('Prelaunch gate failed: '+name)


def validate(out):
    """Exercise native-free known-answer tests before any slot request."""
    verify(out)
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_comparability_conditions.py','-q'],cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TSFM_TEST_THREADS='1'),capture_output=True,text=True)
    (out/'validation.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    write(out/'validation.json',dict(passed=result.returncode==0,returncode=result.returncode,protocol_sha256=sha(out/'protocol.json'),test_sha256=sha(ROOT/'tests/test_comparability_conditions.py')))
    knowledge(out,dict(experiment='validation',status='passed' if result.returncode==0 else 'failed',artifact='validation.log',next_decision='Isolated plants before requesting slot.'))
    print(result.stdout,flush=True)
    if result.returncode:raise RuntimeError('Validation failed')


def worker(out,model):
    """Native capture matches the audited recipe; this function is never called in preparation."""
    require_grant(out);require_ready(out);p=verify(out,targets=False)
    cp=p['checkpoints'][model]
    for file,digest in cp['weights'].items():assert sha(Path(cp['path'])/file)==digest,'Checkpoint changed'
    from experiments.feature_search.real_background_poc import Native
    from experiments.feature_search.selective_study import capture,patched
    from experiments.feature_search.coherence_study import normalization
    from experiments.feature_search.selective_tree import fuse
    from experiments.feature_search.comparability_replication import check_gates
    started=time.monotonic();native=Native(model,cp['path'])
    import torch
    torch.set_num_interop_threads(1)
    seq=native.model.stacked_xf if model=='timesfm' else native.model.encoder.block;cfg=p['capture'][model]
    sites={cfg['first']:seq[int(cfg['first'])],cfg['half']:seq[int(cfg['half'])]}
    with np.load(out/'inputs.npz') as f:x=f['history'];parts=f['parts']
    plans=read(out/'plans.json');hidden=[];preds=[];fused=[];gates=dict(repeat=0.,normalizer_identity=0.,first_self=0.,half_self=0.,input_closure=0.)
    for i,xx in enumerate(x):
        xx=xx[None];pred,h=capture(native,xx,sites);assert list(h[cfg['half']][0].shape)==cfg['shape']
        gates['repeat']=max(gates['repeat'],float(abs(native.predict(xx)[0]-pred).max()))
        for label,cut in [('first_self',cfg['first']),('half_self',cfg['half'])]:gates[label]=max(gates[label],float(abs(patched(native,xx,sites[cut],h[cut])-pred).max()))
        with normalization(native,xx,'background'):
            locked=native.predict(xx)[0];_,zero=capture(native,np.zeros_like(xx),{cfg['first']:sites[cfg['first']]})
        gates['normalizer_identity']=max(gates['normalizer_identity'],float(abs(locked-pred).max()));check_gates(gates)
        if plans[i]['accepted']:
            leaf=parts[i];gates['input_closure']=max(gates['input_closure'],float(abs(leaf.sum(0)-xx[0]).max()/max(1,float(abs(xx).max()))));check_gates(gates)
            with normalization(native,np.repeat(xx,2,axis=0),'background'):_,hs=capture(native,leaf,{cfg['first']:sites[cfg['first']]})
            state=fuse(hs[cfg['first']],zero[cfg['first']][0],[1.,1.]);response=patched(native,xx,sites[cfg['first']],state[None])
        else:response=pred
        hidden.append(h[cfg['half']][0]);preds.append(pred[0]);fused.append(response[0])
        if i%20==0:print(model,i,'/',len(x),round(time.monotonic()-started,1),flush=True)
        if time.monotonic()-started>900 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3*2**30:raise RuntimeError('Resource cap')
    dest=out/f'{model}_native.npz';np.savez_compressed(dest,hidden=hidden,native=preds,fusion=fused)
    record=dict(model=model,rows=len(x),gates=gates,seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sha256=sha(dest))
    write(out/f'{model}_worker.json',record);knowledge(out,dict(experiment='native_'+model,status='complete',exact_findings=record,evidence_class='Instrumentation validity only',next_decision='Finish serial capture, then local FIT and frozen extrapolation.'))


def run(out):
    """Only an explicit new grant can start serial900-second/3-GiB monitored workers."""
    require_grant(out);require_ready(out);verify(out);execution={}
    for model in ('bolt','timesfm'):
        if (out/f'{model}_worker.json').exists():raise ValueError('Refuse native overwrite')
        started=time.monotonic();peak=0;failure=None
        with (out/f'{model}.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-m','experiments.feature_search.comparability_conditions','worker','--out',str(out),'--model',model],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:peak=max(peak,int(next(s.split()[1] for s in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024)
                except (FileNotFoundError,StopIteration):pass
                if peak>3*2**30 or time.monotonic()-started>900:failure='Resource cap';proc.kill();proc.wait();break
                time.sleep(.2)
        execution[model]=dict(returncode=proc.returncode,seconds=time.monotonic()-started,peak_rss_bytes=peak,failure=failure);write(out/f'{model}_execution.json',execution[model])
        if proc.returncode or failure:
            knowledge(out,dict(experiment='native_'+model,status='failed',exact_findings=execution[model],artifact=model+'.log',next_decision='Stop; preserve failed run and diagnose.'));raise RuntimeError('Native worker failed')
    write(out/'execution.json',execution)


def read_capture(out,ids):
    """Return only requested rows; local FIT callers request only the96 FIT observations."""
    result={}
    for m in ('bolt','timesfm'):
        assert sha(out/f'{m}_native.npz')==read(out/f'{m}_worker.json')['sha256']
        with np.load(out/f'{m}_native.npz') as f:result[m]={k:f[k][ids].astype(float) for k in ('hidden','native','fusion')}
    return result


def base_specs(p,a,b,ep):
    """Retrieve fixed source-forecast and input recipes; no local or TEST reselection."""
    return {s['arm']:s for s in p['base_specs'] if (s['A'],s['B'],s['endpoint'])==(a,b,ep)}


def choose_added(banks,y,rows):
    """Eight candidate choices, equal source folds, identical trained/random search budget."""
    candidates=[(rank,alpha) for rank in (8,16) for alpha in ALPHAS];scores=[]
    for rank,alpha in candidates:
        losses=[];indices=[]
        for tr,va,base,extra in banks:
            pp=ridge((2*base+extra[rank])/3,y,tr,va,weights(rows,tr),alpha)
            losses.extend(np.mean((pp-y[va])**2,axis=1));indices.extend(va)
        scores.append(float(np.mean(list(group_values(losses,rows,indices).values()))))
    pick=stable_argmin(scores)
    return dict(rank=candidates[pick][0],alpha=candidates[pick][1],cv_loss=scores[pick],cv_losses=scores)


def extra_kernels(x,h,rows,tr,family,seed,width):
    """Fold-local trained or random Gram/PCA with the same rank cap and global scaling."""
    w=weights(rows,tr);cap=len({rows[i]['source'] for i in tr})-1
    features=h if family=='trained' else random_features(x,tr,w,family,seed,width)[0]
    gram=features@features.T
    return {rank:pca_kernel(gram,tr,w,min(rank,cap)) for rank in (8,16)}


def fit_local(out):
    """Fit only newFIT labels and freeze local coefficients before TEST evaluation."""
    p=verify(out);require_ready(out);allrows=read(out/'rows.json');ids,_=fit_ids(allrows);rows=[allrows[i] for i in ids]
    with np.load(out/'inputs.npz') as f:history=f['history'][ids].astype(float)
    allplans=read(out/'plans.json');x=descriptors(history,[allplans[i] for i in ids]);capture=read_capture(out,ids)
    mean,sd=history.mean(1),history.std(1);support=dict(input=x);targets={}
    for m in ('bolt','timesfm'):
        support[m+'_hidden']=capture[m]['hidden'].reshape(len(rows),-1);support[m+'_forecast']=(capture[m]['native']-mean[:,None])/sd[:,None]
    for a,b,ep in TARGETS:targets[a,b,ep]=support[b+'_forecast'] if ep=='forecast' else (capture[b]['fusion']-capture[b]['native'])/sd[:,None]
    tr=np.arange(len(rows));w=weights(rows,tr);sw=np.sqrt(w);partitions=folds(rows,tr)+[(tr,tr)];specs=[];coeff={}
    for a in ('bolt','timesfm'):
        af=support[a+'_forecast'];banks=[kernel_bank(x,{'A_forecast':af@af.T},rows,t) for t,v in partitions]
        for aa,b,ep in TARGETS:
            if aa!=a:continue
            fixed=base_specs(p,a,b,ep);prefix=f'{a}_to_{b}__{ep}';y=targets[a,b,ep]
            definitions=[]
            for arm in ('input','input+A_forecast'):
                s=fixed[arm];definitions.append((arm,dict(kernel=s['kernel'],alpha=s['alpha']),banks[-1][arm][s['kernel']]))
            for family,seed in [('trained',None)]+[(f,s) for f in FAMILIES for s in SEEDS]:
                extras=[extra_kernels(x,support[a+'_hidden'],rows,t,family,seed,p['budget']['widths'][a]) for t,v in partitions]
                base_name=fixed['input+A_forecast']['kernel']
                cv=[(t,v,bank['input+A_forecast'][base_name],extra) for (t,v),bank,extra in zip(partitions[:-1],banks[:-1],extras[:-1])]
                selected=choose_added(cv,y,rows);selected.update(family=family,seed=seed,base_kernel=base_name)
                arm='trained' if family=='trained' else f'{family}_{seed}'
                definitions.append((arm,selected,(2*banks[-1]['input+A_forecast'][base_name]+extras[-1][selected['rank']])/3))
            for arm,s,kk in definitions:
                key=prefix+'__'+arm;mu=w@y;co=sw[:,None]*solve(kk*sw[:,None]*sw[None,:]+s['alpha']*np.eye(len(tr)),(y-mu)*sw[:,None],assume_a='pos')
                coeff[key+'__coef']=co;coeff[key+'__mean']=mu;coeff[key+'__train_kernel']=kk;specs.append(dict(key=key,A=a,B=b,endpoint=ep,arm=arm,**s))
    np.savez_compressed(out/'local_support.npz',**support);np.savez_compressed(out/'local_probes.npz',**coeff)
    write(out/'local_fit_rows.json',rows);write(out/'local_probe_specs.json',specs)
    binding={f:sha(out/f) for f in ('local_support.npz','local_probes.npz','local_fit_rows.json','local_probe_specs.json')}
    write(out/'local_fit_complete.json',dict(fit_rows=96,fit_sources=20,TEST_labels_used=False,hashes=binding))
    knowledge(out,dict(experiment='local_fit',status='complete',exact_findings=dict(probes=len(specs),sources=20,rows=96),controls=['TEST excluded before target construction','Fixed source-forecast recipe','Eight choices per added arm'],next_decision='Score once on32 TEST sources; frozen extrapolation remains separate.'))


def score_local(out):
    """Evaluate local probes and the two-family primary on untouched TEST targets."""
    p=verify(out)
    for file,digest in read(out/'local_fit_complete.json')['hashes'].items():assert sha(out/file)==digest
    allrows=read(out/'rows.json');_,test=fit_ids(allrows);rows=[allrows[i] for i in test];fitrows=read(out/'local_fit_rows.json')
    with np.load(out/'inputs.npz') as f:history=f['history'][test].astype(float)
    with np.load(out/'targets.npz') as f:future=f['future'][test]
    allplans=read(out/'plans.json');plans=[allplans[i] for i in test];newx=descriptors(history,plans);capture=read_capture(out,test)
    with np.load(out/'local_support.npz') as f:support={k:f[k] for k in f.files}
    with np.load(out/'local_probes.npz') as f:coeff={k:f[k] for k in f.files}
    specs=read(out/'local_probe_specs.json');n=len(fitrows);tr=np.arange(n);te=np.arange(n,n+len(rows));x=np.concatenate([support['input'],newx]);mu,sd=history.mean(1),history.std(1)
    predictions={};results=[];all_losses={}
    for a,b,ep in TARGETS:
        af=np.concatenate([support[a+'_forecast'],(capture[a]['native']-mu[:,None])/sd[:,None]])
        hidden=np.concatenate([support[a+'_hidden'],capture[a]['hidden'].reshape(len(rows),-1)])
        bank=kernel_bank(x,{'A_forecast':af@af.T},fitrows+rows,tr);target=(capture[b]['native']-mu[:,None])/sd[:,None] if ep=='forecast' else (capture[b]['fusion']-capture[b]['native'])/sd[:,None]
        losses={}
        for s in [s for s in specs if (s['A'],s['B'],s['endpoint'])==(a,b,ep)]:
            if s['arm'] in ('input','input+A_forecast'):kk=bank[s['arm']][s['kernel']]
            else:
                extras=extra_kernels(x,hidden,fitrows+rows,tr,s['family'],s['seed'],p['budget']['widths'][a])
                kk=(2*bank['input+A_forecast'][s['base_kernel']]+extras[s['rank']])/3
            key=s['key'];np.testing.assert_allclose(kk[np.ix_(tr,tr)],coeff[key+'__train_kernel'],atol=1e-9,rtol=1e-9)
            pred=kk[np.ix_(te,tr)]@coeff[key+'__coef']+coeff[key+'__mean'];predictions[key]=pred;losses[s['arm']]=np.mean((pred-target)**2,axis=1)
        all_losses[a,b,ep]=losses
        for case in ['ALL']+[c[0] for c in CONDITIONS]:
            ids=np.array([i for i,r in enumerate(rows) if case=='ALL' or r['case']==case]);summaries={arm:interval(list(group_values(v[ids],rows,ids).values())) for arm,v in losses.items()}
            results.append(dict(axis='local_FIT',A=a,B=b,endpoint=ep,case=case,losses=summaries,response_or_forecast_energy=float(np.mean(target[ids]**2))))
    primary_losses=all_losses['timesfm','bolt','fusion_response'];random={f:[primary_losses[f'{f}_{s}'] for s in SEEDS] for f in FAMILIES}
    primary=primary_rule(primary_losses['trained'],random,rows);write(out/'primary_result.json',primary)
    behavior=[]
    for m in ('bolt','timesfm'):
        values=behavior_metrics(history,future,capture[m]['native'],capture[m]['fusion'])
        for case in ['ALL']+[c[0] for c in CONDITIONS]:
            ids=np.array([i for i,r in enumerate(rows) if case=='ALL' or r['case']==case])
            behavior.append(dict(model=m,case=case,metrics={k:interval(list(group_values(v[ids],rows,ids).values())) for k,v in values.items()},exact_zero_response_fraction=float(np.mean(values['response_energy'][ids]==0)),tree_eligible=sum(plans[i]['accepted'] for i in ids),rows=len(ids)))
    np.savez_compressed(out/'local_predictions.npz',**predictions);write(out/'local_results.json',results);write(out/'behavior_correctness.json',behavior)
    knowledge(out,dict(experiment='local_FIT_TEST',status='complete',evidence_class='Prospectively frozen conditional predictive feature-efficiency comparison',exact_findings=primary,controls=dict(behavior_artifact='behavior_correctness.json',all_arms='local_results.json'),limitations=['Synthetic conditions only','No causal or equivalent semantics','Bootstrap conditions on localFIT and3 fixed initializations'],next_decision='Review BOTH-family result; frozen extrapolation cannot rescue a failed primary.'))


def score_frozen(out):
    """Apply all old trained/random probes without updating them; descriptive by condition."""
    from experiments.feature_search.comparability_replication import selected_kernel
    p=verify(out);allrows=read(out/'rows.json');_,test=fit_ids(allrows);rows=[allrows[i] for i in test];oldrows=read(OLD/'fit_rows.json');n=len(oldrows);tr=np.arange(n);te=np.arange(n,n+len(rows))
    with np.load(out/'inputs.npz') as f:history=f['history'][test].astype(float)
    plans=read(out/'plans.json');newx=descriptors(history,[plans[i] for i in test]);capture=read_capture(out,test);mu,sd=history.mean(1),history.std(1)
    with np.load(OLD/'fit_support.npz') as f:support={k:f[k] for k in f.files}
    with np.load(OLD/'frozen_probes.npz') as f:oldcoef={k:f[k] for k in f.files}
    with np.load(RANDOM/'frozen_random_probes.npz') as f:randomcoef={k:f[k] for k in f.files}
    olds=read(OLD/'probe_specs.json');randoms=read(RANDOM/'probe_specs.json');predictions={};results=[]
    x=np.concatenate([support['input'],newx])
    for a,b,ep in TARGETS:
        af=np.concatenate([support[a+'_forecast'],(capture[a]['native']-mu[:,None])/sd[:,None]])
        ah=np.concatenate([support[a+'_hidden'],capture[a]['hidden'].reshape(len(rows),-1)]);bh=np.concatenate([support[b+'_hidden'],capture[b]['hidden'].reshape(len(rows),-1)])
        bank=kernel_bank(x,{'A_forecast':af@af.T,'A_hidden':ah@ah.T,'B_hidden':bh@bh.T},oldrows+rows,tr)
        yy=(capture[b]['native']-mu[:,None])/sd[:,None] if ep=='forecast' else (capture[b]['fusion']-capture[b]['native'])/sd[:,None]
        losses={}
        for s in [s for s in olds if (s['A'],s['B'],s['endpoint'])==(a,b,ep)]:
            kk=selected_kernel(bank,s['arm'],s);key=s['key'];np.testing.assert_allclose(kk[np.ix_(tr,tr)],oldcoef[key+'__train_kernel'],atol=1e-9,rtol=1e-9)
            pp=kk[np.ix_(te,tr)]@oldcoef[key+'__coef']+oldcoef[key+'__mean'];predictions[key]=pp;losses[s['arm']]=np.mean((pp-yy)**2,axis=1)
        for s in [s for s in randoms if (s['A'],s['B'],s['endpoint'])==(a,b,ep)]:
            extra=extra_kernels(x,ah,oldrows+rows,tr,s['family'],s['seed'],p['budget']['widths'][a]);kk=(2*bank['input+A_forecast'][s['base_kernel']]+extra[s['rank']])/3;key=s['key']
            np.testing.assert_allclose(kk[np.ix_(tr,tr)],randomcoef[key+'__train_kernel'],atol=1e-9,rtol=1e-9)
            pp=kk[np.ix_(te,tr)]@randomcoef[key+'__coef']+randomcoef[key+'__mean'];predictions[key]=pp;losses[f"{s['family']}_{s['seed']}"]=np.mean((pp-yy)**2,axis=1)
        for case in ['ALL']+[c[0] for c in CONDITIONS]:
            ids=np.array([i for i,r in enumerate(rows) if case=='ALL' or r['case']==case])
            results.append(dict(axis='frozen_extrapolation_descriptive',A=a,B=b,endpoint=ep,case=case,losses={arm:interval(list(group_values(v[ids],rows,ids).values())) for arm,v in losses.items()}))
    np.savez_compressed(out/'frozen_predictions.npz',**predictions);write(out/'frozen_extrapolation.json',results)
    knowledge(out,dict(experiment='frozen_extrapolation',status='complete',evidence_class='Descriptive condition-shift generalization with unchanged old probes',exact_findings={'artifact':'frozen_extrapolation.json','condition_cells':len(results)},limitations=['Not the local-FIT primary','No TEST adaptation'],next_decision='Compare axes without conflating extrapolation and feature efficiency.'))


def main():
    """No implicit launch: prepare/validate are safe under the user’s native HOLD."""
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','validate','run','worker','fit_local','score_local','score_frozen']);parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--model',choices=['bolt','timesfm']);a=parser.parse_args()
    if a.command not in ('run','worker'):resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
    if a.command=='worker':worker(a.out,a.model)
    else:globals()[a.command](a.out)


if __name__=='__main__':main()
