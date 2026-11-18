"""Frozen reuse-only learned-versus-random feature-efficiency follow-up.

Random maps consume full history plus its normalizer, not model internals.
Readout widths, PCA ranks, ridge grid and old-FIT folds match the trained-hidden
extension. Families and three initializations are never selected using TEST.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from scipy.linalg import solve
from experiments.feature_search.comparability_core import (ALPHAS,descriptors,folds,group_values,interval,kernel_bank,pca_kernel,ridge,stable_argmin,weights)
from experiments.feature_search.comparability_pilot import ROOT,read,write,sha

UP=ROOT/'runs/comparability_replication4952000'
AUDIT=ROOT/'runs/comparability_replication_audit4952100'
DEFAULT=ROOT/'runs/comparability_random4953000'
SEEDS=(4953000,4953001,4953002)
FAMILIES=('rff','mlp_relu')
TARGETS=(('bolt','timesfm','forecast'),('timesfm','bolt','forecast'),('timesfm','bolt','fusion_response'))


def full_input(x):
    """Retain all128 normalized times and the two history-only normalizer descriptors."""
    if x.shape[1]!=258:raise ValueError('Unexpected frozen descriptor schema')
    return np.column_stack([x[:,:128],x[:,234:236]])


def random_features(x,train,w,family,seed,width):
    """Fixed random map; standardization and RFF bandwidth use only the supplied FIT rows."""
    xx=full_input(x);mean=w@xx[train];sd=np.sqrt(w@((xx[train]-mean)**2));sd[sd<1e-8]=1
    z=(xx-mean)/sd;rng=np.random.default_rng(seed)
    projection=rng.normal(size=(width,z.shape[1]))
    if family=='rff':
        fit=z[train];raw=fit@fit.T
        distances=np.maximum(np.diag(raw)[:,None]+np.diag(raw)[None,:]-2*raw,0)
        bandwidth=max(float(np.median(distances[np.triu_indices(len(train),1)])),1e-10)
        phase=rng.uniform(0,2*np.pi,size=width)
        feature=np.sqrt(2/width)*np.cos(np.sqrt(2/bandwidth)*(z@projection.T)+phase)
    elif family=='mlp_relu':
        bias=rng.normal(size=width)
        feature=np.maximum(np.sqrt(2/z.shape[1])*(z@projection.T)+bias,0)
        bandwidth=None
    else:raise ValueError('Unknown random map')
    return feature,dict(width=width,input_dimensions=z.shape[1],random_parameters=projection.size+width,bandwidth=bandwidth,seed=seed,family=family)


def fit_targets(rows):
    """Read only old FIT labels from legacy captures, never fresh64 TEST outcomes."""
    result={};cache={}
    for origin in sorted({r['run'] for r in rows}):
        p=ROOT/'runs'/origin
        with np.load(p/'inputs.npz') as f:history=f['history'].astype(float)
        for model in ('bolt','timesfm'):
            records=read(p/f'{model}_forecasts.json')
            native={r['row']:np.asarray(r['forecast'],float) for r in records if r['method']=='native'}
            fusion={r['row']:np.asarray(r['forecast'],float) for r in records if r['method']=='fusion' and r['depth']==1 and r['cut']=='0' and r['weight']=='equal' and r['blend']==1}
            cache[origin,model]=(history,native,fusion)
    for a,b,ep in TARGETS:
        target=[]
        for r in rows:
            assert r['split']=='fit'
            history,native,fusion=cache[r['run'],b];i=r['local_row'];xx=history[i]
            target.append((native[i]-xx.mean())/xx.std() if ep=='forecast' else (fusion.get(i,native[i])-native[i])/xx.std())
        result[f'{a}_to_{b}__{ep}']=np.asarray(target)
    return result


def owned_sources():
    """Bind new code/tests and immutable readout dependencies."""
    return [Path(__file__),ROOT/'tests/test_comparability_random_features.py',ROOT/'experiments/feature_search/comparability_random_audit.py',ROOT/'experiments/feature_search/comparability_core.py',ROOT/'experiments/feature_search/comparability_pilot.py']


def record_event(out,event):
    """Persist preparation/control/fitting history after each completed step."""
    data=read(out/'knowledge.json');data.setdefault('events',[]).append(dict(unix_time=time.time(),**event))
    (out/'knowledge.json').write_text(__import__('json').dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(out):
    """Freeze control recipes and a held-native new-condition proposal before scoring TEST."""
    if not read(AUDIT/'audit_summary.json')['passed']:raise RuntimeError('Prior replication audit not passed')
    out.mkdir(parents=True,exist_ok=False)
    specs=read(UP/'probe_specs.json');selected=[s for s in specs if s['arm'] in ('input','input+A_forecast','input+A_forecast+A_hidden')]
    upstream=['protocol.json','probe_specs.json','fit_rows.json','fit_support.npz','inputs.npz','plans.json','rows.json','bolt_native.npz','timesfm_native.npz','predictions.npz','results.json']
    protocol=dict(version=1,question='Does the trained hidden block provide feature efficiency beyond equally budgeted untrained nonlinear maps of full input?',
        evidence='Post-hoc exploratory predictive follow-up on the already evaluated64-source cohort. Historical results cannot be made unseen by freezing this new analysis. No new native inference or new information beyond deterministic input transforms is claimed.',
        fit='Exactly oldFIT20 source groups/96 rows. Same4 source-grouped folds, source weights, stable tie rule and FIT-only preprocessing as trained extension. No fresh64 target enters selection.',
        targets=[dict(A=a,B=b,endpoint=e) for a,b,e in TARGETS],families=list(FAMILIES),initialization_seeds=list(SEEDS),
        maps=dict(input='All128 normalized history values plus history mean/SD and logSD;130 values recover full physical history. FIT-only coordinate standardization of input axes.',rff='D cosine features, Gaussian projection variance2/FIT median squared distance (fixed gamma1), random uniform phase, sqrt(2/D) amplitude.',mlp_relu='One untrained affine/ReLU hidden layer of widthD, Gaussian He weights and N(0,1) biases. No learned map weights or nonlinear-family tuning.'),
        budget=dict(widths=dict(bolt=4608,timesfm=5120),PCA_ranks=[8,16],alpha_grid=list(ALPHAS),candidates_per_init=8,initializations=3,source_budget=20,row_budget=96,
            scope='Matches flattened feature width, supervised readout rank/grid/folds/labels; does NOT match pretrained network architecture, parameter count, inference FLOPs or pretraining compute.'),
        base='Freeze original input+Aforecast kernel separately for each target; retain its input and Aforecast blocks. Replace added trained-hidden kernel by random-feature PCA kernel. All3 blocks equal weight1/3. Random PCA rank/alpha selected per family/init using only oldFIT, same8 choices as trained extension.',
        report='Report every initialization, no best seed or family. For each of2 families x3 targets, average squared loss across the3 fixed initializations (NOT prediction ensembling), then average cases within source. Report random-minus-trained loss and gain over input+Aforecast, paired source bootstrap4000 draws. Six trained-advantage intervals use99.16666666666667% coverage; individual-init intervals95%, all exploratory.',
        interpretation='A trained advantage is conditional feature/readout efficiency under these recipes, not information-theoretic independence, causality or semantic equivalence. Random competitiveness is counterevidence to uniqueness; failure to reject difference is not equivalence.',
        frozen_arm_review=selected,
        constraints=dict(native_launch='HOLD; no native entry point',cpu_threads=1,address_space_bytes=3*2**30),
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in owned_sources()},upstream_hashes={f:sha(UP/f) for f in upstream},audit_sha256=sha(AUDIT/'audit_summary.json'))
    write(out/'protocol.json',protocol);write(out/'freeze.json',dict(sha256=sha(out/'protocol.json'),unix_time=time.time()))
    proposal=dict(status='PROPOSAL ONLY; HOLD NATIVE; no slot requested or consumed',question='Separate out-of-condition extrapolation from within-condition feature efficiency.',
        synthetic_design=dict(fit_seeds=list(range(4954000,4954020)),test_seeds=list(range(4954020,4954052)),history=128,horizon=32,
            conditions=[dict(name='anchor',frequencies=[5.35,17.35],amplitudes=[.5,8.]),dict(name='weaker',frequencies=[5.35,17.35],amplitudes=[.5,2.]),dict(name='shifted_frequency',frequencies=[7.35,23.35],amplitudes=[.5,8.]),dict(name='shifted_weaker',frequencies=[7.35,23.35],amplitudes=[.5,2.]),dict(name='shifted_balanced',frequencies=[7.35,23.35],amplitudes=[.5,.5])],
            rows=260,fit_budget='20 source groups,96 fitted rows: omit shifted_balanced for first4 FIT seeds to match original missing balanced condition count. All32 TEST sources retain all5 conditions. No TEST choice.',
            capture='Only native, fixed first-block depth1 fusion, and matching half-depth hidden; identical normalizer/replay/source/checkpoint gates. Serial1thread,3GiB,900s/model caps.'),
        analysis=[dict(axis='extrapolation',procedure='Apply every original oldFIT frozen trained/random probe without refitting to all new TEST conditions; anchor separates seed replication from condition shift.'),dict(axis='feature_efficiency',procedure='Separately fit every representation/readout on only the new20 FIT sources using equal96-row budget and frozen search grids, then compare on the same32 held-out TEST sources. Never mix fitted and frozen-arm conclusions.')],
        controls='Strong full-history/RBF input, input+Aforecast, trained hidden, both random families/all3 seeds; source dependence, initialization variability and response energy retained.',
        interpretation='Frozen failure with local-FIT recovery points to extrapolation limits. Trained advantage after equal local FIT is conditional feature efficiency. Neither pattern establishes causal or equivalent semantics.',
        natural_leg='Not pooled with synthetic shifts. A later separately frozen natural-context cohort needs source-disjoint FIT/TEST with substantially more than4 TEST sources/domain. Existing real4 cannot establish that expansion succeeds.',
        decision='Do not launch from this proposal. Review random controls first, then freeze any approved native design before capture.')
    write(out/'new_condition_proposal.json',proposal)
    write(out/'knowledge.json',dict(question=protocol['question'],design=protocol,exact_findings={'status':'protocol frozen; no new TEST scoring'},controls=['Completed25-plant/60-interval upstream audit','Unchanged original primary arms'],limitations=[protocol['evidence'],protocol['budget']['scope']],evidence_class='Prepared post-hoc predictive feature-efficiency follow-up',next_decision='Validate planted controls, fit oldFIT only, then score the frozen random recipes using cached data. Native proposal remains held.'))


def verify(out):
    """Reject any altered protocol, source, or original evidence."""
    p=read(out/'protocol.json');assert sha(out/'protocol.json')==read(out/'freeze.json')['sha256']
    for file,digest in p['code_hashes'].items():assert sha(ROOT/file)==digest,file
    for file,digest in p['upstream_hashes'].items():assert sha(UP/file)==digest,file
    return p


def validate(out):
    """Run new known-answer and regression plants before opening replication TEST values."""
    verify(out)
    run=subprocess.run([sys.executable,'-m','pytest','tests/test_comparability_random_features.py','-q'],cwd=ROOT,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TSFM_TEST_THREADS='1'),capture_output=True,text=True)
    (out/'validation.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    write(out/'validation.json',dict(passed=run.returncode==0,returncode=run.returncode))
    record_event(out,dict(experiment='validation',passed=run.returncode==0,artifact='validation.log'))
    print(run.stdout,flush=True)
    if run.returncode:raise RuntimeError('Validation failed')


def support():
    """Read only serialized oldFIT features and metadata."""
    with np.load(UP/'fit_support.npz') as f:arrays={k:f[k] for k in f.files}
    rows=read(UP/'fit_rows.json');assert len(rows)==96 and len({r['source'] for r in rows})==20
    return arrays,rows


def choose(candidates,y,rows,fold_banks):
    """Select rank and ridge using exactly source-grouped oldFIT risks."""
    scores=[]
    for rank,alpha in candidates:
        loss=[];indices=[]
        for tr,va,base,random in fold_banks:
            kernel=(2*base+random[rank])/3
            pred=ridge(kernel,y,tr,va,weights(rows,tr),alpha)
            loss.extend(np.mean((pred-y[va])**2,axis=1));indices.extend(va)
        scores.append(float(np.mean(list(group_values(loss,rows,indices).values()))))
    picked=stable_argmin(scores)
    return dict(rank=candidates[picked][0],alpha=candidates[picked][1],cv_loss=scores[picked],all_cv_losses=scores)


def fit(out):
    """Fit18 random readouts without loading any fresh64 outcome arrays."""
    p=verify(out)
    if not read(out/'validation.json')['passed']:raise RuntimeError('Validation gate')
    if not read(out/'plant_audit.json')['passed']:raise RuntimeError('Planted regression gate')
    arrays,rows=support();x=arrays['input'];tr=np.arange(len(rows));w=weights(rows,tr);sw=np.sqrt(w)
    y=fit_targets(rows);frozen=read(UP/'probe_specs.json');specs=[];coefficients={};started=time.monotonic()
    candidates=[(rank,alpha) for rank in (8,16) for alpha in ALPHAS]
    for a in ('bolt','timesfm'):
        partitions=folds(rows,tr)+[(tr,tr)]
        baseline_banks=[kernel_bank(x,{'A_forecast':arrays[a+'_forecast']@arrays[a+'_forecast'].T},rows,t) for t,v in partitions]
        for family in FAMILIES:
            for seed in SEEDS:
                random_banks=[];meta=None
                for (t,v),base in zip(partitions,baseline_banks):
                    ww=weights(rows,t);features,meta=random_features(x,t,ww,family,seed,p['budget']['widths'][a]);gram=features@features.T
                    cap=len({rows[i]['source'] for i in t})-1
                    random_banks.append({rank:pca_kernel(gram,t,ww,min(rank,cap)) for rank in (8,16)})
                for aa,b,ep in TARGETS:
                    if aa!=a:continue
                    target=f'{a}_to_{b}__{ep}'
                    base_spec=next(s for s in frozen if s['A']==a and s['B']==b and s['endpoint']==ep and s['arm']=='input+A_forecast')
                    key=f'{target}__{family}__{seed}'
                    cv=[(t,v,base['input+A_forecast'][base_spec['kernel']],random) for (t,v),base,random in zip(partitions[:-1],baseline_banks[:-1],random_banks[:-1])]
                    choice=choose(candidates,y[target],rows,cv)
                    kk=(2*baseline_banks[-1]['input+A_forecast'][base_spec['kernel']]+random_banks[-1][choice['rank']])/3
                    mean=w@y[target];coef=sw[:,None]*solve(kk*sw[:,None]*sw[None,:]+choice['alpha']*np.eye(len(tr)),(y[target]-mean)*sw[:,None],assume_a='pos')
                    coefficients[key+'__coef']=coef;coefficients[key+'__mean']=mean;coefficients[key+'__train_kernel']=kk
                    specs.append(dict(key=key,A=a,B=b,endpoint=ep,family=family,seed=seed,base_kernel=base_spec['kernel'],feature_recipe=meta,**choice))
        print('oldFIT random probes complete',a,round(time.monotonic()-started,1),flush=True)
    np.savez_compressed(out/'frozen_random_probes.npz',**coefficients);write(out/'probe_specs.json',specs)
    write(out/'fit_complete.json',dict(seconds=time.monotonic()-started,probes=len(specs),fit_sources=20,fit_rows=96,hashes={f:sha(out/f) for f in ('frozen_random_probes.npz','probe_specs.json')},fresh_test_values_opened=False))
    record_event(out,dict(experiment='oldFIT_random_probes',status='complete',exact_findings=read(out/'fit_complete.json'),controls=read(out/'plant_audit.json'),next_decision='Cached TEST evaluation permitted; native remains HOLD.'))


def evaluate(out):
    """Score all frozen maps without selecting a family, initialization or new parameter."""
    started=time.monotonic();p=verify(out)
    for f,h in read(out/'fit_complete.json')['hashes'].items():assert sha(out/f)==h
    arrays,fitrows=support();rows=read(UP/'rows.json');plans=read(UP/'plans.json');n=len(fitrows);tr=np.arange(n);te=np.arange(n,n+len(rows))
    with np.load(UP/'inputs.npz') as f:history=f['history'].astype(float)
    xx=np.concatenate([arrays['input'],descriptors(history,plans)])
    native={}
    for m in ('bolt','timesfm'):
        with np.load(UP/f'{m}_native.npz') as f:native[m]={k:f[k].astype(float) for k in ('native','fusion')}
    with np.load(UP/'predictions.npz') as f:original={k:f[k] for k in f.files}
    with np.load(out/'frozen_random_probes.npz') as f:coefs={k:f[k] for k in f.files}
    specs=read(out/'probe_specs.json');all_predictions={};per_seed=[];all_losses={};replay_error=0.
    sd=history.std(1);mean=history.mean(1)
    for a in ('bolt','timesfm'):
        af=np.concatenate([arrays[a+'_forecast'],(native[a]['native']-mean[:,None])/sd[:,None]])
        bank=kernel_bank(xx,{'A_forecast':af@af.T},fitrows+rows,tr)
        for family in FAMILIES:
            for seed in SEEDS:
                features,_=random_features(xx,tr,weights(fitrows,tr),family,seed,p['budget']['widths'][a]);gram=features@features.T
                kernels={rank:pca_kernel(gram,tr,weights(fitrows,tr),rank) for rank in (8,16)}
                for spec in [s for s in specs if s['A']==a and s['family']==family and s['seed']==seed]:
                    key,b,ep=spec['key'],spec['B'],spec['endpoint'];prefix=f'{a}_to_{b}__{ep}'
                    kk=(2*bank['input+A_forecast'][spec['base_kernel']]+kernels[spec['rank']])/3
                    err=float(abs(kk[np.ix_(tr,tr)]-coefs[key+'__train_kernel']).max());assert err<1e-9
                    replay_error=max(replay_error,err)
                    prediction=kk[np.ix_(te,tr)]@coefs[key+'__coef']+coefs[key+'__mean'];all_predictions[key]=prediction
                    target=(native[b]['native']-mean[:,None])/sd[:,None] if ep=='forecast' else (native[b]['fusion']-native[b]['native'])/sd[:,None]
                    losses={name:np.mean((pred-target)**2,axis=1) for name,pred in [('random',prediction),('base',original[prefix+'__input+A_forecast']),('trained',original[prefix+'__input+A_forecast+A_hidden'])]}
                    all_losses[key]=losses
                    stats={name:interval(list(group_values(v,rows,np.arange(len(rows))).values())) for name,v in losses.items()}
                    gain=interval(list(group_values(losses['base']-losses['random'],rows,np.arange(len(rows))).values()))
                    advantage=interval(list(group_values(losses['random']-losses['trained'],rows,np.arange(len(rows))).values()))
                    per_seed.append(dict(A=a,B=b,endpoint=ep,family=family,seed=seed,losses=stats,random_gain_over_base=gain,trained_advantage=advantage))
    summaries=[]
    for a,b,ep in TARGETS:
        for family in FAMILIES:
            values=[all_losses[f'{a}_to_{b}__{ep}__{family}__{seed}'] for seed in SEEDS]
            averaged=np.mean([v['random'] for v in values],axis=0);trained=values[0]['trained'];base=values[0]['base']
            summary=dict(A=a,B=b,endpoint=ep,family=family,initializations=list(SEEDS),aggregation='Mean squared loss across3 fixed initializations, then source mean; not an ensemble of predictions.',
                random_loss=interval(list(group_values(averaged,rows,np.arange(len(rows))).values())),trained_loss=interval(list(group_values(trained,rows,np.arange(len(rows))).values())),base_loss=interval(list(group_values(base,rows,np.arange(len(rows))).values())),
                trained_advantage=interval(list(group_values(averaged-trained,rows,np.arange(len(rows))).values()),coverage=1-.05/6),random_gain_over_base=interval(list(group_values(base-averaged,rows,np.arange(len(rows))).values())))
            summaries.append(summary)
    np.savez_compressed(out/'predictions.npz',**all_predictions);write(out/'per_initialization.json',per_seed);write(out/'results.json',summaries)
    knowledge=read(out/'knowledge.json');knowledge.update(status='complete',exact_findings=summaries,per_initialization_artifact='per_initialization.json',controls=dict(validation=read(out/'validation.json'),plant_audit=read(out/'plant_audit.json'),fit=read(out/'fit_complete.json'),max_frozen_train_kernel_replay_error=replay_error,upstream_audit=str(AUDIT.relative_to(ROOT))),limitations=[p['evidence'],p['budget']['scope'],'Three fixed initializations do not quantify all random-feature families.','Source bootstrap conditions on original FIT and fixed initializations.'],evidence_class='Exploratory post-hoc predictive feature-efficiency comparison on previously evaluated cached sources.',next_decision='Review all random-family/seed results; the new-condition proposal remains native-HOLD. Original primary results remain intact.')
    (out/'knowledge.json').write_text(__import__('json').dumps(knowledge,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(out/'execution.json',dict(evaluation_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,native_calls=0,cpu_threads=1))
    print('Cached evaluation complete; no native inference.',flush=True)


def main():
    """Respect one CPU core and a hard3-GiB cap; expose no native launch command."""
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','validate','fit','evaluate']);p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args()
    {'prepare':prepare,'validate':validate,'fit':fit,'evaluate':evaluate}[a.command](a.out)


if __name__=='__main__':main()
