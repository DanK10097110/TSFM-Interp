"""Reproducible live-model search, fresh-context evaluation and replay audit.

Run from tsfm_lens/: python experiments/dynamic_steering_study.py --help
This driver registers ordinary repository predictors. The search itself has
no model dispatch, layer names, native-head implementation or gradient access.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tsfm_lens.config import ModelConfig,DataConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.steering import DynamicSteerer,SearchConfig,SteeringModel,SteeringRequest,TASKS


def contexts(n,seed,length=128):
    rng=np.random.default_rng(seed);t=np.arange(length);result=[]
    for i in range(n):
        x=rng.uniform(.5,1.5)*np.sin(2*np.pi*t/24+rng.uniform(-np.pi,np.pi))
        x+=rng.uniform(-.3,.3)*t/length+rng.normal(0,.12,length)
        x+=rng.uniform(-.5,.5)*(t>rng.integers(30,100))
        result.append(x*rng.uniform(.5,2)+rng.uniform(-2,2))
    return np.asarray(result,dtype=np.float32)


def requests(suite,point=False):
    if suite=='context_windows':
        return [SteeringRequest('context_ignore',s,context_window='recent',forecast_window='long') for s in (.1,.3,1.)]+[
            SteeringRequest('context_emphasize',s,context_window='old',forecast_window='short') for s in (.1,.5)]+[
            SteeringRequest('context_focus',context_window='old',outside_weight=w,forecast_window='long') for w in (.8,0.)]
    if suite=='pilot':
        return [SteeringRequest('level',.1,forecast_window='short'),
                SteeringRequest('trend',.1,forecast_window='long'),
                SteeringRequest('seasonality',.1,period=24),
                SteeringRequest('context_ignore',.5,context_window='recent')]+(
                    [] if point else [SteeringRequest('uncertainty',.2)])
    result=[]
    for kind in TASKS:
        if point and kind in ('uncertainty','asymmetry','quantile'):continue
        kw=dict(period=24,value=0.,lower=-.5,upper=.5,quantile=.9)
        strength=.1 if kind not in ('context_ignore','context_focus','context_emphasize') else .5
        result.append(SteeringRequest(kind,strength,**kw))
    result += [SteeringRequest('level',.1,forecast_window=w) for w in ('short','long')]
    if not point:result += [SteeringRequest('uncertainty',s,forecast_window=w) for w in ('short','long') for s in (-.2,.2)]
    result += [SteeringRequest('context_ignore',1.,context_window='recent'),
               SteeringRequest('context_emphasize',.5,context_window='old'),
               SteeringRequest('context_focus',context_window='old',outside_weight=0.)]
    return result


def register(name,h,context_length=128):
    settings={'bolt':('chronos_bolt','amazon/chronos-bolt-small',{}),
              'timesfm':('timesfm','google/timesfm-2.5-200m-pytorch',{}),
              'chronos2':('chronos2','amazon/chronos-2',{}),
              'timer':('generic_hf','thuml/timer-base-84m',{'auto_class':'AutoModelForCausalLM'})}
    adapter,checkpoint,kwargs=settings[name]
    a=build_adapter(ModelConfig(name=name,adapter=adapter,checkpoint=checkpoint,kwargs=kwargs),
                    DataConfig(context_len=context_length,horizon=h),torch.device('cpu'),torch.float32)
    return SteeringModel.from_adapter(a,quantiles=[.5] if name=='timer' else [.1,.5,.9])


def run(args):
    torch.set_num_threads(2);started=time.monotonic()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=SearchConfig(max_candidates=args.candidates,rank=args.rank,calibration_samples=args.calibration,
        validation_samples=args.validation,radius=args.radius,refinement_steps=args.steps,
        max_seconds=args.seconds,max_calls=10000,probe_batch=args.batch,include_input=not args.internal_only,
        context_mode=args.context_mode,basis_strategy=args.basis)
    rs=requests(args.suite,args.model=='timer')
    protocol=dict(arguments=vars(args),config=asdict(cfg),requests=[asdict(r) for r in rs],
        data=f'independent synthetic seeds: calibration {args.train_seed}, validation {args.validation_seed}, test {args.test_seed}',
        evidence='bounded control demonstration, not forecast accuracy or semantic-feature discovery')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];arrays={}
    try:
        model=register(args.model,args.horizon,args.context);steerer=DynamicSteerer(model,cfg)
        print('loaded',args.model,'seconds',time.monotonic()-started,flush=True)
        train=contexts(max(2,args.calibration),args.train_seed,args.context)
        valid=contexts(args.validation,args.validation_seed,args.context)
        test=contexts(args.tests,args.test_seed,args.context)
        reports=steerer.fit(train,rs,prediction_length=args.horizon,validation_contexts=valid)
        (out/'discovery.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('discovered',len(reports),'seconds',time.monotonic()-started,flush=True)
        steerer.save(out/'profiles')
        for k,r in enumerate(rs):
            result=steerer.steer(test,r,prediction_length=args.horizon,auto_discover=False)
            replay=steerer.replay(test,result)
            for i,d in enumerate(result.diagnostics):
                rows.append(dict(request_index=k,context_index=i,kind=r.kind,window=r.forecast_window,
                    context_window=r.context_window,strength=r.strength,replay_max_abs=float(np.max(abs(replay[i]-result.forecast[i]))),**d))
            for name,value in [('baseline',result.baseline),('forecast',result.forecast),('target',result.requested_change)]:
                arrays[f'{k}_{name}']=value
            (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(kind=r.kind,window=r.forecast_window,errors=[round(d['relative_error'],4) for d in result.diagnostics],
                site=result.diagnostics[0]['site']['module'],seconds=round(time.monotonic()-started,1))),flush=True)
        np.savez_compressed(out/'forecasts.npz',contexts=test,**arrays)
        summary=dict(n=len(rows),successes=sum(r['achieved'] for r in rows),
            mean_error=float(np.mean([r['relative_error'] for r in rows])),
            replay_max_abs=max(r['replay_max_abs'] for r in rows),seconds=time.monotonic()-started,
            model_calls=model.calls,evaluated_rows=model.evaluated_rows)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        lines=['# Dynamic steering experiment findings','',json.dumps(summary,indent=2),'',
               '| Request | Forecast window | Successes | Mean relative error |','|---|---|---:|---:|']
        for k,r in enumerate(rs):
            group=[row for row in rows if row['request_index']==k]
            lines.append(f"| {r.kind} ({r.strength:g}) | {r.forecast_window} | {sum(v['achieved'] for v in group)}/{len(group)} | {np.mean([v['relative_error'] for v in group]):.4f} |")
        lines += ['', 'Selection used separate validation contexts. Test seeds were not used for discovery.',
                  'Success means <=10% error relative to the requested change, including all protected coordinates.',
                  'Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.',
                  'Input interventions, if selected, are explicitly identified in rows.json.',
                  'Failure is bounded to this search/rank/radius and does not establish impossibility.']
        (out/'findings.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(summary),flush=True)
    except Exception as error:
        (out/'findings.md').write_text(f'# Incomplete dynamic steering experiment\n\n{type(error).__name__}: {error}\n\nCompleted {len(rows)} test rows. Protocol and partial rows retained.\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--model',choices=['bolt','timesfm','chronos2','timer'],required=True)
    p.add_argument('--suite',choices=['pilot','full','context_windows'],default='pilot')
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--rank',type=int,default=128)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--context-mode',choices=['auto','discover','input_policy'],default='auto')
    p.add_argument('--basis',choices=['auto','random'],default='auto')
    p.add_argument('--train-seed',type=int,default=71);p.add_argument('--validation-seed',type=int,default=72)
    p.add_argument('--test-seed',type=int,default=73)
    p.add_argument('--candidates',type=int,default=4);p.add_argument('--calibration',type=int,default=1)
    p.add_argument('--validation',type=int,default=1);p.add_argument('--tests',type=int,default=1)
    p.add_argument('--radius',type=float,default=.2);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--seconds',type=float,default=240);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--internal-only',action='store_true')
    run(p.parse_args())
