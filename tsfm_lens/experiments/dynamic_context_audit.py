"""Audit actual history sensitivity under explicit relevance policies.

Ignore/focus have a mathematically defined invariance. Emphasis changes the
input policy, but a nonlinear forecaster need not become more sensitive in
every direction; measure that rather than asserting it.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from dynamic_steering_study import register,contexts
from tsfm_lens.steering import DynamicSteerer,SteeringRequest
from tsfm_lens.steering.requests import window_mask


def run(args):
    torch.set_num_threads(2);started=time.monotonic()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    c=256 if args.model=='timer' else 128
    cases=[SteeringRequest('context_ignore',1.,context_window=w) for w in ('old','recent')]
    cases += [SteeringRequest('context_focus',context_window=w,outside_weight=0.) for w in ('old',(24,48))]
    cases += [SteeringRequest('context_emphasize',.5,context_window=w) for w in ('old','recent')]
    (out/'protocol.json').write_text(json.dumps(dict(arguments=vars(args),seed=91,context_length=c,
        perturbation='independent Gaussian selected-window perturbation with SD .2 context SD',
        requests=[asdict(r) for r in cases]),indent=2)+'\n')
    model=register(args.model,32,c);controller=DynamicSteerer(model)
    x=contexts(args.tests,91,c);rng=np.random.default_rng(92);rows=[];arrays={}
    for k,r in enumerate(cases):
        mask=window_mask(r.context_window,c,context=True)
        perturb=~mask if r.kind=='context_focus' else mask
        changed=x.copy();changed[:,perturb]+=rng.normal(size=(len(x),int(perturb.sum())))*.2*x.std(1,keepdims=True)
        first=controller.steer(x,r,prediction_length=32)
        second=controller.steer(changed,r,prediction_length=32)
        replay=controller.replay(x,first)
        for i in range(len(x)):
            baseline=float(np.linalg.norm(second.baseline[i]-first.baseline[i]))
            controlled=float(np.linalg.norm(second.forecast[i]-first.forecast[i]))
            rows.append(dict(case=k,context=i,kind=r.kind,context_window=r.context_window,
                baseline_sensitivity=baseline,controlled_sensitivity=controlled,
                ratio=controlled/baseline if baseline>1e-12 else None,
                input_relative_norm=first.diagnostics[i]['input_relative_norm'],
                replay_max_abs=float(np.max(abs(replay[i]-first.forecast[i])))))
        arrays[f'{k}_baseline']=first.baseline;arrays[f'{k}_forecast']=first.forecast
        arrays[f'{k}_perturbed_baseline']=second.baseline;arrays[f'{k}_perturbed_forecast']=second.forecast
        print(json.dumps(dict(kind=r.kind,window=r.context_window,ratios=[row['ratio'] for row in rows if row['case']==k])),flush=True)
        (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(out/'forecasts.npz',contexts=x,**arrays)
    invariance=[r for r in rows if r['kind']!='context_emphasize']
    emphasis=[r['ratio'] for r in rows if r['kind']=='context_emphasize' and r['ratio'] is not None]
    summary=dict(n=len(rows),invariance_cases=len(invariance),
        exact_invariance=sum(r['controlled_sensitivity']==0 for r in invariance),
        max_invariance_sensitivity=max(r['controlled_sensitivity'] for r in invariance),
        emphasis_mean_ratio=float(np.mean(emphasis)),emphasis_increases=sum(r>1 for r in emphasis),
        replay_max_abs=max(r['replay_max_abs'] for r in rows),seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'findings.md').write_text('# Context relevance audit\n\n'+json.dumps(summary,indent=2)+
        '\n\nIgnore/focus remove specified input dependence by construction and were checked on actual forecasts. '
        'This is input preprocessing, not learned attention steering. Emphasis amplifies input deviations; '
        'the nonlinear forecast response is not guaranteed to amplify every perturbation. No forecast accuracy claim.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,choices=['bolt','timesfm','timer'])
    p.add_argument('--out',required=True);p.add_argument('--tests',type=int,default=8)
    run(p.parse_args())
