"""Dose/radius tests separating local rank, conditioning and nonlinear error."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .steering_study import make_contexts
from ..steering import interval_shift,seasonal_shift,quantile_shift,steer_hidden,trust_region_direction,response_metrics
from ..steering.native import BoltHead
from ..steering.timesfm import TimesFMHead


def run(args):
    """Save a prospective protocol and each context's actual nonlinear outcomes."""
    start=time.time()
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(vars(args),scope='adaptive native-head control; not semantic vector or calibrated uncertainty',
                  success='relative error < 0.1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    head=(BoltHead if args.model=='bolt' else TimesFMHead)()
    context,meta=make_contexts(args.test,args.seed,shifted=args.shifted)
    cache=head.capture(context)
    rows,arrays=[],{}
    identity=0.
    for i,(h,base,x,m) in enumerate(zip(cache['hidden'],cache['normalized'],context,meta)):
        jac=head.jacobian(h)
        for dose in args.doses:
            targets=dict(widen=interval_shift(base,head.quantiles,dose),
                         narrow=interval_shift(base,head.quantiles,-dose),
                         season=seasonal_shift(x,32,head.quantiles,period=m['period'],amplitude=dose),
                         upper=quantile_shift(32,head.quantiles,channel=.9,amount=dose))
            for task,target in targets.items():
                for radius in args.radii:
                    cap=radius*np.linalg.norm(h)
                    _,bound=trust_region_direction(jac,target.flatten(),cap)
                    for steps in args.steps:
                        result=steer_hidden(head.forward,head.tensor(h),target,max_norm=cap,steps=steps,
                                             ridge=args.ridge,tolerance=.005)
                        change=result.forecast-base
                        metrics=response_metrics(change,target)
                        row=dict(seed=m['seed'],task=task,dose=dose,radius=radius,steps=steps,
                                 **metrics,first_order_bound=bound,
                                 relative_hidden_norm=float(np.linalg.norm(result.delta)/np.linalg.norm(h)),
                                 median_drift_rms=float(np.sqrt(np.mean(change[:,4]**2))),
                                 width_change=float(np.mean(change[:,-1]-change[:,0])),
                                 crossing_fraction=float(np.mean(np.diff(result.forecast,axis=-1)<0)),
                                 baseline_crossing_fraction=float(np.mean(np.diff(base,axis=-1)<0)),
                                 history=result.history)
                        rows.append(row)
                        key=f'{i}_{task}_{dose:g}_{radius:g}_{steps}'
                        arrays[key+'_forecast']=result.forecast
                        arrays[key+'_target']=target
                        arrays[key+'_delta']=result.delta.astype('float32')
                        if i<2 and dose==args.doses[0] and radius==args.radii[0] and steps==args.steps[-1]:
                            actual=head.predict_edited(x[None],result.delta[None])[0]
                            replay=result.forecast*cache['scale'][i]+cache['location'][i]
                            error=float(np.max(abs(actual-replay)))
                            identity=max(identity,error)
                            if not np.allclose(actual,replay,atol=1e-5,rtol=1e-5):
                                raise RuntimeError('full-model edited replay failed')
        (out/'progress.json').write_text(json.dumps(dict(completed=i+1,total=args.test,seconds=time.time()-start)))
        print(json.dumps(dict(completed=i+1,total=args.test,seconds=time.time()-start)),flush=True)
    summary={}
    for task in targets:
        for dose in args.doses:
            for radius in args.radii:
                for steps in args.steps:
                    subset=[r for r in rows if (r['task'],r['dose'],r['radius'],r['steps'])==(task,dose,radius,steps)]
                    summary[f'{task}/{dose:g}/{radius:g}/{steps}']=dict(n=len(subset),
                        error=float(np.mean([r['relative_error'] for r in subset])),
                        collateral=float(np.mean([r['collateral'] for r in subset])),
                        gain=float(np.mean([r['gain'] for r in subset])),
                        success=float(np.mean([r['relative_error']<.1 for r in subset])),
                        first_order_bound=float(np.mean([r['first_order_bound']['relative_error'] for r in subset])),
                        relative_hidden_norm=float(np.mean([r['relative_hidden_norm'] for r in subset])),
                        crossing_fraction=float(np.mean([r['crossing_fraction'] for r in subset])))
    result=dict(protocol=protocol,rows=rows,summary=summary,seconds=time.time()-start,
                identity=dict(capture=cache['replay_max_abs'],edited_full_model_max_abs=identity))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'interventions.npz',context=context,baseline=cache['normalized'],**arrays)
    lines=['# Nonlinear steering findings','',f'Model {args.model}; {args.test} fresh seeds; {result["seconds"]:.2f}s.',
           '', '| Task / dose / radius / steps | Error | Success | Linear lower bound |',
           '| --- | ---: | ---: | ---: |']
    for key,value in summary.items():
        lines.append(f'| {key} | {value["error"]:.6f} | {value["success"]:.3f} | {value["first_order_bound"]:.6f} |')
    lines+=['','The lower bound is local to the original Jacobian. Nonlinear edits may beat it.',
            'Steps are selected prospectively, with backtracking on model response only. No future labels.',
            'Width control is not calibration; a low error is not evidence for monosemantic neurons.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(seconds=result['seconds'],identity=result['identity'])),flush=True)


def main():
    """Short default proof; explicit sample counts for longer replications."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--model',choices=['bolt','timesfm'],default='bolt')
    p.add_argument('--test',type=int,default=8)
    p.add_argument('--seed',type=int,default=5190000)
    p.add_argument('--doses',nargs='+',type=float,default=[.05,.2])
    p.add_argument('--radii',nargs='+',type=float,default=[.2,.5])
    p.add_argument('--steps',nargs='+',type=int,default=[1,4])
    p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--shifted',action='store_true')
    args=p.parse_args()
    if args.test<1 or any(not 0<d<1 for d in args.doses) or min(args.radii)<=0 or min(args.steps)<1:
        p.error('invalid sample, dose, radius or iteration budget')
    run(args)


if __name__=='__main__':
    main()
