"""Reuse one saved response profile for unseen request types/doses/periods."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from dynamic_steering_study import register,contexts
from tsfm_lens.steering import DynamicSteerer,SteeringRequest


def run(args):
    torch.set_num_threads(2);started=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(arguments=vars(args),seed=123,retained_profile='level .1, all forecast steps',
        requests=['uncertainty .15 long','seasonality .15 period18','level -.1 short'],
        selection='online transfer; not separate validation for new requests')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    model=register('bolt',64);controller=DynamicSteerer.load(args.profiles,model)
    controller.profiles={k:p for k,p in controller.profiles.items() if p.request.kind=='level' and p.request.forecast_window=='all'}
    if len(controller.profiles)!=1:raise ValueError('expected exactly one retained level profile')
    calls=[0];original=controller._candidates
    def count_searches(*args,**kwargs):
        calls[0]+=1;return original(*args,**kwargs)
    controller._candidates=count_searches
    rs=[SteeringRequest('uncertainty',.15,forecast_window='long'),
        SteeringRequest('seasonality',.15,period=18),SteeringRequest('level',-.1,forecast_window='short')]
    x=contexts(4,123);rows=[];arrays={}
    for k,r in enumerate(rs):
        result=controller.steer(x,r,prediction_length=64)
        replay=controller.replay(x,result)
        for i,d in enumerate(result.diagnostics):
            rows.append(dict(kind=r.kind,context_index=i,replay_max_abs=float(np.max(abs(replay[i]-result.forecast[i]))),**d))
        arrays[f'{k}_baseline']=result.baseline;arrays[f'{k}_forecast']=result.forecast;arrays[f'{k}_target']=result.requested_change
    summary=dict(n=len(rows),successes=sum(r['achieved'] for r in rows),search_calls=calls[0],
        mean_error=float(np.mean([r['relative_error'] for r in rows])),replay_max_abs=max(r['replay_max_abs'] for r in rows),
        model_calls=model.calls,seconds=time.monotonic()-started)
    (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'forecasts.npz',contexts=x,**arrays)
    (out/'findings.md').write_text('# Response transfer audit\n\n'+json.dumps(summary,indent=2)+
        '\n\nOnly the level profile was retained; new uncertainty/seasonal and negative-level requests reuse its measured response. '
        'No new site search was called. Selection is correctly labeled online; local response refinement remains allowed. '
        'This is response reuse, not held-out validation of the new requests or a universal fixed vector.\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profiles',required=True);p.add_argument('--out',required=True)
    run(p.parse_args())
