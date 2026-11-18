"""Challenge native-head control at Bolt's complete final encoder output.

Control coordinates either edit the REG token, broadcast one shared direction
over all tokens with unit Frobenius normalization, or independently edit all
tokens. The actual frozen decoder and forecast head remain downstream. Native
full-model hooks audit every control geometry on the first context.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .steering_study import make_contexts
from ..steering import interval_shift,seasonal_shift,quantile_shift,steer_hidden,trust_region_direction
from ..steering.native import BoltHead


def run(args):
    """Test prospective token scopes with all nonlinear acceptance measured."""
    torch.set_num_threads(2)
    start=time.time()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(vars(args),site='encoder output after final norm; includes REG',
                  scope='context-specific internal control; original tokens plus edit; no labels',
                  radius_units='fraction of whole encoder-output Frobenius norm',
                  patterns=dict(shared='same direction over all tokens divided by sqrt(token count)',
                                reg='only REG token',full='independent per-token edits'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    head=BoltHead()
    model=head.model
    contexts,metadata=make_contexts(args.test,args.seed,shifted=args.shifted)
    rows,arrays=[],{}
    worst_identity=0.
    for i,(x,meta) in enumerate(zip(contexts,metadata)):
        batch=head.tensor(x[None])
        with torch.no_grad():
            enc,loc_scale,embeds,mask=model.encode(context=batch)
            decoder=model.decode(embeds,mask,enc)
            baseline=head.forward(decoder[:,0])[0].numpy()
            native=model(context=batch).quantile_preds[0,:,:32].T.numpy()
        original=enc.detach()
        n,d=enc.shape[1:]
        loc,scale=[t.numpy().item() for t in loc_scale]
        replay=baseline*scale+loc
        if not np.allclose(replay,native,atol=2e-6,rtol=2e-6):
            raise RuntimeError('upstream suffix does not reproduce native forecast')
        worst_identity=max(worst_identity,float(abs(replay-native).max()))
        for pattern in args.patterns:
            dimension=n*d if pattern=='full' else d
            def expand(z):
                if pattern=='full':
                    return z.reshape(1,n,d)
                if pattern=='shared':
                    return z.reshape(1,1,d).expand(1,n,d)/np.sqrt(n)
                weights=z.new_zeros((1,n,1));weights[:,-1]=1
                return weights*z.reshape(1,1,d)
            def forward(z):
                hidden=original+expand(z)
                query=model.decode(embeds,mask,hidden)
                return head.forward(query[:,0])[0]
            zero=torch.zeros(dimension)
            jac=torch.autograd.functional.jacobian(lambda z:forward(z).flatten(),zero,vectorize=True).numpy()
            targets=dict(widen=interval_shift(baseline,head.quantiles,args.dose),
                         season=seasonal_shift(x,32,head.quantiles,period=meta['period'],amplitude=args.dose),
                         upper=quantile_shift(32,head.quantiles,channel=.9,amount=args.dose))
            for task,target in targets.items():
                cap=args.radius*float(torch.linalg.vector_norm(enc))
                _,bound=trust_region_direction(jac,target.flatten(),cap)
                result=steer_hidden(forward,zero,target,max_norm=cap,steps=args.steps,
                                    ridge=args.ridge,tolerance=.005)
                row=dict(seed=meta['seed'],pattern=pattern,task=task,**result.metrics,
                         relative_hidden_norm=float(np.linalg.norm(result.delta)/torch.linalg.vector_norm(enc)),
                         crossing_fraction=float(np.mean(np.diff(result.forecast,axis=-1)<0)),
                         first_order_bound=bound,history=result.history)
                rows.append(row)
                key=f'{i}_{pattern}_{task}'
                arrays[key+'_forecast']=result.forecast
                arrays[key+'_target']=target
                arrays[key+'_delta']=result.delta.astype('float32')
                if i==0:
                    addition=expand(head.tensor(result.delta))
                    hits=[]
                    def hook(module,inputs,output):
                        hits.append(1)
                        output.last_hidden_state=output.last_hidden_state+addition
                        return output
                    handle=model.encoder.register_forward_hook(hook)
                    try:
                        with torch.no_grad():
                            actual=model(context=batch).quantile_preds[0,:,:32].T.numpy()
                    finally:
                        handle.remove()
                    replay=result.forecast*scale+loc
                    error=float(abs(actual-replay).max())
                    worst_identity=max(worst_identity,error)
                    if len(hits)!=1 or not np.allclose(actual,replay,atol=1e-5,rtol=1e-5):
                        raise RuntimeError('upstream full-model edit audit failed')
        print(json.dumps(dict(completed=i+1,total=args.test,seconds=time.time()-start)),flush=True)
    summary={}
    for pattern in args.patterns:
        for task in targets:
            chosen=[r for r in rows if r['pattern']==pattern and r['task']==task]
            summary[f'{pattern}/{task}']=dict(error=float(np.mean([r['relative_error'] for r in chosen])),
                success=float(np.mean([r['relative_error']<.1 for r in chosen])),
                lower_bound=float(np.mean([r['first_order_bound']['relative_error'] for r in chosen])))
    result=dict(protocol=protocol,rows=rows,summary=summary,seconds=time.time()-start,identity_max_abs=worst_identity)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'interventions.npz',contexts=contexts,**arrays)
    (out/'findings.md').write_text('# Upstream steering findings\n\n'+json.dumps(summary,indent=2)+
        '\n\nErrors are relative to the entire requested quantile/time change.\n'
        'Bounds apply only to the first-order response in the declared token subspace.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--test',type=int,default=4)
    p.add_argument('--seed',type=int,default=5200000)
    p.add_argument('--dose',type=float,default=.2)
    p.add_argument('--radius',type=float,default=.05)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--patterns',nargs='+',choices=['reg','shared','full'],default=['reg','shared'])
    p.add_argument('--shifted',action='store_true')
    run(p.parse_args())
