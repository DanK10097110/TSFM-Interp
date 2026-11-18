"""Separate a continuation prior from the prediction functional it optimizes.

Paired tiny-model MSE/absolute-loss training exactly marginalizes surviving and
vanishing futures. This is a mechanism control, never pretrained-model attribution.
"""
from pathlib import Path
import argparse
import resource
import time
import numpy as np
from experiments.feature_search.prior_training import bank
from experiments.feature_search.coverage_training import make_model
from experiments.feature_search.coherence_study import write, sha


def absolute_risk(pred, tone, probability):
    """Expected absolute loss for a two-point conditional future distribution."""
    return probability*np.mean(np.abs(pred-tone),axis=-1)+(1-probability)*np.mean(np.abs(pred),axis=-1)


def median_distance(pred, tone, probability):
    """Distance to the full median set; the half-probability optimum is nonunique."""
    if probability<.5:
        return np.mean(np.abs(pred),axis=-1)
    if probability>.5:
        return np.mean(np.abs(pred-tone),axis=-1)
    low,high=np.minimum(0,tone),np.maximum(0,tone)
    return np.mean(np.maximum(low-pred,0)+np.maximum(pred-high,0),axis=-1)


def run(out,steps=400):
    """Hold histories, initializations and minibatches fixed across loss/prior arms."""
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    protocol=dict(train=512,test=256,steps=steps,batch=64,lr=.002,
                  seeds=[4963100,4963101],probabilities=[.25,.5,.75],
                  architectures=['mlp','transformer'],objectives=['mse','absolute'],
                  train_seed=4963200,test_seed=4963300,threads=1,
                  cap_seconds=180,cap_bytes=2*2**30,
                  expected_risk='Exactly marginalized two futures; no Monte Carlo branch noise',
                  adequacy='Descriptive optimization residuals retained; no failed arm discarded',
                  scope='Tiny training control only; actual pretrained objectives and priors are unknown',
                  source_sha=sha(__file__),model_sha=sha(Path(__file__).with_name('coverage_training.py')),
                  bank_sha=sha(Path(__file__).with_name('prior_training.py')))
    write(out/'protocol.json',protocol)
    x,y,_=bank(protocol['train_seed'],512);ex,ey,_=bank(protocol['test_seed'],256)
    arrays=dict(train_history=x,train_tone=y,test_history=ex,test_tone=ey)
    records=[];tx=torch.tensor(x);ty=torch.tensor(y)
    write(out/'knowledge.json',dict(status='running',question=__doc__,design=protocol))
    try:
        for arch in protocol['architectures']:
            for seed in protocol['seeds']:
                for p in protocol['probabilities']:
                    for objective in protocol['objectives']:
                        model=make_model(arch,seed);opt=torch.optim.Adam(model.parameters(),lr=.002)
                        rng=np.random.default_rng(seed);trace=[]
                        for step in range(steps):
                            ix=rng.integers(0,512,64);pred=model(tx[ix]);tone=ty[ix]
                            loss=((pred-p*tone)**2).mean() if objective=='mse' else (p*torch.abs(pred-tone)+(1-p)*torch.abs(pred)).mean()
                            opt.zero_grad();loss.backward();opt.step();trace.append(float(loss.detach()))
                            if time.monotonic()-start>180 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*2**30:
                                raise RuntimeError('Frozen training resource cap exceeded')
                        model.eval()
                        with torch.no_grad(): pred=model(torch.tensor(ex)).numpy().astype(float)
                        tone=ey.astype(float);energy=np.mean(tone**2,axis=-1);absscale=np.mean(np.abs(tone),axis=-1)
                        key=f'{arch}_{seed}_{p}_{objective}';arrays[key]=pred
                        risk=absolute_risk(pred,tone,p);optimum=min(p,1-p)*absscale
                        row=dict(architecture=arch,seed=seed,probability=p,objective=objective,key=key,
                                 mean_gain=float(np.mean(np.sum(pred*tone,axis=-1)/np.sum(tone**2,axis=-1))),
                                 mean_target_nmse=float(np.mean(np.mean((pred-p*tone)**2,axis=-1)/energy)),
                                 median_set_distance=float(np.mean(median_distance(pred,tone,p)/absscale)),
                                 absolute_risk_excess=float(np.mean((risk-optimum)/absscale)),
                                 final_train_loss=float(np.mean(trace[-25:])))
                        records.append(row)
        np.savez_compressed(out/'predictions.npz',**arrays)
        results=dict(records=records,seconds=time.monotonic()-start,
                     peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     predictions_sha=sha(out/'predictions.npz'))
        write(out/'results.json',results)
        write(out/'knowledge.json',dict(status='complete_pending_independent_audit',question=__doc__,design=protocol,
              exact_findings=results,limitations=['Two toy architectures and two initializations; not pretrained-model attribution.',
              'At probability .5 the conditional median is any point between the two futures; intermediate gains are not optimization failures.',
              'Absolute loss is twice pinball loss at quantile .5; minimizing it targets marginal medians, not a joint future path.'],
              next_decision='Replay saved risks independently; no pretrained inference or large training follows this control.'))
    except Exception as exc:
        write(out/'knowledge.json',dict(status='failed',question=__doc__,design=protocol,error=str(exc),completed_arms=records))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.out)
