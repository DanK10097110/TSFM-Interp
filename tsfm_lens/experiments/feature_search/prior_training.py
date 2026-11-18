"""Matched continuation-prior training, with an analytic conditional-mean control.

Identical histories and initialization across arms; only the future survival
probability changes. This identifies a mechanism in small models, not the cause
of any pretrained TSFM's behavior. Future branches are not new independent data.
"""
from pathlib import Path
import argparse,time,resource
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.coverage_training import make_model


def bank(seed,n):
    """All frequencies observed in history; future survival is unobservable."""
    rng=np.random.default_rng(seed);freq=rng.choice([4.,5.35,128/19.2,9.],n);amp=rng.uniform(.4,1.6,n);phase=rng.uniform(-np.pi,np.pi,n)
    t=np.arange(-128,32);tone=amp[:,None]*np.sin(2*np.pi*freq[:,None]*t/128+phase[:,None])
    noise=rng.normal(0,.1,(n,128));x=tone[:,:128]+noise
    return x.astype(np.float32),tone[:,128:].astype(np.float32),np.column_stack([freq,amp,phase])


def expected_loss(pred,tone,probability):
    """Exactly marginalize the two futures; avoid random branch sampling noise."""
    pred,tone=np.asarray(pred,float),np.asarray(tone,float)
    return probability*np.mean((pred-tone)**2,axis=-1)+(1-probability)*np.mean(pred**2,axis=-1)


def run(out,steps=200,seeds=(495710,495711)):
    """Paired optimization, held-out histories, and analytic risk decomposition."""
    import torch
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    protocol=dict(seeds=list(seeds),steps=steps,train=512,test=256,probabilities=[0.,.5,1.],architectures=['mlp','transformer'],batch=64,lr=.002,source_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('coverage_training.py')),cap_seconds=600,cap_bytes=2*2**30,fit_gate='p1 TEST conditional-mean NMSE <=0.15 per architecture/seed before interpreting a null as an adequate learned mechanism',scope='Controlled tiny models, not attribution of TSFM training; exact conditional-mean labels equal expected two-branch squared-loss gradients')
    write(out/'protocol.json',protocol);x,tone,meta=bank(4957000,512);ex,etone,emeta=bank(4958000,256);arrays=dict(train_history=x,train_tone=tone,train_latents=meta,test_history=ex,test_tone=etone,test_latents=emeta);records=[]
    for arch in protocol['architectures']:
        for seed in seeds:
            for probability in protocol['probabilities']:
                model=make_model(arch,seed);optimizer=torch.optim.Adam(model.parameters(),lr=.002);rng=np.random.default_rng(seed);tx=torch.tensor(x);ty=torch.tensor(probability*tone);trace=[]
                for step in range(steps):
                    ix=rng.integers(0,len(x),64);pred=model(tx[ix]);loss=((pred-ty[ix])**2).mean();optimizer.zero_grad();loss.backward();optimizer.step();trace.append(float(loss.detach()))
                    if time.monotonic()-start>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*2**30:raise RuntimeError('Training pilot resource cap')
                model.eval()
                captured=[]
                head=model.head if arch=='transformer' else model.net[-1]
                handle=head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().copy()))
                try:
                    with torch.no_grad():pred=model(torch.tensor(ex)).numpy();fit=model(tx).numpy()
                finally:handle.remove()
                key=f'{arch}_{seed}_{probability}';arrays[key+'_test_hidden']=captured[0];arrays[key+'_train_hidden']=captured[1];torch.save(model.state_dict(),out/(key+'.pt'));arrays[key]=pred;scale=np.mean(etone.astype(float)**2,axis=1);gain=np.sum(pred*etone,axis=1)/np.sum(etone**2,axis=1)
                row=dict(architecture=arch,seed=seed,training_probability=probability,key=key,training_mse=float(np.mean((fit-probability*tone)**2)),last_loss=float(np.mean(trace[-25:])),mean_gain=float(gain.mean()),conditional_mean_nmse=float(np.mean(np.mean((pred-probability*etone)**2,axis=1)/scale)),risks={str(q):float(np.mean(expected_loss(pred,etone,q)/scale)) for q in protocol['probabilities']})
                records.append(row);print(row,flush=True)
    np.savez_compressed(out/'predictions.npz',**arrays)
    write(out/'results.json',dict(records=records,seconds=time.monotonic()-start,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,predictions_sha256=sha(out/'predictions.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=200);a=p.parse_args();run(a.out,a.steps)
