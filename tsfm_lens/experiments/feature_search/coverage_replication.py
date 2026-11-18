"""Fresh training-corpus replication and optimization-budget control for coverage.

Six independently generated corpora, paired initializations, fixed 400/800-step
checkpoints. Neither checkpoint is selected using test performance.
"""
import argparse,time,resource
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from experiments.feature_search.coverage_training import dataset,evaluation,make_model
from experiments.feature_search.coherence_study import write,sha


def run(out):
    """Run the fixed crossed architecture/coverage/time design on a single CPU."""
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.monotonic();rows=[];saved={}
    write(out/'protocol.json',dict(corpora=list(range(493900,493906)),train_blocks=128,test_blocks=256,steps=[400,800],batch=64,learning_rate=.002,architectures=['mlp','transformer'],cap_seconds=600,source_sha256=sha(__file__),inference='Exploratory replication; paired corpus plus initialization is the unit, n=6; no selected checkpoints'))
    for seed in range(493900,493906):
        ex,ey,labels=evaluation(seed+100,256);scale=np.var(ex,axis=1)
        saved[f'{seed}_future']=ey;saved[f'{seed}_history']=ex
        for arm,covered in [('withheld',False),('covered',True)]:
            x,y,_=dataset(seed,128,covered);tx,ty=torch.tensor(x),torch.tensor(y)
            for architecture in ['mlp','transformer','ridge']:
                if architecture=='ridge':
                    model=Ridge(alpha=.01).fit(x,y);pred=model.predict(ex);checkpoints=[0]
                else:
                    model=make_model(architecture,seed);optimizer=torch.optim.Adam(model.parameters(),lr=.002);generator=torch.Generator().manual_seed(seed+1);checkpoints=[400,800]
                for checkpoint in checkpoints:
                    if architecture!='ridge':
                        model.train()
                        for step in range(400):
                            ix=torch.randint(len(tx),(64,),generator=generator);loss=((model(tx[ix])-ty[ix])**2).mean();optimizer.zero_grad();loss.backward();optimizer.step()
                            if time.monotonic()-started>600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*2**30: raise RuntimeError('Training cap')
                        model.eval()
                        with torch.no_grad():pred=model(torch.tensor(ex)).numpy()
                    key=f'{seed}_{architecture}_{arm}_{checkpoint}';saved[key]=pred
                    error=np.mean((pred-ey)**2,axis=1)/scale
                    summary={group:float(np.mean(error[[i for i,r in enumerate(labels) if r['group']==group]])) for group in ['single','seen','heldout']}
                    rows.append(dict(seed=seed,architecture=architecture,arm=arm,steps=checkpoint,error=summary))
        print(seed,'complete',flush=True)
    np.savez_compressed(out/'predictions.npz',**saved)
    write(out/'test_labels.json',labels)
    write(out/'results.json',dict(results=rows,seconds=time.monotonic()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,predictions_sha256=sha(out/'predictions.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
