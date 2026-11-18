"""Test whether per-history normalization alone prevents linear composition.

Reuse the six fresh training corpora, with the same past-only mean/SD transform
as the neural models. This is a retrospective competing-explanation control.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from experiments.feature_search.coverage_training import dataset,evaluation
from experiments.feature_search.coherence_study import write,sha


def run(out):
    """Fit normalized ridge without tuning and score all three test groups."""
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(corpora=list(range(493900,493906)),alpha=.01,normalization='past mean and population SD, matching neural forward',scope='Exploratory normalization control on existing replication corpora',source_sha256=sha(__file__)))
    rows=[];arrays={}
    for seed in range(493900,493906):
        ex,ey,labels=evaluation(seed+100,256);mu=ex.mean(1,keepdims=True);sd=np.maximum(ex.std(1,keepdims=True),1e-6)
        for arm in ['withheld','covered']:
            x,y,_=dataset(seed,128,arm=='covered');tm=x.mean(1,keepdims=True);ts=np.maximum(x.std(1,keepdims=True),1e-6)
            model=Ridge(alpha=.01).fit((x-tm)/ts,(y-tm)/ts)
            pred=model.predict((ex-mu)/sd)*sd+mu;e=np.mean((pred-ey)**2,axis=1)/np.var(ex,axis=1)
            summary={group:float(np.mean(e[[i for i,r in enumerate(labels) if r['group']==group]])) for group in ['single','seen','heldout']}
            rows.append(dict(seed=seed,arm=arm,error=summary));arrays[f'{seed}_{arm}']=pred
        arrays[f'{seed}_history']=ex;arrays[f'{seed}_future']=ey
    np.savez_compressed(out/'predictions.npz',**arrays);write(out/'test_labels.json',labels);write(out/'results.json',rows)
    print(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
