"""Fit four donor-dose detectors using only the two archived SAEs and the historical M4 states."""
import argparse
from pathlib import Path

import numpy as np

from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH,dictionary,encode,hidden
from research.activation_transfer.moment.dictionary_transfer.weather.study import MATERIAL,file_hash

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
SEED=22112201
TARGET=22112202
RIDGE=(100.,10.,1.,.1,.01)
THRESH=(1.,.75,.5,.25,.125,0.)


def apply(raw_coef,threshold):
    return np.where(raw_coef>=threshold,np.maximum(raw_coef,0.),0.)


def infer_doses(receiver_codes,fit):
    """Weather-independent application of the M4 fit, [history, 4096] to four nonnegative donor-dose estimates."""
    x=np.asarray(receiver_codes,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(fit['center']):
        raise ValueError('Expected a batch of full recipient SAE codes in checkpoint feature order')
    raw=((x-fit['center'])*fit['inverse_sd'])@fit['slope']+fit['bias']
    return raw,apply(raw,fit['threshold'])


def train(inputs,labels,training,valid):
    center=inputs[training].mean(0);scale=inputs[training].std(0)
    inv=np.divide(1.,scale,out=np.zeros_like(scale),where=scale>0)
    x=(inputs[training]-center)*inv
    val=(inputs[valid]-center)*inv
    u=labels[training].mean(0);y=labels[training]-u
    eigen,basis=np.linalg.eigh(x@x.T/len(training))
    eigen=np.maximum(eigen,0.)
    posed=basis.T@y;mult=x.T@basis/len(training)
    positive_median=np.array([np.median(z[z>0]) for z in labels[training].T])
    weights=np.zeros((x.shape[1],labels.shape[1]))
    best=np.full(labels.shape[1],np.inf);penalty=np.zeros_like(best);threshold=np.zeros_like(best)
    pred=np.zeros_like(labels[valid])
    for lam in RIDGE:
        slopes=mult@(posed/(eigen[:,None]+lam))
        output=val@slopes+u
        for amount in THRESH:
            cut=amount*positive_median
            candidate=apply(output,cut)
            loss=np.mean((candidate-labels[valid])**2,axis=0)
            take=loss<best
            best[take]=loss[take];pred[:,take]=candidate[:,take];threshold[take]=cut[take];penalty[take]=lam
            weights[:,take]=slopes[:,take]
    return dict(center=center,inverse_sd=inv,slope=weights,bias=u,threshold=threshold,lam=penalty,
                val_prediction=pred,val_observed=labels[valid],validation_mse=best,positive_fit_median=positive_median)


def run(out):
    recfile=MATCH/f'large_{SEED}_to_large_matches.npz';grpfile=MATCH/f'large_{SEED}_to_large_groups.npz'
    rec=dict(np.load(recfile));grp=dict(np.load(grpfile));source_atoms=rec['atoms']
    h=hidden('large')[:,-1].copy()
    with np.load(MATERIAL/'large/train.npz') as f:ids=f['ids']
    assert len(h)==1024 and len(set(ids))==1024
    s=dictionary('large',SEED);t=dictionary('large',TARGET)
    labs=encode(s,h)[:,source_atoms].astype(float);codes=encode(t,h).astype(float)
    counts=np.sum(labs>0,axis=0);k=np.argsort(-counts,kind='stable')[:4]
    perm=np.random.default_rng(26112291).permutation(len(ids));tr,va=perm[:768],perm[768:]
    fit=train(codes,labs[:,k],tr,va)
    _,found=infer_doses(codes[va],fit)
    np.testing.assert_allclose(found,fit['val_prediction'],atol=1e-8)
    vec=t.W_dec.detach().numpy().astype(float)
    gc=grp['sixteen_coeff'][k];gids=grp['sixteen_ids'][k]
    group=np.einsum('nq,nqd->nd',gc,vec[gids]);np.testing.assert_allclose(group,grp['sixteen_vector'][k],atol=1e-6)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'fit.npz',**fit,atoms=source_atoms[k],donor_cols=k,match=rec['corr_id'][k],group=group,group_ids=gids,group_coeff=gc,
                        original_atoms=source_atoms,historical_firing=counts,training_ids=ids[tr],validation_ids=ids[va],
                        groups_sha256=file_hash(grpfile),matching_sha256=file_hash(recfile),receiver_seed=TARGET,donor_seed=SEED)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.out)
