"""Predict target native SAE amplitudes from genuinely different TSFM architectures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.moment.crosscoder.evaluate import apply, calibrate
from research.activation_transfer.moment.crosscoder.mechanism import gate_predictions, map_inputs, targets
from research.activation_transfer.replication.data import CACHE as TRAIN, HERE as REP
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.cross_seed_groups.study import population
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
N = 768
DOMAINS = ('daily','monash_traffic','noaa')


def materials(source, target):
    with np.load(TRAIN/'data.npz') as f:
        keep = f['split'] == 'train'
        raw, ids = f['x'][keep], f['ids'][keep]
        train_ids = set(ids)
        assert len(ids)==len(train_ids)==1024
    names = [source,target]
    states = []
    for name in names:
        toks,_,ordered = read_materials(TRAIN,name)
        np.testing.assert_array_equal(ordered[keep],ids)
        states.append([toks[keep].mean(1)])
    xs,identity,domain,cluster = [raw],[ids],[np.full(len(raw),'m4_fitval')],[np.arange(len(raw))]
    native = []
    with np.load(REP/'results'/f'{target}_seed0.npz') as f:
        nominated = f['atoms']
    for pop in DOMAINS:
        for i,name in enumerate(names):
            x,h,_,curr,station,_,atoms,z,*_=population(name,pop)
            if i==0:
                xs.append(x);identity.append(curr);domain.append(np.full(len(x),pop));cluster.append(station)
                assert not train_ids.intersection(curr) and len(curr)==(184 if pop=='noaa' else 128)
            else:
                np.testing.assert_array_equal(curr,identity[-1]);np.testing.assert_array_equal(x,xs[-1])
                np.testing.assert_array_equal(station,cluster[-1]);np.testing.assert_array_equal(nominated,atoms)
                native.append(z)
            states[i].append(h.mean(1))
    ids=np.concatenate(identity)
    assert len(np.unique(ids))==len(ids)
    all_x = np.concatenate(xs)
    scaled = all_x.astype(float)
    scaled = ((scaled-scaled.mean(1,keepdims=True))/scaled.std(1,keepdims=True)).astype('f4')
    assert len({sha256(row.tobytes()).digest() for row in scaled}) == len(ids)
    return (all_x, *(np.concatenate(v) for v in states), ids, np.concatenate(domain),
            np.concatenate(cluster),nominated,np.concatenate(native))


def top_four(count, candidates):
    # Return ordered IDs; ties follow atom ID, including candidates specified in arbitrary order.
    candidates=np.sort(candidates)
    return candidates[np.argsort(-count[candidates],kind='stable')[:4]]


def fit(target):
    source='timesfm' if target=='bolt' else 'bolt'
    x,src,recv,ids,domain,station,nom,native = materials(source,target)
    sae = load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        counts=np.zeros(sae.dict_size,dtype=int)
        for k in range(0,N,128):
            counts += (sae.encode(torch.from_numpy(recv[k:k+128])).numpy()>0).sum(0)
    four = top_four(counts,nom)
    global_four=top_four(counts,np.arange(len(counts)))
    atoms=np.array(list(dict.fromkeys([*nom,*global_four])),dtype=int)
    measures,top,dose = targets(sae,recv,atoms)
    np.testing.assert_array_equal(gate_predictions(measures),dose)
    np.testing.assert_allclose(dose[1024:,:len(nom)],native,rtol=1e-5,atol=3e-5)
    np.testing.assert_array_equal(dose[1024:,:len(nom)]>0,native>0)
    validation=slice(N,1024)
    best=np.inf
    for lam,pred in map_inputs(src[:N],recv[:N],src):
        mse=float(np.mean((pred[validation]-recv[validation])**2))
        if mse<best:
            best,best_lam,hidden=mse,lam,pred.astype('f4')
    _,top_pred,estimate=targets(sae,hidden,atoms)
    frequent=top_four(counts,np.arange(len(counts)))
    assert np.array_equal(frequent,global_four)
    common=np.argsort(-counts,kind='stable')[:sae.k]
    normalized=x.astype(float)
    normalized=(normalized-normalized.mean(1,keepdims=True))/np.sqrt(normalized.var(1,keepdims=True)+1e-5)
    predictions={'state_exact':estimate}
    diagnostics=dict(map_penalty=best_lam,validation_hidden_mse=best,train_histories=N,source=source,target=target)
    for arm,features in [('direct',src),('raw32',normalized[:,-32:])]:
        w,b,gate,metadata=calibrate(features[:1024],dose[:1024])
        _,pred=apply(features,w,b,gate)
        predictions[arm]=pred.astype('f4');diagnostics[arm]=metadata
    cutoff_best=np.full(len(atoms),np.inf)
    chosen=np.full(len(atoms),np.nan)
    gated=np.zeros_like(dose)
    for lam, signed in map_inputs(src[:N],measures[:N],src):
        proposal=gate_predictions(signed)
        losses=np.mean((proposal[validation]-dose[validation])**2,axis=0)
        use=losses<cutoff_best
        chosen[use]=lam;cutoff_best[use]=losses[use];gated[:,use]=proposal[:,use]
    predictions['explicit_gate']=gated
    diagnostics['gate_penalties']=chosen.tolist()
    diagnostics['selection']=dict(historical=nom.tolist(),four_nominated=four.tolist(),four_global=global_four.tolist(),
                                  train_fires=dict(zip(map(str,atoms),map(int,counts[atoms]))))
    gains={}
    for pop in DOMAINS:
        rows=domain==pop
        overlap=(top[rows,:,None]==top_pred[rows,None,:]).any(2).sum(1)/sae.k
        gains[pop]=dict(hidden_r2=float(1-np.sum((recv[rows]-hidden[rows]).astype(float)**2)/np.sum((recv[rows]-recv[:N].mean(0,dtype=float))**2)),
                        mean_topk_overlap=float(np.mean(overlap)),common_topk_overlap=float(np.isin(top[rows],common).mean()))
    diagnostics['full_state']=gains
    OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(OUT/f'{source}_to_{target}.npz',ids=ids,domain=domain,cluster=station,atoms=atoms,
                        nominated=nom,four_nominated=four,four_global=global_four,truth=dose,**predictions)
    (OUT/f'{source}_to_{target}_fit.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',choices=('bolt','timesfm'))
    args=parser.parse_args();torch.set_num_threads(2);fit(args.target)
