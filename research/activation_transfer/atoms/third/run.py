"""Execute cross-model literal source-atom ablation on two frozen recipients."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.materials import CACHE as ORIGINAL
from research.activation_transfer.atoms.third.collect import CACHE, HERE
from research.activation_transfer.atoms.vectors import bridge, choose_atoms, coactive
from research.activation_transfer.boundary.capture import ALL_MODELS, adapter_for, dictionary_path, digest
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.experiment import derangements
from tsfm_lens.sae.train import load_sae_checkpoint


def prepare(source, target, cache, original):
    with np.load(original/'data.npz') as d:
        ids, split, x = d['ids'],d['split'],d['x']
    train=split=='train'; test=np.isin(split, ['daily','electricity'])
    if (train.sum(),test.sum()) != (1024,320):
        raise ValueError('Unexpected study split')
    states, codes, dec, toks, positions, hashes = {},{},{},{},{},{}
    for name in (source,target):
        directory=cache if name=='chronos2' else original
        with np.load(directory/(name+'.npz')) as d:
            np.testing.assert_array_equal(d['ids'],ids)
            tokens=d['tokens']; positions[name]=d['positions']; hashes[name]=str(d['sae_sha256'])
        path=dictionary_path(name)
        if digest(path)!=hashes[name]:
            raise ValueError(f'Checkpoint provenance mismatch for {name}')
        sae=load_sae_checkpoint(str(path))
        h=tokens.mean(axis=1)
        with torch.inference_mode():
            codes[name]=sae.encode(torch.from_numpy(h)).numpy()
            dec[name]=sae.W_dec.numpy().copy()
        states[name]=h[train]; toks[name]=tokens[test]
    selected=choose_atoms(codes[source][train],16,2026112017)
    if source=='timesfm':
        archived=json.loads((HERE.parent/'results/timesfm_metadata.json').read_text())
        np.testing.assert_array_equal(selected, archived['source_atoms'])
    eligible=np.flatnonzero((codes[target][train]>0).mean(0)>=.01)
    matched,corr=coactive(codes[source][train][:,selected],codes[target][train],eligible)
    directions=dec[source][selected]; dense,_,_=bridge(states[source],states[target],directions,2026112018)
    z=codes[source][test][:,selected]; active=z>0
    permutations=np.stack(list(derangements(len(selected))))
    scaled=[]
    norms=np.linalg.norm(dense,axis=1,keepdims=True)
    for p in permutations:
        scaled.append(dense[p]*norms/np.maximum(norms[p],1e-15))
    arms={source:{'native':(z,directions)},target:{'dense':(z,dense),
        'coactive':(codes[target][test][:,matched],dec[target][matched]),
        **{'wrong'+str(i):(z,v) for i,v in enumerate(scaled)}}}
    meta=dict(source=source,target=target,sha256=hashes,select_seed=2026112017,
              train_frequency=(codes[source][train][:,selected]>0).mean(0).tolist(),
              atoms=selected.tolist(),coactive_match=matched.tolist(),train_correlations=corr.tolist(),
              eligible_receiver_atoms=len(eligible),derangements=permutations.tolist(),
              dense_norm=norms.ravel().tolist())
    payload=dict(ids=ids[test],split=split[test],atoms=selected,active=active,z=z,
                 matched_target_codes=codes[target][test][:,matched])
    return x[test],toks,positions,arms,payload,meta


def execute(source, cache, original, batch):
    target='timesfm' if source=='chronos2' else 'chronos2'
    x,tokens,positions,arms,arrays,meta=prepare(source,target,cache,original)
    active=arrays['active']; rows,atoms=np.nonzero(active); meta['replay']={}
    for name in (source,target):
        enc=adapter_for(name); layer=ALL_MODELS[name][-1]; tok=tokens[name]
        selector=last_window_slice(enc,positions[name]); zero=np.zeros((len(x),tok.shape[-1]),'f4')
        ordinary=change_forecasts(enc,layer,x,tok,None,selector,batch)
        clean=change_forecasts(enc,layer,x,tok,zero,selector,batch)
        packed=change_forecasts(enc,layer,x[rows],tok[rows],zero[rows],selector,batch)
        errors=packed-clean[rows]
        noise=np.sqrt(np.mean(errors.astype(float)**2,axis=-1))
        replay=dict(plain_vs_clean_max=float(np.max(abs(clean-ordinary))),
                    packed_vs_clean_max=float(abs(errors).max()),max_packed_rms=float(noise.max()),
                    n_active_pairs=len(rows),arm_single_max_error={},floor_pairs={})
        if replay['plain_vs_clean_max']>1e-5 or replay['max_packed_rms']>1e-5:
            raise ValueError(f'Zero-dose floor exceeds the registered threshold for {name}: {replay}')
        for key,(strength,direction) in arms[name].items():
            result=run_arm(enc,layer,selector,x,tok,clean,active,strength,direction,batch)
            i,j=rows[0],atoms[0]
            delta=np.asarray(-strength[i,j]*direction[j],dtype='f4')[None]
            direct=change_forecasts(enc,layer,x[i:i+1],tok[i:i+1],delta,selector,1)
            single=change_forecasts(enc,layer,x[i:i+1],tok[i:i+1],None,selector,1)
            err=direct[0]-single[0]-result[i,j]
            np.testing.assert_allclose(err,0,atol=2e-5)
            replay['arm_single_max_error'][key]=float(abs(err).max())
            replay['floor_pairs'][key]=int((np.sqrt(np.mean(result[rows,atoms].astype(float)**2,axis=-1)) <= 10*noise).sum())
            arrays[key]=result
            print(source,'->',target,key,len(rows),flush=True)
        meta['replay'][name]=replay
        del enc
    if source=='timesfm':
        with np.load(HERE.parent/'results/timesfm_effects.npz') as old:
            ref=np.isin(old['split'],['daily','electricity'])
            np.testing.assert_array_equal(old['ids'][ref],arrays['ids'])
            diff=old['source_native'][ref]-arrays['native']
            np.testing.assert_allclose(diff,0,atol=2e-5)
            meta['archived_identical_source_max_error']=float(abs(diff).max())
    np.savez_compressed(HERE/(source+'_effects.npz'),**arrays)
    (HERE/(source+'_metadata.json')).write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=['chronos2','timesfm']); p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--original',type=Path,default=ORIGINAL); p.add_argument('--batch',type=int,default=24)
    args=p.parse_args(); torch.set_num_threads(3); execute(args.source,args.cache,args.original,args.batch)
