"""Patch equal-source-dose adjacent-window and atom directions through the original clean map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.materials import CACHE, HERE as ATOMS, MODELS, adapter_for
from research.activation_transfer.atoms.matched_windows.prepare import HERE, SCRATCH
from research.activation_transfer.atoms.vectors import bridge
from research.activation_transfer.boundary.capture import digest
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice


def unit_windows(delta):
    norm=np.linalg.norm(delta.astype(np.float64),axis=1)
    unit=np.divide(delta,norm[:,None],out=np.zeros(delta.shape,dtype=np.float64),where=norm[:,None]>0)
    return unit,norm


def requested_doses(codes,decoder):
    return codes.astype(np.float64)*np.linalg.norm(decoder.astype(np.float64),axis=1)[None]


def patch_vectors(enc,layer,selector,x,tokens,base,active,dose,per_row,batch=24):
    output=np.zeros((*active.shape,base.shape[-1]),dtype=np.float32)
    rows,features=np.nonzero(active)
    for start in range(0,len(rows),batch):
        i,j=rows[start:start+batch],features[start:start+batch]
        delta=(dose[i,j,None]*per_row[i]).astype(np.float32)
        moved=change_forecasts(enc,layer,x[i],tokens[i],delta,selector,batch)
        # A zero write has exactly zero effect, even if batch grouping produces fp32 noise.
        output[i,j]=np.where(np.any(delta!=0,axis=1)[:,None],moved-base[i],0.)
    return output


def study(source,cache,scratch,results):
    target=next(m for m in MODELS if m!=source)
    with np.load(cache/'data.npz') as f:
        all_ids=f['ids'];train=f['split']=='train';daily=f['split']=='daily';x=f['x'][daily]
    ids=all_ids[daily]
    with np.load(cache/(source+'_vectors.npz')) as f:
        records=dict(f)
    archived_path=ATOMS/'results'/(source+'_effects.npz')
    with np.load(archived_path) as f:
        idx=f['split']=='daily';np.testing.assert_array_equal(ids,f['ids'][idx])
        archived={k:f[k][idx] for k in ['source_native','target_dense','source_codes','active']}
    meta=json.loads((ATOMS/'results'/(source+'_metadata.json')).read_text())
    np.testing.assert_array_equal(records['source_atoms'],meta['source_atoms'])
    np.testing.assert_array_equal(all_ids,records['ids'])
    codes=records['source_codes'][daily]
    np.testing.assert_array_equal(codes,archived['source_codes'])
    active=(codes>0);np.testing.assert_array_equal(active,archived['active'])
    htrain,tokens,positions={}, {}, {}
    for model in (source,target):
        with np.load(cache/(model+'.npz')) as f:
            np.testing.assert_array_equal(all_ids,f['ids'])
            htrain[model]=f['tokens'][train].mean(1)
            tokens[model]=f['tokens'][daily];positions[model]=f['positions']
            assert str(f['sae_sha256'])==meta['sae_sha256'][model]
    with np.load(scratch/(source+'_shift.npz')) as f:
        np.testing.assert_array_equal(ids,f['ids']);np.testing.assert_array_equal(positions[source],f['positions'])
        unit,norm=unit_windows(f['delta']); replay_hidden=float(f['native_replay_absmax']);spans=f['spans']
    dirs=records['source_decoder'];n_atoms=len(dirs)
    combined=np.vstack([dirs,unit])
    mapped,_,_=bridge(htrain[source],htrain[target],combined,2026112018)
    np.testing.assert_allclose(mapped[:n_atoms],records['target_dense'],atol=1e-8,rtol=1e-8)
    dose=requested_doses(codes,dirs)
    zero=norm==0
    arr,idx=np.nonzero(active)
    written=np.linalg.norm((dose[arr,idx,None]*unit[arr]).astype(np.float32).astype(np.float64),axis=1)
    observed=np.linalg.norm((-codes[arr,idx,None]*dirs[idx]).astype(np.float32).astype(np.float64),axis=1)
    keep=~zero[arr]
    np.testing.assert_allclose(written[keep],observed[keep],atol=2e-6,rtol=2e-6)
    diagnostics={'source':source,'target':target,'feature_ids':records['source_atoms'].tolist(),
                 'n_series':len(x),'active_cases':int(active.sum()),'active_counts':active.sum(0).tolist(),
                 'source_sae_sha256':str(records['source_sae_sha']),'target_sae_sha256':str(records['target_sae_sha']),
                 'archived_atom_effects_sha256':digest(archived_path),'vector_sha256':digest(cache/(source+'_vectors.npz')),
                 'max_atom_map_abs_error':float(np.abs(mapped[:n_atoms]-records['target_dense']).max()),
                 'max_source_dose_abs_error':float(np.max(np.abs(written[keep]-observed[keep]))),
                 'max_source_dose_relative_error':float(np.max(np.abs(written[keep]/observed[keep]-1))),
                 'source_hidden_replay_max':replay_hidden,'source_native_spans':spans.tolist(),
                 'zero_window_series':ids[zero].tolist(),'zero_window_active_cases':int(active[zero].sum()),'replays':{}}
    result={'ids':ids,'feature_ids':records['source_atoms'],'active':active,'source_codes':codes,
            'source_dose':dose,'window_source_norm':norm,'mapped_unit_norm':np.linalg.norm(mapped[n_atoms:],axis=1),
            'atom_source':archived['source_native'],'atom_target':archived['target_dense']}
    for model,side,vectors,atom_vectors,existing in [
            (source,'source',unit,dirs,archived['source_native']),
            (target,'target',mapped[n_atoms:],records['target_dense'],archived['target_dense'])]:
        enc=adapter_for(model);layer=MODELS[model][-1]; pos=positions[model];tok=tokens[model]
        selector=last_window_slice(enc,pos)
        ordinary=change_forecasts(enc,layer,x,tok,None,selector,24)
        replay=change_forecasts(enc,layer,x,tok,np.zeros_like(tok[:,0]),selector,24)
        clean_error=float(np.max(np.abs(replay-ordinary)))
        assert clean_error < 5e-5
        # Replay atom effects as a site, index, sign, dose, and baseline audit; score the archived arrays.
        recalculated=run_arm(enc,layer,selector,x,tok,replay,active,codes,atom_vectors,24)
        atom_error=float(np.max(np.abs(recalculated-existing)))
        np.testing.assert_allclose(recalculated,existing,atol=2e-4,rtol=.002)
        result['window_'+side]=patch_vectors(enc,layer,selector,x,tok,replay,active,dose,vectors)
        diagnostics['replays'][side]={'model':model,'site':layer,'positions':pos.tolist(),
                                      'clean_forecast_absmax':clean_error,'archived_atom_forecast_absmax':atom_error}
    np.savez_compressed(results/(source+'_pairs.npz'),**result)
    (results/(source+'_audit.json')).write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=MODELS);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--scratch',type=Path,default=SCRATCH);p.add_argument('--results',type=Path,default=HERE/'results')
    a=p.parse_args();torch.set_num_threads(3);study(a.source,a.cache,a.scratch,a.results)
