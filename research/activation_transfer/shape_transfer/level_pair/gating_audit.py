"""Recompute constrained grid histories, then independently inspect singleton SAE ranks and cutoffs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.shape_transfer.level_pair.gating import CONDITIONS,coordinate_targets
from research.activation_transfer.shape_transfer.level_pair.inputs import HERE
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.sae.train import load_sae_checkpoint


def independent_grid(x):
    # Uncentered four-row pseudoinverse. The primary constructor demeans the
    # series first and solves a separate three-row Gram system.
    x=np.asarray(x,dtype=float)
    length=x.shape[1]
    rows=np.zeros((4,length))
    rows[0]=1/length;rows[1,-32:]=1/32;rows[2,-1]=1.
    t=np.linspace(-.5,.5,32);rows[3,-32:]=t/(t@t)
    inverse=np.linalg.pinv(rows)
    native=x@rows.T
    residual=x-native@inverse.T
    energy=np.sum(x*x,axis=1)
    residual_energy=np.sum(residual**2,axis=1)
    sd=x.std(axis=1);waves={};feasible={}
    for mode in ('native','joint'):
        for cond,target in [('low',0.),('high',1.25)]:
            key=mode+'_'+cond
            required=native.copy()
            required[:,1]=native[:,0]+sd*target
            if mode=='joint':
                required[:,2]=native[:,0]+.5*sd
                required[:,3]=0.
            projected=required@inverse.T
            remaining=energy-np.sum(projected**2,axis=1)
            feasible[key]=remaining>=0
            waves[key]=(projected+residual*np.sqrt(np.maximum(remaining,0)/residual_energy)[:,None]).astype('f4')
        both=feasible[mode+'_low']&feasible[mode+'_high']
        feasible[mode+'_low']=both;feasible[mode+'_high']=both
    return waves,feasible


def inspect(model,out,inputs,waves,feasible):
    saved=dict(np.load(out/(model+'_gating.npz')))
    effects=dict(np.load(out/(model+'_effects.npz')))
    adapter=adapter_for(model);sae=load_sae_checkpoint(str(dictionary_path(model)))
    site=MODELS[model][-1];which=effects['positions'];atom=int(saved['atom'])
    all_waves=dict(clean=inputs['inputs'],minus=inputs['edits'][:,0,0],plus=inputs['edits'][:,0,1],**waves)
    rows=[0,1,128,129];checks=[]
    for i in rows:
        for condition in CONDITIONS:
            if not saved[condition+'_eligible'][i]:continue
            if condition in feasible:assert feasible[condition][i]
            with torch.inference_mode():
                token=capture_raw_tokens(adapter,all_waves[condition][i:i+1],[site])[site][0,which]
                hidden=token.mean(axis=0)
                pre=((hidden-sae.b_dec)@sae.W_enc+sae.b_enc).numpy()
            others=np.delete(pre,atom)
            rank=1+np.count_nonzero(others>pre[atom])
            threshold=max(0.,float(np.sort(others)[-int(saved['k'])]))
            margin=float(pre[atom])-threshold
            expected={field:saved[condition+'_'+field][i] for field in ('pre','cutoff','margin','rank','z')}
            observed=dict(pre=float(pre[atom]),cutoff=threshold,margin=margin,rank=int(rank),
                          z=float(pre[atom]) if margin>0 else 0.)
            for field in ('pre','cutoff','margin','z'):
                np.testing.assert_allclose(observed[field],expected[field],atol=2.2e-4,rtol=1e-4)
            # Batch-dependent FP32 differences can reverse very close ties.
            near=np.count_nonzero(np.abs(others-pre[atom])<.00022)
            assert abs(rank-expected['rank'])<=near,(model,i,condition,rank,expected,near)
            checks.append(dict(id=str(saved['ids'][i]),domain=str(saved['split'][i]),condition=condition,
                               errors={k:float(abs(observed[k]-expected[k])) for k in observed}))
    return checks


def audit(out):
    inp=dict(np.load(out/'inputs.npz'));waves,feasible=independent_grid(inp['inputs'])
    orig,ok,_=coordinate_targets(inp['inputs']);size={}
    for k in waves:
        np.testing.assert_array_equal(feasible[k],ok[k])
        size[k]=float(np.max(np.abs(waves[k][ok[k]]-orig[k][ok[k]])))
        np.testing.assert_allclose(waves[k][ok[k]],orig[k][ok[k]],atol=4e-6)
    result=dict(independent_input_max_abs=size,models={m:inspect(m,out,inp,waves,feasible) for m in MODELS})
    (out/'gating_replay.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args();torch.set_num_threads(1);audit(a.out)
