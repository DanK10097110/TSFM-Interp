"""Test many-to-many maps and compare code responses at a fixed target dictionary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import GRID, RANK, Ridge, SPLITS, ci, metric_vectors, read_data, train_validate_one
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS
from research.activation_transfer.boundary.seed_identity import encodings


@torch.inference_mode()
def response(sae,ds):
    clean=encodings(sae,ds.clean)
    edited=encodings(sae,ds.states[:,:,1].copy())
    pre=sae.pre_activations(torch.tensor(ds.states[:,:,1].copy()))
    clean['locked']=clean['hard']
    edited['locked']=torch.relu(pre).numpy()*(clean['hard'][:,None]>0)
    delta={k:e-clean[k][:,None] for k,e in edited.items()}
    delta['dense']=ds.states[:,:,1]-ds.clean[:,None]
    delta['input']=.03*ds.cues
    clean['dense']=ds.clean
    return clean,delta


def clean_map(x,y,dx):
    train,valid=SPLITS['train'][1],SPLITS['valid'][1]
    reg=Ridge(x[:train],y[:train],RANK)
    z=reg.transform(x[train:valid])
    losses=[np.square(reg.predict(z,c)-y[train:valid]).mean() for c in GRID]
    penalty=GRID[np.argmin(losses)]
    # The derivative of an affine map discards both intercepts.
    delta=dx.reshape(-1,dx.shape[-1]).astype(np.float64)
    projected=(delta@reg.train.T)@reg.U
    pred=reg.predict(projected,penalty)-reg.mean_y
    return pred,penalty


def run(model,scratch):
    tr,te=read_data(scratch,'m4',model),read_data(scratch,'reserve',model)
    saes=[load_sae_checkpoint(str(scratch/f'seed{seed}_{model}.pt')) for seed in [1,2]]
    clean,delta=zip(*(response(s,tr) for s in saes))
    _,new=zip(*(response(s,te) for s in saes))
    out={'model':model,'ids':te.ids.tolist(),'target_seed':2,'source_seed':1,'targets':{}}
    for arm in ['hard','shoulder','locked']:
        target=delta[1][arm]; goal=new[1][arm]
        summary,cos={},{ }
        for source in ['hard','shoulder','locked','dense','input']:
            for route in (['edit','clean'] if source!='input' else ['edit']):
                if route=='edit':
                    pred,penalty,_=train_validate_one(delta[0][source],target,new[0][source],RANK)
                else:
                    pred,penalty=clean_map(clean[0][source],clean[1][arm],new[0][source])
                pred=pred.reshape(goal.shape)
                v=metric_vectors(pred,goal)
                label=route+'_'+source
                cos[label]=v['cosine']
                summary[label]={**{k:ci(a) for k,a in v.items()},'penalty':penalty,
                                'relative_pooled_mse':float(np.square(pred-goal).sum()/np.square(goal).sum())}
        for a,b in [('edit_shoulder','edit_hard'),('edit_locked','edit_hard'),('edit_dense','edit_hard'),
                    ('edit_hard','clean_hard'),('edit_shoulder','clean_shoulder')]:
            summary[a+'_minus_'+b]=ci(cos[a]-cos[b])
        out['targets'][arm]=summary
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--scratch',type=Path,default=DEFAULT_SCRATCH)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    args.out.write_text(json.dumps(run(args.model,args.scratch),indent=2,allow_nan=False)+'\n')
