"""Match independently trained SAE features within one model's identical residual coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import ci, metric_vectors, read_data
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS
from research.activation_transfer.boundary.smooth import smooth_codes


@torch.inference_mode()
def encodings(sae, state):
    pre=sae.pre_activations(torch.as_tensor(state))
    sm,_=smooth_codes(pre,sae.k)
    return {'hard':sae.sparsify(pre).numpy(),'shoulder':sm['shoulder'].numpy()}


def unit_rows(x):
    x = np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-20)


def match_features(x,y):
    similarity=unit_rows(x) @ unit_rows(y).T
    a,b=linear_sum_assignment(similarity,maximize=True)
    return a,b,similarity[a,b]


def random_targets(indices,frequency,seed):
    rng=np.random.default_rng(seed)
    ordering=np.argsort(frequency[indices],kind='stable')
    out=indices.copy()
    for group in np.array_split(ordering,np.ceil(len(ordering)/16).astype(int)):
        out[group]=rng.permutation(indices[group])
    return out


def fit_scale(x,y):
    return np.maximum(0.,(x*y).sum(axis=0)/np.maximum((x*x).sum(axis=0),1e-20))


def apply_match(source,target_shape,a,b,scale):
    z=np.zeros(target_shape,dtype=np.float32)
    z[...,b]=source[...,a]*scale
    return z


def run(model,scratch):
    train,test=read_data(scratch,'m4',model),read_data(scratch,'reserve',model)
    saes=[load_sae_checkpoint(str(scratch/f'seed{seed}_{model}.pt')) for seed in [1,2]]
    tr=[encodings(s,train.clean[:128]) for s in saes]
    clean=[encodings(s,test.clean) for s in saes]
    alt=[encodings(s,test.states[:,:,1].copy()) for s in saes]
    for i,s in enumerate(saes):
        tr[i]['locked'] = tr[i]['hard']
        clean[i]['locked'] = clean[i]['hard']
        with torch.inference_mode():
            raw_alt = torch.relu(s.pre_activations(torch.tensor(test.states[:,:,1].copy()))).numpy()
        alt[i]['locked'] = raw_alt * (clean[i]['hard'][:,None]>0)
    pools=[np.flatnonzero((row['hard']>0).any(axis=0)) for row in tr]
    rates=(tr[1]['hard']>0).sum(0)
    out={'model':model,'n_train':128,'n_test':len(test.ids),'test_ids':test.ids.tolist(),
         'calibration_active_counts':[len(p) for p in pools]}
    for name in ['decoder','activation']:
        if name=='decoder':
            features=[s.W_dec.detach().numpy()[p] for s,p in zip(saes,pools)]
        else:
            features=[row['hard'][:,p].T for row,p in zip(tr,pools)]
            features=[f-f.mean(1,keepdims=True) for f in features]
        ia,ib,sim=match_features(*features)
        a,b=pools[0][ia],pools[1][ib]
        res={'n_pairs':len(a),'assignment_similarity_quantiles':np.quantile(sim,[0,.25,.5,.75,1]).tolist(),
             'source_test_support_coverage':ci((clean[0]['hard'][:,a]>0).sum(-1)/(clean[0]['hard']>0).sum(-1)),
             'target_test_support_coverage':ci((clean[1]['hard'][:,b]>0).sum(-1)/(clean[1]['hard']>0).sum(-1))}
        res['target_coordinate_oracle'] = {}
        for arm in ['hard','shoulder','locked']:
            dz = alt[1][arm]-clean[1][arm][:,None]
            restricted = np.zeros_like(dz)
            restricted[...,b] = dz[...,b]
            res['target_coordinate_oracle'][arm+'_delta_cosine'] = ci(metric_vectors(restricted,dz)['cosine'])
        table={}
        for label,targets in [('match',[b]),('shuffle',[random_targets(b,rates,72643+s) for s in range(8)])]:
            stats={}
            for arm in ['hard','shoulder','locked']:
                cv,dv,ov=[] ,[],[]
                for target_ids in targets:
                    w=fit_scale(tr[0][arm][:,a],tr[1][arm][:,target_ids])
                    pc=apply_match(clean[0][arm],clean[1][arm].shape,a,target_ids,w)
                    source_delta=alt[0][arm]-clean[0][arm][:,None]
                    target_delta=alt[1][arm]-clean[1][arm][:,None]
                    pd=apply_match(source_delta,target_delta.shape,a,target_ids,w)
                    cv.append(metric_vectors(pc,clean[1][arm])['cosine'])
                    dv.append(metric_vectors(pd,target_delta)['cosine'])
                    if arm=='hard':
                        # Unscaled assignment quantifies identity overlap even for a zero calibration slope.
                        moved=apply_match((clean[0][arm]>0).astype(np.float32),clean[1][arm].shape,
                                          a,target_ids,1.)>0
                        ov.append((moved & (clean[1][arm]>0)).sum(-1)/(clean[1][arm]>0).sum(-1))
                stats[arm+'_clean_cosine']=np.mean(cv,axis=0)
                stats[arm+'_delta_cosine']=np.mean(dv,axis=0)
                if ov:
                    stats['correct_target_active_fraction']=np.mean(ov,axis=0)
            stats['shoulder_minus_hard_delta_cosine']=stats['shoulder_delta_cosine']-stats['hard_delta_cosine']
            stats['locked_minus_hard_delta_cosine']=stats['locked_delta_cosine']-stats['hard_delta_cosine']
            stats['hard_clean_minus_delta_cosine']=stats['hard_clean_cosine'][:,None]-stats['hard_delta_cosine']
            table[label]=stats
            res[label]={k:ci(z) for k,z in stats.items()}
        res['match_minus_shuffle']={k:ci(v-table['shuffle'][k]) for k,v in table['match'].items()}
        out[name]=res
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--scratch',type=Path,default=DEFAULT_SCRATCH)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    args.out.write_text(json.dumps(run(args.model,args.scratch),indent=2,allow_nan=False)+'\n')
