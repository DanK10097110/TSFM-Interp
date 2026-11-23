"""Extend the frozen code-smoothing experiment to paired Chronos-2 responses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import SPLITS, fit_maps, pca, read_data, within
from research.activation_transfer.boundary.capture import ALL_MODELS, DEFAULT_SCRATCH, dictionary_path
from research.activation_transfer.boundary.smooth import smooth_reps


def study(scratch, dataset):
    train = {m:read_data(scratch,'m4',m) for m in ALL_MODELS}
    test = {m:read_data(scratch,dataset,m) for m in ALL_MODELS}
    for collection in [train,test]:
        for m in ALL_MODELS:
            np.testing.assert_array_equal(collection[m].ids,collection['bolt'].ids)
            np.testing.assert_array_equal(collection[m].raw,collection['bolt'].raw)
    reps_train,reps_test,encoders,gate = {},{},{},{}
    for m in ALL_MODELS:
        encoders[m] = load_sae_checkpoint(str(dictionary_path(m)))
        reps_train[m],_ = smooth_reps(encoders[m],train[m])
        reps_test[m],extra = smooth_reps(encoders[m],test[m])
        gate[m] = {key:{'any_clean_id_diff':float(r['support_difference'].mean()),
                        'any_edited_id_diff':float(r['edited_support_difference'].mean())}
                   for key,r in extra.items()}
    model = 'chronos2'
    c2 = test[model]
    within_c2 = within(encoders[model],c2,slice(None),
                       pca(train[model].clean[:SPLITS['train'][1]],encoders[model].k),
                       hashlib.sha256(c2.raw.tobytes()).hexdigest())
    out = {'dataset':dataset,'checkpoint':ALL_MODELS[model][3],'site':ALL_MODELS[model][-1],
           'training_horizon':64,'test_ids':c2.ids.tolist(),'within':within_c2,'gate':gate,'transfer':{}}
    for other in ['bolt','timesfm']:
        for a,b in [('chronos2',other),(other,'chronos2')]:
            pair = a + '_to_' + b
            pred = {}
            out['transfer'][pair] = fit_maps(reps_train[a],reps_train[b]['dense'],
                                              reps_test[a],reps_test[b]['dense'],export=pred)
            exports = {f'{b}_cross_{arm}':v for arm,v in pred.items()}
            for arm in ['dense','decoded','fixed_decoded','shoulder','subtract']:
                key = arm+'_decoded' if arm in {'shoulder','subtract'} else arm
                exports[f'{b}_native_{arm}'] = reps_test[b][key]
            np.savez(scratch/f'{dataset}_third_{pair}_predictions.npz',ids=c2.ids,**exports)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['fast','weekly'],required=True)
    p.add_argument('--scratch',type=Path,default=DEFAULT_SCRATCH)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    args.out.write_text(json.dumps(study(args.scratch,args.dataset),indent=2,allow_nan=False)+'\n')
