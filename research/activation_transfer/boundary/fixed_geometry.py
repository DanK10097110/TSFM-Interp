"""Isolate a learned 32-dimensional span from correlations among its encoder axes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import SPLITS, ci, load_study
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, dictionary_path
from research.activation_transfer.boundary.fixed import fixed_bases, score


def orthogonal_parts(columns):
    q,r = np.linalg.qr(columns.astype(np.float64))
    return q.astype(np.float32),r.astype(np.float32)


def norm_fraction(x, q):
    return np.square(x @ q).sum(-1)/np.maximum(np.square(x).sum(-1),1e-30)


def run(scratch):
    train,test = load_study(scratch,'m4'), load_study(scratch,'reserve')
    out = {}
    for a,b in [('bolt','timesfm'),('timesfm','bolt')]:
        def dense(ds, m):
            return ds[m].states[:,:,1]-ds[m].clean[:,None]
        x,y,xt,yt = dense(train,a),dense(train,b),dense(test,a),dense(test,b)
        s = load_sae_checkpoint(str(dictionary_path(a)))
        bases,_ = fixed_bases(s,train[a].clean[:SPLITS['train'][1]])
        freq = bases['frequent_sae'][0]
        q,r = orthogonal_parts(freq)
        cases = {'learned_orthogonal':[q],'learned_original':[freq], 'isotropic':bases['isotropic'],
                 'random_same_gram':[m@r for m in bases['isotropic']]}
        res,vec = {},{}
        for arm,rows in cases.items():
            result = [score(x@proj,y,xt@proj,yt) for proj in rows]
            cos = np.mean([rr[0]['cosine'] for rr in result],axis=0)
            vec[arm] = cos
            qs = [orthogonal_parts(proj)[0] for proj in rows]
            res[arm] = {'cosine':ci(cos),'penalty':[t[2] for t in result],
                        'train_edit_energy_fraction':ci(np.mean([norm_fraction(x[:128],v) for v in qs],axis=0)),
                        'reserve_edit_energy_fraction':ci(np.mean([norm_fraction(xt,v) for v in qs],axis=0))}
        for c,d in [('learned_orthogonal','isotropic'),('learned_original','random_same_gram'),
                    ('learned_original','learned_orthogonal')]:
            res[c+'_minus_'+d] = ci(vec[c]-vec[d])
        res['rank_over_width'] = len(r)/len(q)
        res['original_spectral_condition'] = float(np.linalg.cond(r))
        out[a+'_to_'+b] = res
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch',type=Path,default=DEFAULT_SCRATCH)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.scratch),indent=2)+'\n')
