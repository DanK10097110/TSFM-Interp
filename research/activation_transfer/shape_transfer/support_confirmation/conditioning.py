"""Measure conditioning of native decoder rows, residual spans, and the balanced forecast design."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT
from research.activation_transfer.shape_transfer.support_confirmation.residual import PAIRS,orthogonal_part
from research.activation_transfer.shape_transfer.support_confirmation.supports import basis,fixed_directions
from tsfm_lens.sae.train import load_sae_checkpoint


def summarize(v):
    return np.quantile(v,[.1,.5,.9]).tolist()


def measure(cache,out):
    main=dict(np.load(out/'short_bolt.npz'))
    model=load_sae_checkpoint(str(dictionary_path('bolt')));decoder=model.W_dec.detach().numpy()
    _,_,static,_=fixed_directions('bolt',model)
    d={key:decoder[main[key+'_ids']].astype(float) for key in ('native','own_0')}
    d.update({k:np.broadcast_to(q,(len(main['ids']),*q.shape)) for k,q in static.items()})
    axes={k:basis(v) for k,v in d.items()}
    with np.load(cache/'bolt_early.npz') as f:j=boost_shape(f['jac'].astype(float),float(main['weight']))
    output={}
    for arm,(source,remove) in PAIRS.items():
        orth,strength,ranks,_=orthogonal_part(axes[source],axes[remove])
        before=np.linalg.svd(d[source],compute_uv=False)
        design_before=np.linalg.svd(j@axes[source].swapaxes(1,2),compute_uv=False)
        design_after=np.linalg.svd(j@orth.swapaxes(1,2),compute_uv=False)
        def ratios(values):return summarize(values[:,0]/np.maximum(values[:,-1],1e-14))
        output[arm]=dict(q10_q50_q90=dict(original_raw_row_condition=ratios(before),
            original_orthonormal_row_condition=[1.,1.,1.],residual_row_condition=ratios(strength),
            original_balanced_eight_lead_condition=ratios(design_before),residual_balanced_eight_lead_condition=ratios(design_after),
            baseline_min_output_singular=summarize(design_before[:,-1]),residual_min_output_singular=summarize(design_after[:,-1])),
            min_residual_rank=int(ranks.min()),min_relative_lstsq_regularizer=.001)
    (out/'conditioning.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=CACHE/'short')
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(1);measure(a.cache,a.out)
