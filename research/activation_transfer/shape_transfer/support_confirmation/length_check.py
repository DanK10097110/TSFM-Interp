"""Compare native dictionary identities and derivative geometry on the same old items at two lengths."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as OLD
from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE,OUT,probe
from research.activation_transfer.shape_transfer.support_confirmation.supports import active_ids,basis,count_overlap,fixed_directions,select_supports
from tsfm_lens.sae.train import load_sae_checkpoint


def prepare(cache):
    with np.load(OLD/'data.npz') as f:
        select=f['split']=='daily';x=np.asarray(f['x'][select,-256:],float);ids=f['ids'][select]
    x=((x-x.mean(1)[:,None])/x.std(1)[:,None]).astype('f4')
    cache.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache/'data.npz',x=x,ids=ids,split=np.full(len(x),'daily'))


def check(model,cache,out):
    probe(model,cache,8)
    with np.load(cache/'data.npz') as f:items=f['ids']
    with np.load(OLD/'data.npz') as f:use=f['split']=='daily';np.testing.assert_array_equal(f['ids'][use],items)
    dct=load_sae_checkpoint(str(dictionary_path(model)));dec=dct.W_dec.detach().numpy()
    ranking,_,fixed,_=fixed_directions(model,dct)
    report={};rec=dict(ids=items)
    for key,loc,ix in [('480',OLD,np.flatnonzero(use)),('256',cache,np.arange(len(items)))]:
        h,pos,ids=read_materials(loc,model)
        np.testing.assert_array_equal(ids[ix],items)
        with np.load(loc/(model+'_early.npz')) as f:j=f['jac'][ix].astype(float)
        jc=j-j.mean(1,keepdims=True)
        current=active_ids(dct,h[ix].mean(1));supports,assignment=select_supports(current,ranking)
        q=basis(dec[current]);other=q[assignment['swap_2301']]
        s=dict(native=q,other=other,frequent=basis(dec[ranking[:32]]),pca=fixed['pca'])
        rec[key+'_ids']=current;rec[key+'_overlap']=count_overlap(supports['native'],supports['swap_2301'])
        rec[key+'_positions']=pos
        for label,axes in s.items():
            projected=jc@axes.swapaxes(-2,-1)
            fraction=np.sum(projected**2,axis=(1,2))/np.sum(jc**2,axis=(1,2))
            rec[key+'_'+label]=fraction
    within=count_overlap(rec['480_ids'],rec['256_ids'])
    rec['same_item_overlap']=within
    draws=resampling_counts(len(items),draws=3000)
    def mean_ci(a):
        return dict(mean=float(a.mean()),median=float(np.median(a)),ci95=np.quantile(draws@a/len(a),[.025,.975]).tolist())
    report['within_item_overlap']=mean_ci(within)
    for key in ('overlap','native','other','frequent','pca'):
        a=rec['256_'+key];b=rec['480_'+key]
        report[key]={'256':mean_ci(a),'480':mean_ci(b),'short_minus_long':mean_ci(a-b)}
    np.savez_compressed(out/(model+'_context_check.npz'),**rec)
    (out/(model+'_context_check.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['data',*MODELS])
    p.add_argument('--cache',type=Path,default=CACHE/'old256');p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='data':prepare(a.cache)
    else:check(a.stage,a.cache,a.out)
