"""Measure all subsets of a three-part decomposition of an actual clean-token restoration."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.level_pair.inputs import HERE,CACHE
from research.activation_transfer.shape_transfer.level_pair.run import NAMED,capture,decode_states,replacement_forecast
from research.activation_transfer.shape_transfer.level_pair.score import half_difference,mean_stat
from tsfm_lens.sae.train import load_sae_checkpoint


def run(name,cache,out,batch):
    f=dict(np.load(out/'inputs.npz'));daily=f['split']=='daily';ids=f['ids'][daily];edits=f['edits'][daily,0]
    adapter=adapter_for(name);layer=MODELS[name][-1];h0,pos,refids=read_materials(cache,name)
    np.testing.assert_array_equal(refids,f['ids']);h0=h0[daily];select=last_window_slice(adapter,pos)
    atom=NAMED[name];sae=load_sae_checkpoint(str(dictionary_path(name)));row=sae.W_dec.detach().numpy()[atom]
    z0,_,d0=decode_states(sae,h0,atom)
    labels={'O':2,'R':4,'OR':6,'NR':5}
    record={key:np.empty((len(ids),2,8),dtype='f4') for key in labels};record['ids']=ids
    for j in range(2):
        x=edits[:,j];h=capture(adapter,layer,pos,x,batch);z,_,d=decode_states(sae,h,atom)
        n=((z0-z)[:,None]*row)[:,None];o=(d0-d)[:,None]-n;r=h0-h-(d0-d)[:,None]
        np.testing.assert_allclose(h+n+o+r,h0,atol=2e-5,rtol=1e-5)
        for arm,bits in labels.items():
            combined=h.astype(float)
            for bit,delta in [(1,n),(2,o),(4,r)]:
                if bits & bit:combined=combined+delta
            record[arm][:,j]=replacement_forecast(adapter,layer,select,x,combined.astype('f4'),batch)
    np.savez_compressed(out/(name+'_factorial.npz'),**record)


def order_average(lift):
    values={k:np.zeros_like(lift[0]) for k in [1,2,4]}
    for order in itertools.permutations([1,2,4]):
        prior=0
        for k in order:
            new=prior|k
            values[k]+=(lift[prior]-lift[new])/6
            prior=new
    np.testing.assert_allclose(sum(values.values()),lift[0]-lift[7],atol=1e-12)
    return values


def score(out):
    report={}
    for name in MODELS:
        with np.load(out/(name+'_effects.npz')) as f,np.load(out/(name+'_factorial.npz')) as g:
            mask=f['split']=='daily';np.testing.assert_array_equal(f['ids'][mask],g['ids'])
            curves={k:half_difference(f[a][mask,0]) for k,a in [(0,'intact'),(1,'named'),(3,'all'),(7,'tokens')]}
            curves.update({k:half_difference(g[a]) for k,a in [(2,'O'),(4,'R'),(6,'OR'),(5,'NR')]})
            lift={k:y.mean(axis=1) for k,y in curves.items()};draws=resampling_counts(len(lift[0]));item={};report[name]=item
            item['subset_lifts']={str(k):mean_stat(v,draws) for k,v in lift.items()}
            values=order_average(lift)
            for k,label in [(1,'named'),(2,'other_decoded'),(4,'residual')]:
                first=lift[0]-lift[k];last=lift[7^k]-lift[7]
                item[label]=dict(first=mean_stat(first,draws),last=mean_stat(last,draws),
                    last_minus_first=mean_stat(last-first,draws),order_average=mean_stat(values[k],draws))
            item['all_three']=mean_stat(lift[0]-lift[7],draws)
    (out/'factorial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=(*MODELS,'score'));p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--batch',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='score':score(a.out)
    else:run(a.stage,a.cache,a.out,a.batch)
