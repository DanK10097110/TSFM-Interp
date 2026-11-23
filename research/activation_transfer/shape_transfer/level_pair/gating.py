"""Measure native TopK cutoffs and compare controlled recent levels shared across domains."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.level_pair.inputs import CACHE,HERE,statistic_rows
from research.activation_transfer.shape_transfer.level_pair.run import NAMED,capture
from tsfm_lens.sae.train import load_sae_checkpoint

CONDITIONS=('clean','minus','plus','native_low','native_high','joint_low','joint_high')


def coordinate_targets(x):
    x=np.asarray(x,float);m=x.mean(1,keepdims=True);v=x-m;sd=x.std(1)
    rows=statistic_rows(x.shape[1]);dual=np.linalg.solve(rows@rows.T,rows)
    stats=v@rows.T;res=v-stats@dual;energy=(v*v).sum(1);resnorm=(res*res).sum(1)
    out={};feasible={};achieved={}
    for mode in ('native','joint'):
        for name,level in [('low',0.),('high',1.25)]:
            key=mode+'_'+name;goals=stats.copy();goals[:,0]=level*sd
            if mode=='joint':goals[:,1]=.5*sd;goals[:,2]=0.
            projection=goals@dual;remaining=energy-(projection*projection).sum(1);ok=remaining>=0
            value=(m+projection+res*np.sqrt(np.maximum(remaining,0)/resnorm)[:,None]).astype('f4')
            np.testing.assert_allclose(value[ok].astype(float)@rows.T,goals[ok],rtol=2e-6,atol=4e-6)
            np.testing.assert_allclose(value[ok].astype(float).std(1),sd[ok],atol=1e-6)
            np.testing.assert_allclose(value[ok].astype(float).mean(1),m[ok,0],atol=1e-6)
            out[key]=value;feasible[key]=ok;achieved[key]=value.astype(float)@rows.T
        common=feasible[mode+'_low']&feasible[mode+'_high']
        feasible[mode+'_low']=common;feasible[mode+'_high']=common
    return out,feasible,achieved


def cutoffs(sae,tokens,named):
    with torch.inference_mode():
        pre=sae.pre_activations(torch.from_numpy(tokens.mean(axis=1)))
        code=sae.sparsify(pre)[:,named];value=pre[:,named].clone()
        rank=1+(pre>value[:,None]).sum(1)
        pre[:,named]=-torch.inf
        cutoff=torch.topk(pre,sae.k,dim=1).values[:,-1].clamp_min(0)
        margin=value-cutoff
        nonsingular=margin!=0
        np.testing.assert_array_equal((code[nonsingular]>0).numpy(),(margin[nonsingular]>0).numpy())
    return {k:t.numpy() for k,t in dict(pre=value,rank=rank,cutoff=cutoff,margin=margin,z=code).items()}


def run(model,cache,out,batch):
    xrec=dict(np.load(out/'inputs.npz'));x=xrec['inputs'];created,feasible,achieved=coordinate_targets(x)
    xs={'clean':x,'minus':xrec['edits'][:,0,0],'plus':xrec['edits'][:,0,1],**created}
    enc=adapter_for(model);layer=MODELS[model][-1];sae=load_sae_checkpoint(str(dictionary_path(model)))
    clean,pos,ids=read_materials(cache,model);np.testing.assert_array_equal(ids,xrec['ids'])
    record=dict(ids=ids,split=xrec['split'],conditions=np.array(CONDITIONS),atom=NAMED[model],k=sae.k,
                native_stats=xrec['clean_stats'],width=sae.W_dec.shape[0])
    for name,obs in xs.items():
        ok=feasible.get(name,np.ones(len(obs),bool))
        token=clean if name=='clean' else capture(enc,layer,pos,obs[ok],batch)
        fields=cutoffs(sae,token,NAMED[model]);record[name+'_eligible']=ok
        record[name+'_stats']=achieved.get(name,obs.astype(float)@statistic_rows(obs.shape[1]).T)
        for k,v in fields.items():
            values=np.full(len(obs),np.nan);values[ok]=v;record[name+'_'+k]=values
    np.savez_compressed(out/(model+'_gating.npz'),**record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',choices=MODELS);p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--batch',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(1);run(a.model,a.cache,a.out,a.batch)
