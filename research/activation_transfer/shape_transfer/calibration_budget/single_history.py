"""Follow up at 48 and 96 calibration rows on the two separate validation domains."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.atoms.scores import aggregate,ci,resampling_counts
from research.activation_transfer.boundary.patch_forecast import change_forecasts,last_window_slice
from research.activation_transfer.shape_transfer.calibration_budget.study import (
    HERE,FRESH_CACHE,FRESH_RESULTS,FORWARD,SEEDS,materials)
from research.activation_transfer.shape_transfer.forward_only.transfer import fit_library
from research.activation_transfer.shape_transfer.full_space.transfer import apply


def fit_and_apply(source,fresh,out,batch):
    target=next(m for m in MODELS if m!=source)
    with np.load(FORWARD/'results'/(target+'_probes.npz')) as z,np.load(out/(target+'_libraries.npz')) as f:
        jac=z['pc_jac'].astype(float);basis=f['basis'];weight=f['weight'];radius=float(f['radius']);cal_ids=f['cal_ids']
    x,h,pos,ids,labels=materials(target,fresh)
    with np.load(out/(source+'_fresh_probes.npz')) as f:
        np.testing.assert_array_equal(ids,f['ids']);codes=f['codes'];request=f['test_decoder_der'].transpose(0,2,1);atoms=f['atoms']
    model=adapter_for(target);select=last_window_slice(model,pos);layer=MODELS[target][-1]
    base=change_forecasts(model,layer,x,h,None,select,batch)
    state=(model,layer,select,x,h,base,codes)
    record=dict(ids=ids,split=labels,atoms=atoms,codes=codes,cal_ids=cal_ids)
    for n,k,rep in [(1,24,8),(3,8,3),(2,24,3)]:
        arm=f'{n}x{k}';effects=[];clips=[];subsets=[];libraries=[]
        for seed in range(SEEDS[0],SEEDS[0]+rep):
            rows=np.random.default_rng(seed).permutation(len(cal_ids))[:n]
            lib,_=fit_library(jac[rows,:,:k],basis[:k],weight)
            e,c=apply(*state,lib@request,radius,batch)
            subsets.append(rows);libraries.append(lib);effects.append(e);clips.append(c)
        record[arm],record[arm+'_clip']=np.stack(effects),np.stack(clips)
        record[arm+'_rows'],record[arm+'_library']=np.stack(subsets),np.stack(libraries)
        f=out/(source+'_single.npz');tmp=out/(source+'_single.partial.npz');np.savez_compressed(tmp,**record);tmp.replace(f)
        print(source,arm,rep,'fits',flush=True)


def score(out):
    report={}
    for source in MODELS:
        with np.load(out/(source+'_single.npz')) as f,np.load(out/(source+'_fresh.npz')) as p,np.load(FRESH_RESULTS/(source+'_native.npz')) as t:
            for z in (f,t): np.testing.assert_array_equal(z['ids'],p['ids']);np.testing.assert_array_equal(z['atoms'],p['atoms'])
            result={};report[source]=result
            for domain in np.unique(f['split']):
                keep=f['split']==domain;mask=f['codes'][keep]>0;k=mask.sum(0)>=8;mask=mask[:,k]
                draws=resampling_counts(len(mask));result[str(domain)]=domain_result={}
                for center in (False,True):
                    cell={};domain_result['centered' if center else 'full']=cell;points={};boots={}
                    truth=t['native'][keep][:,k].astype(float)
                    if center:truth-=truth.mean(axis=-1,keepdims=True)
                    for arm,arr in [('baseline',p['baseline'][None]),('1x24',f['1x24']),('3x8',f['3x8']),('2x24',f['2x24'])]:
                        point=[];boot=[]
                        for values in arr:
                            pred=values[keep][:,k].astype(float)
                            if center:pred-=pred.mean(axis=-1,keepdims=True)
                            _,pt,b=aggregate(mask,pred,truth,draws);point.append(pt['skill']);boot.append(b['skill'])
                        points[arm],boots[arm]=np.array(point),np.stack(boot)
                        cell[arm]=dict(skill=ci(np.mean(point),np.mean(boot,axis=0)),by_calibration_fit=point)
                        if arm!='baseline':
                            delta=points[arm].mean()-points['baseline'][0];dist=boots[arm].mean(0)-boots['baseline'][0]
                            cell[arm]['minus_baseline']=ci(delta,dist)
                            cell[arm]['prior_three_minus_baseline']=ci(points[arm][:3].mean()-points['baseline'][0],boots[arm][:3].mean(0)-boots['baseline'][0])
                    cell['1x24_minus_3x8_matched']=ci((points['1x24'][:3]-points['3x8']).mean(),(boots['1x24'][:3]-boots['3x8']).mean(axis=0))
    (out/'single_scores.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('bolt','timesfm','score'))
    p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--fresh',type=Path,default=FRESH_CACHE);p.add_argument('--batch',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='score':score(a.out)
    else:fit_and_apply(a.stage,a.fresh,a.out,a.batch)
