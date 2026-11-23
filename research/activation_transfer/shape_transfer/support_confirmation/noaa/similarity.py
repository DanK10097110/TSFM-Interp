"""Describe the severity of the actual derivative reassignments using ordinary forecast derivatives alone."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.support_confirmation.data import CACHE as M4CACHE,OUT as M4OUT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import CACHE,OUT
from research.activation_transfer.shape_transfer.support_confirmation.noaa.run import SEEDS,station_derangement


def vectors(j,assignments):
    result={}
    for label,w in [('full',j),('shape',j-j.mean(1,keepdims=True))]:
        h=w.reshape(len(j),-1);norm=np.linalg.norm(h,axis=1)
        cos=[];rat=[]
        for i in assignments:
            cos.extend(np.einsum('nd,nd->n',h,h[i])/(norm*norm[i]))
            rat.extend(norm[i]/norm)
        result[label]=dict(cosine_p10_p50_p90=np.quantile(cos,[.1,.5,.9]).tolist(),
                           other_to_current_norm_p10_p50_p90=np.quantile(rat,[.1,.5,.9]).tolist())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--m4-cache',type=Path,default=M4CACHE);p.add_argument('--m4-out',type=Path,default=M4OUT);a=p.parse_args()
    with np.load(a.cache/'data.npz') as f:station,origin,sd=f['station'],f['origin'],f['sd']
    asgn=[station_derangement(station,origin,k) for k in SEEDS];report={}
    for model in ('bolt','timesfm'):
        with np.load(a.cache/(model+'_early.npz')) as f:jac=f['jac'].astype(float)/sd[:,None,None]
        m={'noaa_same_origin':vectors(jac,asgn)}
        for panel,cache in [('short',a.m4_cache/'short'),('long',a.m4_cache)]:
            if not (cache/(model+'_early.npz')).exists() or not (a.m4_out/(panel+'_'+model+'.npz')).exists():
                m['m4_'+panel]=None
                continue
            with np.load(a.m4_out/(panel+'_'+model+'.npz')) as f:old=[f[f'swap_{k}_assignment'] for k in SEEDS]
            with np.load(cache/(model+'_early.npz')) as f:orig=f['jac'].astype(float)
            m['m4_'+panel]=vectors(orig,old)
        report[model]=m
    (a.out/'jacobian_similarity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
