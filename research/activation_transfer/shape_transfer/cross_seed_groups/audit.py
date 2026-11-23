"""Reconstruct physical interventions on singleton native calls using independently recaptured activations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.cross_seed_groups.study import OUT,models,population
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from tsfm_lens.extraction.extract import capture_raw_tokens


def run(name,pop,out):
    z=dict(np.load(out/(name+'_'+pop+'.npz')));fit=dict(np.load(out/(name+'_fit.npz')))
    x,_,pos,ids,*_=population(name,pop);np.testing.assert_equal(ids,z['ids'])
    sa,ta=models(name);a=adapter_for(name);site=MODELS[name][-1];sel=last_window_slice(a,pos);sd=z['sd']
    decoder=ta.W_dec.detach().numpy().astype(float);don=sa.W_dec.detach().numpy().astype(float)
    g=np.random.default_rng(int(fit['rotation_seed'])).standard_normal((decoder.shape[1],decoder.shape[1]))
    q,rng_r=np.linalg.qr(g)
    q*=np.sign(np.diag(rng_r));rot=decoder@q
    r,c=z['rows'],z['cols']
    co=z['target_z'][r,c]>0;sample=np.unique(np.r_[np.linspace(0,len(r)-1,3,dtype=int),np.flatnonzero(~co)[:1],np.flatnonzero(co)[:1]])
    rows=[]
    for e in sample:
        i,j=int(r[e]),int(c[e]);xx=x[i:i+1];atom=z['atoms'][j];target=fit['match_ids'][j]
        state=capture_raw_tokens(a,xx,[site])[site][:,pos].mean(1)
        with torch.inference_mode():
            s=sa.encode(state)[0,atom].item();t=ta.encode(state)[0,target].item()
        np.testing.assert_allclose([s,t],[z['z'][i,j],z['target_z'][i,j]],atol=8e-4,rtol=4e-4)
        dose=np.linalg.norm(s*don[atom]);writes=dict(donor=-s*don[atom],own=-t*decoder[target],forced=-dose*decoder[target]/np.linalg.norm(decoder[target]))
        for key in ('rank1','rank4','rank16','rotated16'):
            coeffs=fit[key+'_coeff'][j];vec=coeffs@(rot if key=='rotated16' else decoder)[fit[key+'_ids'][j]]
            if key=='rotated16':vec*=np.linalg.norm(writes['rank16'])/(s*np.linalg.norm(vec))
            writes[key]=-s*vec
        base=a.predict(xx,16,[.5])['quantiles'][0,:,0];gaps={}
        for key,v in writes.items():
            with add_native(a,site,sel,v.astype('f4')):
                trial=a.predict(xx,16,[.5])['quantiles'][0,:,0]-base
            expected=z[key][e];err=(trial-expected)/sd[i]
            np.testing.assert_allclose(trial/sd[i],expected/sd[i],atol=.00022,rtol=.002)
            gaps[key]=float(np.max(abs(err)))
        rows.append(dict(item=str(ids[i]),atom=int(atom),target=int(target),dose=s,target_dose=t,max_history_sd=gaps))
    (out/(name+'_'+pop+'_replay.json')).write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=('bolt','timesfm'));p.add_argument('pop',choices=('daily','monash_traffic','noaa'));p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();torch.set_num_threads(1);run(a.model,a.pop,a.out)
