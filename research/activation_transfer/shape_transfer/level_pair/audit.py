"""Reconstruct the constrained inputs independently; replay signed effects with live additive hooks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native,point
from research.activation_transfer.shape_transfer.level_pair.inputs import HERE
from research.activation_transfer.shape_transfer.level_pair.run import ARMS
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.sae.train import load_sae_checkpoint


def alternative_inputs(x):
    n=480;a=np.zeros((4,n));a[0]=1/n;a[1,-32:]=1/32;a[2,-1]=1
    t=np.linspace(-.5,.5,32);a[3,-32:]=t/np.dot(t,t)
    right=np.linalg.pinv(a);obs=np.asarray(x,float);targets=obs@a.T;projection=targets@right.T
    residual=obs-projection;e=np.sum(obs**2,axis=1);left=np.sum(residual**2,axis=1);sd=obs.std(axis=1)
    reference=np.zeros((len(obs),4,2,n),dtype=float)
    for c,dose in enumerate([.25,.5,.5]):
        d=right[:,c+1][None]*dose*sd[:,None]
        lo=np.zeros(len(obs));hi=np.ones(len(obs))
        for _ in range(52):
            middle=(lo+hi)/2;step=middle[:,None]*d
            feasible=(np.sum((projection+step)**2,axis=1)<=e)&(np.sum((projection-step)**2,axis=1)<=e)
            lo=np.where(feasible,middle,lo);hi=np.where(feasible,hi,middle)
        for j,sign in enumerate([-1,1]):
            shifted=projection+sign*lo[:,None]*d
            reference[:,c,j]=shifted+np.sqrt(np.maximum(e-np.sum(shifted**2,axis=1),0)/left)[:,None]*residual
    for j,sgn in enumerate([-1,1]):reference[:,3,j]=obs+sgn*.25*sd[:,None]
    return reference.astype('f4')


def code_and_recon(sae,tokens):
    v=torch.from_numpy(tokens.mean(0,keepdims=True))
    with torch.inference_mode():
        pre=(v-sae.b_dec)@sae.W_enc+sae.b_enc
        values,cols=torch.topk(torch.relu(pre),k=sae.k,dim=1)
        reconstruction=(values[0,:,None]*sae.W_dec[cols[0]]).sum(dim=0)+sae.b_dec
    return dict(zip(cols[0].tolist(),values[0].tolist())),reconstruction.numpy()


def check_model(model,out,inputs):
    f=dict(np.load(out/(model+'_effects.npz')));factor=dict(np.load(out/(model+'_factorial.npz')))
    factor_rows={str(value):i for i,value in enumerate(factor['ids'])};enc=adapter_for(model);layer=MODELS[model][-1];select=last_window_slice(enc,f['positions'])
    sae=load_sae_checkpoint(str(dictionary_path(model)));dec=sae.W_dec.detach().numpy().astype(float);atom=int(f['atoms'][0])
    selected=[]
    for domain in np.unique(f['split']):
        mask=f['split']==domain
        active=(f['edited_z'][:,0].max(axis=1)>0)|(f['clean_z']>0)
        rows=np.flatnonzero(mask&active)[:1].tolist()+np.flatnonzero(mask&~active)[:1].tolist()
        for i in np.flatnonzero(mask):
            if len(rows)>=2:break
            if int(i) not in rows:rows.append(int(i))
        selected+=rows
    checks=[]
    def hidden(x):return capture_raw_tokens(enc,x[None],[layer])[layer][0,f['positions']].numpy()
    for i in selected:
        clean=hidden(inputs['inputs'][i]);z0,d0=code_and_recon(sae,clean);zclean=z0.get(atom,0.)
        for c in range(4):
            for j in range(2):
                obs=inputs['edits'][i,c,j];h=hidden(obs);z,d=code_and_recon(sae,h);zedit=z.get(atom,0.)
                np.testing.assert_allclose(zedit,f['edited_z'][i,c,j],atol=.00016,rtol=.0001)
                named=(zclean-zedit)*dec[atom];deltas={'named':named,'all':d0-d,'tokens':clean-h}
                for q,k in enumerate(f['atoms'][1:]):deltas['control'+str(q)]=(zclean-zedit)*np.linalg.norm(dec[atom])*dec[k]/np.linalg.norm(dec[k])
                item=dict(id=str(f['ids'][i]),cue=str(f['cues'][c]),sign=j*2-1,errors={})
                for arm in ARMS:
                    if arm=='intact':actual=point(enc,obs)
                    else:
                        with add_native(enc,layer,select,deltas[arm]):actual=point(enc,obs)
                    ref=f[arm][i,c,j];np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=.002)
                    item['errors'][arm]=float(np.max(abs(actual-ref)))
                item['factorial_errors']={}
                if c==0 and str(f['ids'][i]) in factor_rows:
                    pos=factor_rows[str(f['ids'][i])]
                    other=(d0-d)[None]-named[None]
                    residual=(clean-h).astype(float)-(d0-d)[None]
                    moves={'O':other,'R':residual,'OR':clean-h-named[None],'NR':named[None]+residual}
                    for label,delta in moves.items():
                        with add_native(enc,layer,select,delta):returned=point(enc,obs)
                        wanted=factor[label][pos,j]
                        np.testing.assert_allclose(returned,wanted,atol=.00012,rtol=.002)
                        item['factorial_errors'][label]=float(abs(returned-wanted).max())
                post,_=code_and_recon(sae,(h+named.astype('f4')).astype('f4'))
                np.testing.assert_allclose(post.get(atom,0.),f['named_lock_z'][i,c,j],atol=2e-4,rtol=.0002)
                checks.append(item)
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args();torch.set_num_threads(1)
    inp=dict(np.load(a.out/'inputs.npz'));other=alternative_inputs(inp['inputs']);np.testing.assert_allclose(other,inp['edits'],atol=6e-5,rtol=2e-6)
    report={'independent_edit_abs_max':float(np.max(abs(other-inp['edits']))),'models':{m:check_model(m,a.out,inp) for m in MODELS}}
    (a.out/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
