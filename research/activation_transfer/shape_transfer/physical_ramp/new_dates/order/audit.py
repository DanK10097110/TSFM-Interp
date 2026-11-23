"""Three independent singleton replications of both target references and all four mapped writes."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tsfm_lens.extraction.hooks import ActivationCatcher
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.physical_ramp.study import training
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.run import panel
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.order.study import OUT

def main(out):
    f=dict(np.load(out/'order.npz'));weather,ids,*_=panel()
    x,y,_,ps,pt=training('timesfm')
    x=x.astype(float);x-=x.mean(0);y=y.astype(float);y-=y.mean(0)
    u,s,v=np.linalg.svd(x,full_matrices=False);rhs=(s/(s**2+.01*s[0]**2))[:,None]*(u.T@y)
    source=adapter_for('timesfm');target=adapter_for('bolt')
    def cap(a,site,pos,raw):
        with torch.no_grad(),ActivationCatcher(a.module,[site]) as catcher:
            response=a.predict(raw[None],16,[.5])['quantiles'][0,:,0].astype(float)
            rep=a.postprocess_tokens(site,catcher.collect()[site])[0,pos].detach().numpy().astype(float)
        return rep.mean(0),response
    def native_pair(a,name,pos,x):
        site=MODELS[name][-1]
        clean,base=cap(a,site,pos,x)
        states={}
        inc=(np.arange(8)*4./7.-2.).astype('f4')
        for goal in ('A','B'):
            change=inc.copy()
            if goal=='B':change[1:7]=inc[1:7][::-1]
            new=x.copy();new[-8:]+=change;h,_=cap(a,site,pos,new)
            states[goal]=h-clean
        return states,base
    rows=[]
    for i in (0,192,384):
        donor,_=native_pair(source,'timesfm',ps,weather[i]);native,base=native_pair(target,'bolt',pt,weather[i])
        vecs={g:(val@v.T)@rhs for g,val in donor.items()}
        maximum={}
        def patch(label,vector):
            with add_native(target,MODELS['bolt'][-1],last_window_slice(target,pt),vector.astype('f4')):
                wave=target.predict(weather[i:i+1],16,[.5])['quantiles'][0,:,0].astype(float)-base
            maximum[label]=float(abs(wave-f[label][i]).max())
            np.testing.assert_allclose(wave,f[label][i],atol=.00018,rtol=.0015)
        for g in 'AB':
            patch('truth_'+g,native[g])
            for donor_id in 'AB':
                code=g+donor_id;candidate=vecs[donor_id]*np.linalg.norm(vecs[g])/np.linalg.norm(vecs[donor_id])
                np.testing.assert_allclose(candidate,f[code+'_vec'][i],atol=.00032,rtol=.0006)
                patch(code,candidate)
        rows.append(dict(id=str(ids[i]),max_raw_celsius=maximum))
    (out/'replay.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();torch.set_num_threads(1);main(args.out)
