"""Rebuild directions with an independent SVD and replay singleton native H16 forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tsfm_lens.extraction.hooks import ActivationCatcher
from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.physical_ramp.study import training
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.run import OUT, OLD, panel

def run(out,old=OLD):
    histories,ids,station,origin,_,_=panel()
    f=dict(np.load(out/'transfer.npz'))
    x,y,_,ps,pt=training('timesfm')
    x=x.astype(float);x-=x.mean(0);y=y.astype(float);y-=y.mean(0)
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    weight=(s/(s**2+.01*s[0]**2))[:,None]*(u.T@y)
    def lift(d):
        return (d@vt.T)@weight
    with np.load(old/'timesfm_native.npz') as frozen:
        generic=lift(np.mean(frozen['ramp_mean'],axis=0))
    np.testing.assert_allclose(generic,f['historical_direction'],rtol=1e-9,atol=1e-9)
    source=adapter_for('timesfm');recipient=adapter_for('bolt')
    def capture(a,site,x,pos):
        with torch.inference_mode(),ActivationCatcher(a.module,[site]) as cap:
            y=a.predict(x[None],16,[.5])['quantiles'][0,:,0].astype(float)
            w=a.postprocess_tokens(site,cap.collect()[site])[0,pos].detach().numpy().astype(float)
        return w.mean(0),y
    def read(a,name,x,pos,kind):
        loc=MODELS[name][-1]
        h,base=capture(a,loc,x,pos);changed=x.copy()
        changed[-8:]+=(np.arange(8)*(4/7)-2).astype('f4') if kind=='ramp' else np.float32(2.)
        t,_=capture(a,loc,changed,pos)
        return t-h,base
    logs=[]
    for i in (0,192,384):
        other=int(f['wrong'][i]);assert station[i]!=station[other] and origin[i]==origin[other]
        native,base=read(recipient,'bolt',histories[i],pt,'ramp')
        orig,source_base=read(source,'timesfm',histories[i],ps,'ramp')
        step,_=read(source,'timesfm',histories[i],ps,'step');wrong,_=read(source,'timesfm',histories[other],ps,'ramp')
        mapped=lift(orig);dose=np.linalg.norm(mapped)
        def normalize(d):
            return dose*d/np.linalg.norm(d)
        moves=dict(truth=native,mapped=mapped,wrong_step=normalize(lift(step)),other_station=normalize(lift(wrong)),historical=normalize(generic))
        gaps={}
        for arm,v in moves.items():
            archive=f['hidden_truth' if arm=='truth' else arm+'_vec'][i]
            np.testing.assert_allclose(v,archive,atol=.00032,rtol=.0006)
            with add_native(recipient,MODELS['bolt'][-1],last_window_slice(recipient,pt),v.astype('f4')):
                wave=recipient.predict(histories[i:i+1],16,[.5])['quantiles'][0,:,0].astype(float)-base
            gaps[arm]=float(np.abs(wave-f[arm][i]).max())
            np.testing.assert_allclose(wave,f[arm][i],atol=.00018,rtol=.0015)
        with add_native(source,MODELS['timesfm'][-1],last_window_slice(source,ps),orig.astype('f4')):
            donor=source.predict(histories[i:i+1],16,[.5])['quantiles'][0,:,0].astype(float)-source_base
        gaps['donor_native']=float(np.abs(donor-f['donor_native'][i]).max())
        np.testing.assert_allclose(donor,f['donor_native'][i],atol=.00018,rtol=.0015)
        logs.append(dict(id=str(ids[i]),other_station=str(ids[other]),max_raw_celsius=gaps))
    (out/'replay.json').write_text(json.dumps(logs,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--historical',type=Path,default=OLD)
    args=parser.parse_args();torch.set_num_threads(1);run(args.out,args.historical)
