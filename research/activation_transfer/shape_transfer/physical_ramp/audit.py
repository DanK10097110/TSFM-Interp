"""Independent spectral reconstruction of frozen displacement transport and live singleton forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.physical_ramp.study import OUT,OTHER,history,training
from tsfm_lens.extraction.hooks import ActivationCatcher


def run(source,out):
    x,ids,station,origin=history();target=OTHER[source]
    record=dict(np.load(out/(source+'_transfer.npz')))
    xtr,ytr,ordered,ps,pt=training(source)
    manifest=Path(__file__).resolve().parents[2]/'atoms/matched_windows/results/manifest.json'
    if not manifest.exists():
        from research.activation_transfer.atoms.matched_windows import patch
        manifest=Path(patch.__file__).parent/'results/manifest.json'
    np.testing.assert_equal(ordered,json.loads(manifest.read_text())['map_train_ids'])
    xtr=xtr.astype(float);xtr-=xtr.mean(0);ytr=ytr.astype(float);ytr-=ytr.mean(0)
    u,s,v=np.linalg.svd(xtr,full_matrices=False);recipient=(u.T@ytr)*(s/(s*s+.01*s[0]**2))[:,None]
    lift=lambda change:(change@v.T)@recipient
    donors=adapter_for(source);receiver=adapter_for(target)
    def capture(model,layer,row,pos):
        with torch.no_grad(),ActivationCatcher(model.module,[layer]) as cap:
            y=model.predict(row[None],16,[.5])['quantiles'][0,:,0].astype(float)
            h=model.postprocess_tokens(layer,cap.collect()[layer])[0,pos].detach().numpy().astype(float).mean(0)
        return h,y
    def changed(row,kind):
        z=row.copy();z[-8:]+=(np.arange(8)*(4/7)-2).astype('f4') if kind=='ramp' else np.float32(2.);return z
    # Identical float32 presentation of the linear dose; check vector values to rounding tolerance.
    def deltas(model,layer,row,pos,kind):
        h,base=capture(model,layer,row,pos);t,_=capture(model,layer,changed(row,kind),pos)
        return t-h,base
    rows=[]
    for i in (0,61,122,183):
        src=MODELS[source][-1];tar=MODELS[target][-1];other=int(record['wrong'][i])
        assert station[i]!=station[other] and origin[i]==origin[other]
        actual,base=deltas(receiver,tar,x[i],pt,'ramp')
        cand,_=deltas(donors,src,x[i],ps,'ramp');cand=lift(cand)
        diff,_=deltas(donors,src,x[i],ps,'step');diff=lift(diff)
        alt,_=deltas(donors,src,x[other],ps,'ramp');alt=lift(alt)
        vs=dict(truth=actual,mapped=cand,wrong_step=diff*np.linalg.norm(cand)/np.linalg.norm(diff),
                other_station=alt*np.linalg.norm(cand)/np.linalg.norm(alt))
        gaps={}
        for key,vec in vs.items():
            wanted=record['target_hidden' if key=='truth' else key+'_vec'][i]
            np.testing.assert_allclose(vec,wanted,atol=.0003,rtol=.0006)
            with add_native(receiver,tar,last_window_slice(receiver,pt),vec.astype('f4')):
                moved=receiver.predict(x[i:i+1],16,[.5])['quantiles'][0,:,0].astype(float)-base
            gap=moved-record[key][i];gaps[key]=float(np.max(abs(gap)))
            np.testing.assert_allclose(moved,record[key][i],atol=.00018,rtol=.0015)
        rows.append(dict(id=str(ids[i]),wrong_id=str(ids[other]),max_raw_celsius=gaps))
    (out/(source+'_replay.json')).write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=OTHER);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(1);run(a.model,a.out)
