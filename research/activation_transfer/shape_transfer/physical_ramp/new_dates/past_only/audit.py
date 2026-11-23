"""Use Python dates, rowwise projection, and independent additive singleton hooks."""
import argparse
from datetime import date
import json
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.past_only.study import OUT,OLD,NEW,panel

def run(out):
    bundle=dict(np.load(out/'cases.npz'))
    x,ids,*_=panel()
    with np.load(OLD/'timesfm_native.npz') as source:
        ids_old=source['ids'];old_delta=source['ramp_mean']
    with np.load(OLD/'timesfm_map.npz') as fit:
        matrix=fit['linear'];pos=fit['target_pos']
    with np.load(NEW/'bolt_native.npz') as native:
        native_vec=native['ramp']
    receiver=adapter_for('bolt');layer=MODELS['bolt'][-1];sel=last_window_slice(receiver,pos)
    mapped_archive=old_delta@matrix
    checks=[]
    for j in [0,174,349]:
        i=int(bundle['case_indices'][j]);assert bundle['ids'][j]==ids[i]
        end=date.fromisoformat(str(ids[i]).rsplit('/',1)[1])
        donors=[k for k,entry in enumerate(ids_old) if date.fromisoformat(str(entry).rsplit('/',1)[1])<end]
        vec=mapped_archive[donors].mean(0)
        vec*=bundle['dose'][j]/np.linalg.norm(vec)
        np.testing.assert_allclose(vec,bundle['write'][j],rtol=1e-10,atol=1e-12)
        probs=receiver.predict(x[i:i+1],16,[.5])['quantiles'][0,:,0].astype(float)
        gaps={}
        for key,v in [('past_only',vec),('truth',native_vec[i])]:
            with add_native(receiver,layer,sel,v.astype('f4')):
                moved=receiver.predict(x[i:i+1],16,[.5])['quantiles'][0,:,0].astype(float)-probs
            np.testing.assert_allclose(moved,bundle[key][j],atol=.00018,rtol=.0015)
            gaps[key]=float(np.max(abs(moved-bundle[key][j])))
        checks.append(dict(id=str(ids[i]),included=len(donors),max_raw_celsius=gaps))
    (out/'replay.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    torch.set_num_threads(1);run(args.out)
