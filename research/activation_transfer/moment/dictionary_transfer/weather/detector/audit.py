"""Independent full-native singleton replay, prioritizing receiver-only false positives."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tsfm_lens.extraction.hooks import ActivationCatcher,token_patch
from research.activation_transfer.moment.dictionary_transfer.match import dictionary
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import OUT,TARGET
from research.activation_transfer.moment.dictionary_transfer.weather.study import histories
from research.activation_transfer.moment.token_paths.geometry import load,layers


def run(out):
    fit=dict(np.load(out/'fit.npz'));ev=dict(np.load(out/'evaluation.npz'));probes=dict(np.load(out/'inferred.npz'))
    x,ids,*_=histories();np.testing.assert_equal(ids,ev['ids'])
    wanted=[]
    for src,det in [(False,True),(True,True),(True,False),(False,False)]:
        ix=np.argwhere(((ev['source_dose']>0)==src)&((probes['dose']>0)==det))
        if len(ix):wanted.append(tuple(map(int,ix[len(ix)//2])))
    model=load('large');sae=dictionary('large',TARGET);site=layers(model)['middle'];outcomes=[]
    for r,j in wanted:
        row=x[r:r+1]
        with ActivationCatcher(model.module,[site]) as cap:
            ordinary=model.predict(row,16,[.5])['point'][0].astype(float)
            all_hidden=cap.collect()[site];slot=model.token_slice(all_hidden.shape[1]).stop-1;h=all_hidden[:,slot]
        with torch.inference_mode():z=sae.encode(h)[0].numpy().astype(float)
        y=((z-fit['center'])*fit['inverse_sd'])@fit['slope'][:,j]+fit['bias'][j]
        dose=y if y>=fit['threshold'][j] and y>0 else 0.
        np.testing.assert_allclose(dose,probes['dose'][r,j],atol=.001,rtol=2e-4)
        dec=sae.W_dec.detach().numpy().astype(float)
        group=fit['group_coeff'][j]@dec[fit['group_ids'][j]]
        moves={'inferred':-dose*group,'natural':-z[fit['match'][j]]*dec[fit['match'][j]]}
        gaps={}
        for name,vec in moves.items():
            change=(h+torch.from_numpy(vec.astype('f4')))[None]
            with token_patch(model.module,site,lambda n:slice(model.token_slice(n).stop-1,model.token_slice(n).stop),change):
                y=model.predict(row,16,[.5])['point'][0].astype(float)-ordinary
            np.testing.assert_allclose(y,ev[name][r,j],atol=.00020,rtol=.002)
            gaps[name]=float(np.max(abs(y-ev[name][r,j])))
        outcomes.append(dict(item=str(ids[r]),atom=int(fit['atoms'][j]),dose=float(dose),donor_code=float(ev['source_dose'][r,j]),
                             max_abs_diff_celsius=gaps))
    (out/'replay.json').write_text(json.dumps(outcomes,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(1);run(a.out)
