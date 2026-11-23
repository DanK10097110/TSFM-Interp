"""Reprobe fixed station origins, independently apply both edits and reconstruct hard TopK in NumPy."""
import json

import numpy as np
import torch
from scipy.stats import rankdata

from research.activation_transfer.moment.dictionary_transfer.match import dictionary
from research.activation_transfer.moment.dictionary_transfer.semantics.core import OUT, SEED, TRANSFER
from research.activation_transfer.moment.dictionary_transfer.semantics.evaluate import weather
from research.activation_transfer.moment.token_paths.geometry import load


def main():
    torch.set_num_threads(1)
    station_ids, raw, atoms, _, _ = weather()
    pick = np.arange(4)*4
    a=load("large")
    last=a.patch_layout(480,16)[0].stop-1
    assert last == 61 and tuple(a.token_time_spans()[-1]) == (472,480)
    site=a.module.encoder.block[11]
    sae=dictionary("large",SEED)
    w,b,c=(s.detach().numpy().astype(float) for s in (sae.W_enc,sae.b_enc,sae.b_dec))
    assert sae.k==32
    records={}
    with np.load(TRANSFER/"input_deltas/results/large_native.npz") as archive, np.load(OUT/"edited_codes.npz") as saved:
        np.testing.assert_array_equal(station_ids, archive["ids"])
        for name in ("base","recent","ramp"):
            x=raw[pick].copy()
            if name=="recent":
                x[:,-8:]+=2
            if name=="ramp":
                x[:,-8:]+=np.arange(8)*(4./7)-2
            states=[]
            def read(_m,_args,out):
                states.append((out[0] if isinstance(out,tuple) else out)[:,last].detach().numpy().copy())
            handle=site.register_forward_hook(read)
            try:
                with torch.inference_mode():
                    a.predict(x,16,[.5])
            finally:
                handle.remove()
            live=states[0]
            pre=np.maximum((live-c)@w+b,0)
            top=np.argpartition(pre,-32,axis=1)[:,-32:]
            independent=np.zeros_like(pre)
            np.put_along_axis(independent,top,np.take_along_axis(pre,top,axis=1),axis=1)
            truth=saved[name+"_code"][pick]
            state_gap=float(abs(live-archive[name+"_state"][pick]).max())
            code_gap=float(abs(independent[:,atoms]-truth).max())
            np.testing.assert_array_equal(independent[:,atoms]>0,truth>0)
            assert state_gap<1e-4 and code_gap<2e-4,(state_gap,code_gap)
            records[name]=dict(max_live_state_error=state_gap,max_code_error_all_16=code_gap,
                               hard_support_mismatches=0)
    clean=json.loads((OUT/"clean.json").read_text())
    _,_,_,codes,stations=weather()
    xt=raw[:,-8:]
    patch=(xt.mean(1)-raw.mean(1))/(raw.std(1)+1e-5)
    lin=(np.linalg.lstsq(np.column_stack((np.ones(8),np.arange(8))),xt.T,rcond=None)[0][1]*7)/(raw.std(1)+1e-5)
    for kind,score in (("level",-patch),("trend",-lin)):
        atom=clean["domains"]["NOAA"]["selected"][kind]["id"]
        y=codes[:,np.flatnonzero(atoms==atom)[0]]>0
        diff=score[y,None]-score[None,~y]
        pair_auc=float(((diff>0)+.5*(diff==0)).mean())
        assert abs(pair_auc-clean["domains"]["NOAA"]["selected"][kind]["auc"]["value"]) < 1e-12
        records["pairwise_auc_"+kind]=pair_auc
    (OUT/"audit.json").write_text(json.dumps(dict(stations=station_ids[:4].tolist(),origin=0,middle="encoder.block.11",
        last_observed_position=last,observed_support_checks=4*3*16,records=records),indent=2)+"\n")


if __name__=="__main__":
    main()
