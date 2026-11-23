"""Bisect real time-series input paths around observed hard SAE atom exchanges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import ci, read_data
from research.activation_transfer.boundary.capture import ALL_MODELS, DEFAULT_SCRATCH, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.boundary.smooth import smooth_codes


def stats(values):
    return {**ci(values),'median':float(np.median(values))}


def contexts(x,cue,t):
    out=x.copy()
    out[:,-64:]+=(.03*t[:,None]).astype(np.float32)*cue
    return out


def tokens(adapter,layer,positions,x):
    parts=[]
    for begin in range(0,len(x),8):
        raw=capture_raw_tokens(adapter,x[begin:begin+8],[layer])[layer].numpy()
        parts.append(raw[:,positions].copy())
    return np.concatenate(parts)


def select_switched(clean_support,alt_support,cap):
    switched=(clean_support[:,None]!=alt_support).any(axis=-1)
    keep=np.flatnonzero(switched.any(axis=1))[:cap]
    family=np.argmax(switched[keep],axis=1)
    return keep,family


@torch.inference_mode()
def study(model,scratch):
    ds=read_data(scratch,'weekly',model)
    name=ALL_MODELS[model][-1]
    sae=load_sae_checkpoint(str(dictionary_path(model)))
    enc=adapter_for(model)
    def codes(h):
        pre=sae.pre_activations(torch.from_numpy(h))
        z=sae.sparsify(pre)
        s,_=smooth_codes(pre,sae.k)
        return {'hard':z.numpy(),'shoulder':s['shoulder'].numpy(),'subtract':s['subtract'].numpy()}
    # Use the frozen cohort's original full-dose classification for screening.
    c0=codes(ds.clean)['hard']>0
    c1=codes(ds.states[:,:,1].copy())['hard']>0
    ids,family=select_switched(c0,c1,32)
    x,cue=ds.raw[ids],ds.cues[ids,family]
    lo=np.zeros(len(x)); hi=np.ones(len(x))
    def evaluate(t):
        xs=contexts(x,cue,t)
        return tokens(enc,name,ds.positions,xs)
    low_tok,high_tok=evaluate(lo),evaluate(hi)
    full_delta=high_tok.mean(1)-low_tok.mean(1)
    base_norm=np.linalg.norm(full_delta,axis=1)
    if np.any(base_norm==0):
        raise ArithmeticError('crossing with zero full-dose dense displacement')
    target_support=codes(low_tok.mean(1))['hard']>0
    available=((codes(high_tok.mean(1))['hard']>0)!=target_support).any(axis=1)
    if not available.all():
        raise RuntimeError('changed selection under replayed initial endpoints')
    decoder=sae.W_dec.detach().numpy()
    out={'model':model,'ids':ds.ids[ids].tolist(),'families':family.tolist(),
         'site':name,'eligible':int(((c0[:,None]!=c1).any(axis=-1)).any(axis=1).sum()),'stages':{}}
    for iteration in range(1,13):
        mid=(lo+hi)/2
        middle=evaluate(mid)
        crossed=((codes(middle.mean(1))['hard']>0)!=target_support).any(axis=1)
        hi[crossed],high_tok[crossed]=mid[crossed],middle[crossed]
        lo[~crossed],low_tok[~crossed]=mid[~crossed],middle[~crossed]
        if iteration not in [4,8,12]:
            continue
        hl,hh=low_tok.mean(1),high_tok.mean(1)
        zl,zh=codes(hl),codes(hh)
        deltas={'dense':hh-hl,**{arm:(zh[arm]-zl[arm])@decoder for arm in zl}}
        lost=(zl['hard']>0)&(zh['hard']<=0)
        new=(zl['hard']<=0)&(zh['hard']>0)
        one=(lost.sum(-1)==1)&(new.sum(-1)==1)
        inp=contexts(x,cue,hi)-contexts(x,cue,lo)
        stage={'input_rms_last64':stats(np.sqrt(np.mean(inp[:,-64:]**2,axis=1))),
               'norm_over_full_native':{k:stats(np.linalg.norm(v,axis=1)/base_norm) for k,v in deltas.items()},
               'n_exactly_one_swap':int(one.sum()),'swapped_atom_count':stats(lost.sum(-1)),
               'full_native_norm':stats(base_norm)}
        if one.any():
            old=lost[one].argmax(-1); fresh=new[one].argmax(-1)
            d0,d1=decoder[old],decoder[fresh]
            c=(d0*d1).sum(-1)/(np.linalg.norm(d0,axis=1)*np.linalg.norm(d1,axis=1))
            stage['single_swap_decoder_cosine']=stats(c)
            stage['single_swap_leaving_coefficient']=stats(zl['hard'][one,old])
        out['stages'][str(iteration)]=stage
    xlow,xhigh=contexts(x,cue,lo),contexts(x,cue,hi)
    repeat=evaluate(lo)
    out['endpoint_repeat_max_dense']=float(np.abs(repeat-low_tok).max())
    selector=last_window_slice(enc,ds.positions)
    y0=change_forecasts(enc,name,xlow,low_tok,None,selector)
    y0_again=change_forecasts(enc,name,xlow,low_tok,np.zeros_like(deltas['dense']),selector)
    out['zero_patch_error']=float(np.abs(y0-y0_again).max())
    yr=change_forecasts(enc,name,xhigh,high_tok,None,selector)-y0
    forecasts={'ordinary_input':yr}
    for label,delta in deltas.items():
        forecasts[label]=change_forecasts(enc,name,xlow,low_tok,delta,selector)-y0_again
    out['forecast_rms']={k:stats(np.sqrt((v.astype(np.float64)**2).mean(axis=-1))) for k,v in forecasts.items()}
    out['brackets']=np.stack([lo,hi],axis=1).tolist()
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=ALL_MODELS,required=True)
    p.add_argument('--scratch',type=Path,default=DEFAULT_SCRATCH)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    args.out.write_text(json.dumps(study(args.model,args.scratch),indent=2,allow_nan=False)+'\n')
