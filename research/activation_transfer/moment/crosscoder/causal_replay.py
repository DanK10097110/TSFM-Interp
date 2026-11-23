"""Independent full-graph checks, direct point-score recomputation, and crossed bootstrap spot checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.moment.dictionary_transfer.match import dictionary
from research.activation_transfer.moment.dictionary_transfer.weather.study import live
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates import run as original
from research.activation_transfer.moment.token_paths.geometry import load
from research.activation_transfer.moment.crosscoder.interventions import OUT


def check(out):
    ids,dates,x=original.panel()
    with np.load(original.OUT/'donor.npz') as target:
        true=target['dose'];truth=target['wave'].astype(float);atoms=target['atoms'];sd=target['sd']
    model=load('large')
    decoder=dictionary('large',22112201).W_dec.detach().numpy()[atoms]
    score_report=json.loads((out/'causal_summary.json').read_text())
    scores=score_report['arms']
    rng=np.random.default_rng(22112397)
    ws=rng.multinomial(35,np.full(35,1/35),3000)
    wt=rng.multinomial(11,np.full(11,1/11),3000)
    result={}
    draws={}
    for path in sorted(out.glob('causal_*.npz')):
        arm=path.stem[7:]
        with np.load(path) as f:
            p=f['dose'];y=f['wave'].astype(float)
        rows=[];features=[]
        for which in (true>0,true==0):
            for gate in (p>0,p==0):
                candidates=np.argwhere(which&gate)
                if len(candidates):
                    # Selected indices differ from those archived during production.
                    r,k=candidates[len(candidates)//3]
                    rows.append(r);features.append(k)
        rows=np.asarray(rows);features=np.asarray(features)
        clean=live(model,x[rows])[0]
        replay=live(model,x[rows],-p[rows,features,None]*decoder[features])[0]-clean
        abs_gap=np.abs(replay-y[rows,features]).max()
        assert abs_gap < .00015
        e=[];l=[];fpos=[]
        for k in range(len(atoms)):
            t=truth[:,k];q=y[:,k]
            t=(t-t.mean(1,keepdims=True))/sd[:,None]
            q=(q-q.mean(1,keepdims=True))/sd[:,None]
            e.append(np.sum(t*t,axis=1))
            l.append(np.sum((q-t)**2,axis=1))
            fpos.append(np.sum(q*q,axis=1)*(true[:,k]==0))
        e,l,fpos=[np.array(z).T for z in (e,l,fpos)]
        atol=1e-8
        np.testing.assert_allclose(1-l.sum(0)/e.sum(0),scores[arm]['per_atom'],atol=atol)
        np.testing.assert_allclose(fpos.sum(0)/e.sum(0),scores[arm]['per_atom_false_positive_energy'],atol=atol)
        rep=[]
        for s,d in zip(ws,wt):
            weight=(s[:,None]*d[None]).reshape(-1,1)
            total=(weight*e).sum(0)
            val=1-np.divide((weight*l).sum(0),total,out=np.full(4,np.nan),where=total>1e-16)
            rep.append(np.nanmean(val))
        draws[arm]=np.asarray(rep)
        ci=np.nanquantile(rep,[.025,.975])
        np.testing.assert_allclose(ci,scores[arm]['all']['crossed95'],atol=atol)
        result[arm]=dict(native_checks=len(rows),rows=rows.tolist(),columns=features.tolist(),
                         native_max_celsius=float(abs_gap),primary_crossed95=ci.tolist())
    (out/'causal_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    if {'masked1','masked2','shuffled1','shuffled2'} <= draws.keys():
        matched=.5*(draws['masked1']+draws['masked2']-draws['shuffled1']-draws['shuffled2'])
        ci=np.nanquantile(matched,[.025,.975]).tolist()
        ref=score_report['paired']['masked_paired_mean_minus_masked_shuffled_mean']
        np.testing.assert_allclose(ci,ref['crossed95'],atol=1e-8)
        report={'both_50pct_target_mask_paired_minus_shuffled_crossed95':ci}
        (out/'causal_pairing_replay.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    torch.set_num_threads(2);check(args.out)
