"""Factor the remaining input structure into early pattern, recent pattern, and relative energy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.shape_transfer.level_pair.factorial import order_average
from research.activation_transfer.shape_transfer.level_pair.gating import coordinate_targets,cutoffs
from research.activation_transfer.shape_transfer.level_pair.inputs import HERE,statistic_rows
from research.activation_transfer.shape_transfer.level_pair.run import NAMED,capture
from tsfm_lens.sae.train import load_sae_checkpoint

OFFSETS=(0,37)
COMPONENTS={1:'early',2:'recent',4:'energy'}


def hybrid_grid(x,daily,traffic):
    x=np.asarray(x,float)
    mean=x.mean(axis=1);sd=x.std(axis=1)
    norm=(x-mean[:,None])/sd[:,None]
    row=statistic_rows(x.shape[1]);inverse=np.linalg.pinv(row)
    residual=norm-(norm@row.T)@inverse.T
    early,recent=residual[:,:-32],residual[:,-32:]
    e1=np.linalg.norm(early,axis=1);e2=np.linalg.norm(recent,axis=1)
    if np.any(np.minimum(e1,e2)<1e-10):
        raise ValueError('A donor pattern has undefined residual orientation')
    fraction=e2**2/(e1**2+e2**2)
    eu=early/e1[:,None];ru=recent/e2[:,None]
    target=np.array([1.25,.5,0.]);projection=target@inverse.T
    energy=x.shape[1]-projection@projection
    assert energy>0
    waves={}
    errors={}
    for bits in range(8):
        prefix=daily if bits&1 else traffic
        tail=daily if bits&2 else traffic
        budget=daily if bits&4 else traffic
        share=fraction[budget]
        a=eu[prefix]*np.sqrt(energy*(1-share))[:,None]
        b=ru[tail]*np.sqrt(energy*share)[:,None]
        y=projection+np.concatenate([a,b],axis=1)
        wave=(mean[prefix,None]+sd[prefix,None]*y).astype('f4')
        after=(wave.astype(float)-mean[prefix,None])/sd[prefix,None]
        goals=np.broadcast_to(target,(len(wave),3))
        actual=after@row.T
        deviations=np.c_[after.mean(1),after.std(1)-1,actual-goals]
        assert np.abs(deviations).max()<2e-6
        waves[bits]=wave;errors[bits]=np.abs(deviations).max(axis=0)
    return waves,fraction,errors


def run(model,out,batch):
    inp=dict(np.load(out/'inputs.npz'))
    gate=dict(np.load(out/(model+'_gating.npz')))
    orig=dict(np.load(out/(model+'_effects.npz')))
    ids=inp['ids'];daily=np.flatnonzero(inp['split']=='daily');traffic=np.flatnonzero(inp['split']=='monash_traffic')
    np.testing.assert_equal(ids,gate['ids'])
    reference,_,_=coordinate_targets(inp['inputs'])
    enc=adapter_for(model);sae=load_sae_checkpoint(str(dictionary_path(model)))
    record=dict(ids=ids,offsets=OFFSETS,daily=daily,domain=inp['split'],target=[1.25,.5,0.])
    for offset in OFFSETS:
        tr=np.roll(traffic,-offset);record['traffic_'+str(offset)]=tr
        waves,fraction,errors=hybrid_grid(inp['inputs'],daily,tr)
        record['fraction_recent_energy']=fraction
        for bits,wave in waves.items():
            key=f'{offset}_{bits}_'
            if bits in (0,7):
                idx=daily if bits==7 else tr
                record[key+'diagonal_input_error']=np.max(np.abs(wave-reference['joint_high'][idx]))
                np.testing.assert_allclose(wave,reference['joint_high'][idx],atol=2e-6)
            hidden=capture(enc,MODELS[model][-1],orig['positions'],wave,batch)
            signals=cutoffs(sae,hidden,NAMED[model]);record[key+'constraint_errors']=errors[bits]
            for field,value in signals.items():
                if bits in (0,7):
                    expected=gate['joint_high_'+field][idx]
                    if field=='rank':assert abs(value-expected).max()<=1
                    else:np.testing.assert_allclose(value,expected,atol=2e-4,rtol=1e-4)
                record[key+field]=value
        print(model,offset,'done',flush=True)
    np.savez_compressed(out/(model+'_splice.npz'),**record)


def summarize(v,draws):
    return {'value':float(v.mean()),'ci95':np.quantile(v[draws].mean(1),[.025,.975]).tolist()}


def score(out):
    result={};rng=np.random.default_rng(2026112193)
    draws=rng.integers(0,128,(2000,128))
    for model in MODELS:
        z=dict(np.load(out/(model+'_splice.npz')));data={};result[model]=data
        for mask,name in [(z['domain']=='daily','daily'),(z['domain']=='monash_traffic','traffic')]:
            e=z['fraction_recent_energy'][mask];data[name+'_tail_energy_fraction']=dict(median=float(np.median(e)),p10_p90=np.quantile(e,[.1,.9]).tolist())
        for off in OFFSETS:
            report={};data[str(off)]=report
            for metric in ('on','pre','rank','margin','cutoff','z'):
                vals={i:z[f'{off}_{i}_'+('z' if metric=='on' else metric)].astype(float) for i in range(8)}
                if metric=='on':vals={k:(v>0).astype(float) for k,v in vals.items()}
                shares=order_average({k:-v for k,v in vals.items()})
                entry=dict(all_daily_minus_traffic=summarize(vals[7]-vals[0],draws),cells={},allocations={})
                for bits,v in vals.items():
                    entry['cells'][str(bits)]=dict(mean=float(np.mean(v)),median=float(np.median(v)))
                    if metric=='on':entry['cells'][str(bits)]['fires']=int(v.sum())
                for k,label in COMPONENTS.items():
                    entry['allocations'][label]=dict(order_average=summarize(shares[k],draws),
                        first=summarize(vals[k]-vals[0],draws),last=summarize(vals[7]-vals[7^k],draws))
                if metric=='on':
                    original=vals[7]>0
                    mixed=vals[5]>0
                    entry['traffic_tail_with_daily_past_and_share']=dict(
                        same_positive=int(np.sum(original&mixed)),
                        only_all_daily=int(np.sum(original&~mixed)),
                        only_mixed=int(np.sum(~original&mixed)),
                        all_daily_minus_mixed=summarize(vals[7]-vals[5],draws))
                    entry['early_by_energy_interaction_at_traffic_tail']=summarize(
                        vals[5]-vals[1]-vals[4]+vals[0],draws)
                report[metric]=entry
            report['max_coordinate_error']=max(float(z[f'{off}_{i}_constraint_errors'].max()) for i in range(8))
            report['max_diagonal_input_error']=max(float(z[f'{off}_{i}_diagonal_input_error']) for i in (0,7))
    (out/'splice.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=(*MODELS,'score'));p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--batch',type=int,default=16)
    opt=p.parse_args();torch.set_num_threads(1)
    if opt.stage=='score':score(opt.out)
    else:run(opt.stage,opt.out,opt.batch)
