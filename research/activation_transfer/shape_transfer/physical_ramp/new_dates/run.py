"""New-site, new-date test of the fixed TimesFM-to-Bolt operator and a genuinely historical ramp control."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.moment.horizon_prefix.new_dates.run import OUT as WEATHER
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.physical_ramp.study import OUT as OLD,CACHE,PANEL,edits,hashed,match_norm,true_captures,wrong_rows

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
ARMS=('mapped','wrong_step','other_station','historical','donor_native')


def panel():
    meta=json.loads((WEATHER/'manifest.json').read_text())
    with np.load(WEATHER/'panel.npz') as f:
        keep=f['cohort']=='secondary';all_x=f['x'];x=all_x[keep].reshape(-1,480).astype('f4')
        stations=f['ids'][keep]
    assert hashed(all_x)==meta['context_sha256']
    dates=np.asarray(meta['origins']);old_dates={'2026-01-10','2026-04-10','2026-07-10','2026-10-10'}
    with np.load(PANEL) as f:assert not set(f['ids'])&set(stations)
    assert not set(dates)&old_dates
    assert x.shape==(385,480) and len(stations)==35 and len(dates)==11
    st=np.repeat(np.arange(35),11);dt=np.tile(np.arange(11),35)
    ids=np.array([f'{stations[i]}/{dates[j]}' for i,j in zip(st,dt)])
    return x,ids,st,dt,stations,dates


def capture(model,out,batch,old=OLD):
    x,ids,st,dt,sites,dates=panel();out.mkdir(exist_ok=True,parents=True)
    with np.load(old/(model+'_map.npz')) as f:pos=f['source_pos']
    a=adapter_for(model);site=MODELS[model][-1]
    clean,base=true_captures(a,site,x,pos,batch);sel=last_window_slice(a,pos)
    noop=forecast16(a,site,x,clean,np.zeros_like(clean[:,0]),sel,batch)
    np.testing.assert_allclose(noop,base,atol=.00012,rtol=2e-5)
    ramp,step=edits(x)
    outdata=dict(station=st,origin=dt,stations=sites,dates=dates,ids=ids,pos=pos,base=noop,window=clean,
                 zero_max=float(np.max(abs(base-noop))),raw_input_sha=hashed(x))
    changes=[('ramp',ramp),('step',step)] if model=='timesfm' else [('ramp',ramp)]
    for kind,v in changes:
        h,y=true_captures(a,site,v,pos,batch)
        d=(h.astype(float)-clean.astype(float)).mean(1);outdata[kind]=d
        if kind=='ramp':
            outdata['whole']=y.astype(float)-base
            outdata['isolated']=forecast16(a,site,x,clean,d.astype('f4'),sel,batch).astype(float)-noop
    tmp=out/(model+'_native.partial.npz');np.savez_compressed(tmp,**outdata);tmp.replace(out/(model+'_native.npz'))
    print(model,'new 385 native cases',flush=True)


def project(out,batch,cache=CACHE,old=OLD):
    x,ids,st,dt,stations,dates=panel();raw=dict(np.load(out/'timesfm_native.npz'));tar=dict(np.load(out/'bolt_native.npz'))
    np.testing.assert_equal(ids,raw['ids']);np.testing.assert_equal(ids,tar['ids'])
    m=np.load(cache/'timesfm_linear.npy');md=dict(np.load(old/'timesfm_map.npz'));assert hashed(m)==str(md['linear_sha'])
    with np.load(old/'timesfm_native.npz') as f:
        history=f['ramp_mean'].mean(0)@m
        historical_ids=f['ids']
    assert not set(historical_ids)&set(ids);given=wrong_rows(st,dt)
    correct=raw['ramp']@m;dose=np.linalg.norm(correct,axis=1)
    vectors=dict(mapped=correct,wrong_step=match_norm(raw['step']@m,dose),other_station=match_norm(correct[given],dose),
                 historical=match_norm(np.broadcast_to(history,correct.shape),dose))
    a=adapter_for('bolt');site=MODELS['bolt'][-1];sel=last_window_slice(a,tar['pos'])
    store=dict(ids=ids,station=st,origin=dt,stations=stations,dates=dates,wrong=given,budget=dose,
               historical_ids=historical_ids,historical_direction=history,hidden_truth=tar['ramp'],truth=tar['isolated'],
               donor_native=raw['isolated'],whole=tar['whole'],raw_input_sha=hashed(x),map_sha=hashed(m))
    for key,v in vectors.items():
        np.testing.assert_allclose(np.linalg.norm(v,axis=1),dose,atol=1e-8)
        store[key+'_vec']=v
        store[key]=forecast16(a,site,x,tar['window'],v.astype('f4'),sel,batch).astype(float)-tar['base']
    tmp=out/'transfer.partial.npz';np.savez_compressed(tmp,**store);tmp.replace(out/'transfer.npz')
    print('385 x 4 candidate policies complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('timesfm','bolt','project'));p.add_argument('--out',type=Path,default=OUT);p.add_argument('--batch',type=int,default=24)
    p.add_argument('--cache',type=Path,default=CACHE);p.add_argument('--historical',type=Path,default=OLD)
    a=p.parse_args();torch.set_num_threads(1)
    if a.stage=='project':project(a.out,a.batch,a.cache,a.historical)
    else:capture(a.stage,a.out,a.batch,a.historical)
