"""Native finite writes on pre-sampled histories and matched, nonidentical source-active coordinates."""
import argparse
import json
import numpy as np
import torch
from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.forward_only.audit import add_native
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import CACHE,OUT,TEST,VIEWS,SEEDS,METHODS,histories,archived_endpoint
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import native_directions,norm_match,RNG

def sample(domains,atom,dose):
    rng=np.random.default_rng(22112496)
    rows=np.concatenate([np.sort(rng.choice(np.flatnonzero(domains==d),32,replace=False)) for d in np.unique(domains)])
    u=rng.uniform(size=(len(rows),2))
    first=[];other=[]
    for row,(a,b) in zip(rows,u):
        eligible=np.flatnonzero(dose[row]>0)
        i=int(a*len(eligible));j=(i+1+int(b*(len(eligible)-1)))%len(eligible)
        first.append(eligible[i]);other.append(eligible[j])
    i,j=np.array(first),np.array(other)
    assert np.all(atom[rows,i]!=atom[rows,j])
    return rows,i,j

def cos(x,y):
    x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
    length=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
    return np.divide(np.sum(x*y,axis=-1),length,out=np.full(len(x),np.nan),where=length>1e-14)

def report(v):
    v=v[np.isfinite(v)];assert len(v)>0
    w=np.random.default_rng(RNG).multinomial(len(v),np.full(len(v),1/len(v)),3000)
    return dict(value=float(v.mean()),history95=np.quantile(w@v/len(v),[.025,.975]).tolist(),cases=len(v))

def model_materials():
    _,_,x,ids,dom=histories();materials={}
    for name in VIEWS:
        a=adapter_for(name);site=MODELS[name][-1]
        h,pos=archived_endpoint(TEST,name,ids);sel=last_window_slice(a,pos)
        with np.load(TEST/(name+'_early.npz')) as f:base=f['forecast']
        jac=np.load(CACHE/(name+'_jac.npy'),mmap_mode='r')
        materials[name]=(a,site,h,sel,base,jac)
    return x,ids,dom,materials

def run(method,seed,data,sampler=sample,suffix='finite'):
    x,ids,dom,models=data
    with np.load(OUT/f'{method}_{seed}_dict.npz') as f:
        source,target=native_directions(f);atoms=f['atom'];dose=f['dose']
    with np.load(OUT/'ridge.npz') as f:mapped=norm_match(source@f['map'],target)
    rows,i,j=sampler(dom,atoms,dose)
    atom=atoms[rows,i];decoy=atoms[rows,j];amplitude=dose[rows,i,None]
    expected=target[atom]
    moves=dict(source=-amplitude*source[atom],paired=-amplitude*expected,
               dense=-amplitude*mapped[atom],other=-amplitude*norm_match(target[decoy],expected))
    for key in ('dense','other'):
        np.testing.assert_allclose(np.linalg.norm(moves[key],axis=1),np.linalg.norm(moves['paired'],axis=1),rtol=1e-9)
    waves={};der={};check={}
    for k,v in moves.items():
        name='timesfm' if k=='source' else 'bolt'
        a,site,h,sel,base,jac=models[name]
        # forecast16 uses a live native receiver suffix after replacing the unedited token(s) plus the write.
        y=forecast16(a,site,x[rows],h[rows],v.astype('f4'),sel,16)
        waves[k]=y.astype(float)-base[rows];der[k]=np.einsum('nhd,nd->nh',jac[rows],v)
        sub=np.unique([0,len(rows)//2-1,len(rows)//2,len(rows)-1]);xx=x[rows[sub]]
        original=a.predict(xx,16,[.5])['quantiles'][...,0]
        with add_native(a,site,sel,v[sub,None,:].astype('f4')):
            real=a.predict(xx,16,[.5])['quantiles'][...,0].astype(float)-original
        gap=float(np.max(abs(real-waves[k][sub])))
        assert gap<1.5e-4,(k,gap)
        check[k]=gap
    sd=x[rows].astype(float).std(1)[:,None]
    # The precision runner computes its four-per-history bootstrap separately.
    stats={}
    for domain in (np.unique(dom) if suffix=='finite' else []):
        use=dom[rows]==domain
        ref=waves['source'][use]
        centered=(ref-ref.mean(1,keepdims=True))/sd[use]
        sims={};stats[str(domain)]={}
        for name,v in waves.items():
            wave=v[use]
            value=(wave-wave.mean(1,keepdims=True))/sd[use]
            sims[name]=cos(ref,wave)
            linear=cos(wave,der[name][use])
            stats[str(domain)][name]=dict(cos=report(sims[name]),finite_linear_median=float(np.nanmedian(linear)),
                rms_history_sd=float(np.sqrt(np.mean(value**2))),
                skill_against_source=float(1-np.sum((value-centered)**2)/np.sum(centered**2)))
        for control in ('other','dense'):
            stats[str(domain)]['paired_minus_'+control]=report(sims['paired']-sims[control])
    np.savez_compressed(OUT/f'{method}_{seed}_{suffix}.npz',rows=rows,ids=ids[rows],domains=dom[rows],feature=atom,
        other_feature=decoy,dose=amplitude.ravel(),history_sd=sd[:,0],**waves,**{'jac_'+k:z for k,z in der.items()})
    (OUT/f'{method}_{seed}_{suffix}.json').write_text(json.dumps(dict(metrics=stats,native_replay_max=check),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);p.add_argument('seeds',type=int,nargs='+',choices=SEEDS)
    args=p.parse_args();torch.set_num_threads(2);data=model_materials()
    for seed in args.seeds:run(args.method,seed,data)
