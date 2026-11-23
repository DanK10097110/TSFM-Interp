"""On fixed source-selected coordinates, compare patch shape with native forecast shape."""
import argparse
import json
import numpy as np
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import CACHE,OUT,SEEDS,METHODS,VIEWS,histories

B=3000
RNG=22112495

def native_directions(archive):
    # Crosscoder decoders are stored in normalized-view units. Each view has its own physical scale.
    return [archive[k].astype(float)*scale for k,scale in zip(('source_decoder','target_decoder'),archive['scale'])]

def norm_match(v, reference):
    n=np.linalg.norm(v,axis=-1,keepdims=True)
    assert np.min(n)>1e-14
    return v*(np.linalg.norm(reference,axis=-1,keepdims=True)/n)

def agreements(source,target,active,level=False):
    """Return matching-coordinate and mean other-active signed alignments, per event."""
    if level:
        sa=np.sign(source.mean(axis=-1,keepdims=True))
        ta=np.sign(target.mean(axis=-1,keepdims=True))
        a_ok=active & (sa[...,0]!=0);b_ok=active & (ta[...,0]!=0)
    else:
        sc=source-source.mean(axis=-1,keepdims=True)
        tc=target-target.mean(axis=-1,keepdims=True)
        sn=np.linalg.norm(sc,axis=-1,keepdims=True);tn=np.linalg.norm(tc,axis=-1,keepdims=True)
        sa=np.divide(sc,sn,out=np.zeros_like(sc),where=sn>1e-13)
        ta=np.divide(tc,tn,out=np.zeros_like(tc),where=tn>1e-13)
        a_ok=active & (sn[...,0]>1e-13);b_ok=active & (tn[...,0]>1e-13)
    valid=a_ok & b_ok
    same=np.sum(sa*ta,axis=-1)
    other_sum=(ta*b_ok[...,None]).sum(axis=1,keepdims=True)-ta*b_ok[...,None]
    count=b_ok.sum(axis=1,keepdims=True)-b_ok.astype(int)
    other=np.divide(np.sum(sa*other_sum,axis=-1),count,out=np.zeros_like(same),where=count>0)
    if level:
        same=(1+same)/2;other=(1+other)/2
    return np.where(valid,same,np.nan),np.where(valid&(count>0),other,np.nan)

def summary(v,atom):
    row=np.nanmean(v,axis=1)
    assert np.isfinite(row).all()
    w=np.random.default_rng(RNG).multinomial(len(row),np.full(len(row),1/len(row)),B)
    ci=np.quantile(w@row/len(row),[.025,.975])
    use=np.isfinite(v);counts=np.bincount(atom[use],minlength=1024)
    total=np.bincount(atom[use],weights=v[use],minlength=1024);take=counts>=8
    return dict(mean=float(row.mean()),history95=ci.tolist(),n_histories=len(row),events=int(use.sum()),
                eligible_atoms=int(take.sum()),equal_atom_min8=float(np.mean(total[take]/counts[take])))

def run(method,seed):
    _,_,x,ids,dom=histories()
    with np.load(OUT/f'{method}_{seed}_dict.npz') as f:
        ds,dt=native_directions(f);atom=f['atom'];dose=f['dose']
    with np.load(OUT/'ridge.npz') as f:
        map_w=f['map'];sp=f['timesfm_probe'];tp=f['bolt_probe']
        np.testing.assert_array_equal(ids,f['eval_ids'])
    dense=norm_match(ds@map_w,dt)
    active=dose>0
    js,jt=[np.load(CACHE/(name+'_jac.npy'),mmap_mode='r') for name in VIEWS]
    js,jt=js.astype(float),jt.astype(float)
    source_patch=(ds@sp)[atom];paired_patch=(dt@tp)[atom];dense_patch=(dense@tp)[atom]
    source_j=ds[atom]@js.transpose(0,2,1);paired_j=dt[atom]@jt.transpose(0,2,1);dense_j=dense[atom]@jt.transpose(0,2,1)
    metrics={}
    definitions=[('patch32',source_patch,paired_patch,dense_patch),
                 ('bins4',*(w.reshape(256,32,4,8).mean(axis=-1) for w in (source_patch,paired_patch,dense_patch))),
                 ('level',source_patch,paired_patch,dense_patch),('forecast16',source_j,paired_j,dense_j)]
    for label,sa,ta,mapped in definitions:
        same,other=agreements(sa,ta,active,level=label=='level')
        ordinary,other_dense=agreements(sa,mapped,active,level=label=='level')
        metrics[label+'_paired']=same;metrics[label+'_other']=other;metrics[label+'_dense']=ordinary
        metrics[label+'_adv']=same-other;metrics[label+'_dense_adv']=ordinary-other_dense
        metrics[label+'_paired_minus_dense']=same-ordinary
    metrics['patch_minus_forecast_adv']=metrics['patch32_adv']-metrics['forecast16_adv']
    metrics['bins_minus_forecast_adv']=metrics['bins4_adv']-metrics['forecast16_adv']
    result={}
    for domain in np.unique(dom):
        take=dom==domain
        result[str(domain)]={k:summary(v[take],atom[take]) for k,v in metrics.items()}
    np.savez_compressed(OUT/f'{method}_{seed}_directions.npz',ids=ids,domains=dom,atom=atom,dose=dose,**metrics)
    (OUT/f'{method}_{seed}_directions.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);p.add_argument('seed',type=int,choices=SEEDS);a=p.parse_args()
    run(a.method,a.seed)
