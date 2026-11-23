"""Four real, dose-matched event sets on every history, with clustered uncertainty."""
import argparse
import json
import numpy as np
import torch
from research.activation_transfer.moment.crosscoder.patch_probe import crossed
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import OUT,METHODS,SEEDS
from research.activation_transfer.moment.crosscoder.cross_family_directions.finite import model_materials,run,cos
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import RNG

def four(domains,atom,dose):
    rng=np.random.default_rng(22112497)
    rows=np.repeat(np.arange(len(domains)),4);selected=[];other=[]
    for values in dose:
        active=np.flatnonzero(values>0);assert len(active)>=4
        take=rng.choice(active,4,replace=False);selected.extend(take)
        for i in take:other.append(rng.choice(active[active!=i]))
    return rows,np.asarray(selected),np.asarray(other)

def scores(method,seed):
    f=dict(np.load(OUT/f'{method}_{seed}_precision.npz'))
    assert len(f['rows'])==1024
    info=json.loads((OUT/f'{method}_{seed}_precision.json').read_text())
    summary={}
    for domain in np.unique(f['domains']):
        use=f['domains']==domain;n=128;cols=4
        rows=f['rows'][use].reshape(n,cols)
        assert np.all(rows==rows[:,:1]) and len(np.unique(rows))==128
        sd=f['history_sd'][use,None]
        source=f['source'][use]
        centered=(source-source.mean(1,keepdims=True))/sd
        energy=np.square(centered).sum(1).reshape(n,cols).sum(1)
        rng=np.random.default_rng(RNG)
        weights=rng.multinomial(n,np.full(n,1/n),3000)
        atom=f['feature'][use].reshape(n,cols)
        sims={}; native={}
        for arm in ('source','paired','other','dense'):
            v=f[arm][use];z=(v-v.mean(1,keepdims=True))/sd
            sims[arm]=cos(source,v).reshape(n,cols)
            assert np.isfinite(sims[arm]).all()
            err=np.square(z-centered).sum(1).reshape(n,cols).sum(1)
            rel=1-err.sum()/energy.sum()
            native[arm]=dict(rms_history_sd=float(np.sqrt(np.mean(z*z))),skill=float(rel),
                skill_history95=np.quantile(1-(weights@err)/(weights@energy),[.025,.975]).tolist(),
                finite_linear_median=float(np.median(cos(v,f['jac_'+arm][use]))),
                cos=float(sims[arm].mean()),cos_history95=np.quantile(weights@sims[arm].mean(1)/n,[.025,.975]).tolist())
        contrasts={k:sims['paired']-sims[k] for k in ('other','dense')}
        joint=crossed(contrasts,atom,None,False)
        for name,z in contrasts.items():
            joint[name]['history95']=np.quantile(weights@z.mean(1)/n,[.025,.975]).tolist()
        summary[str(domain)]=dict(n_history=n,events=n*cols,arms=native,paired_minus=joint)
    info['metrics']=summary
    (OUT/f'{method}_{seed}_precision.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);p.add_argument('--score-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);data=None if args.score_only else model_materials()
    for seed in SEEDS:
        if not args.score_only:run(args.method,seed,data,sampler=four,suffix='precision')
        scores(args.method,seed)
