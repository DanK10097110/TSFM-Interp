"""Sensitivity to repeated active coordinate IDs, with each trained dictionary held fixed."""
import argparse
import json
import numpy as np
from research.activation_transfer.moment.crosscoder.patch_probe import crossed
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import OUT,SEEDS,METHODS

KEYS=('patch32_adv','bins4_adv','level_adv','forecast16_adv','patch_minus_forecast_adv','bins_minus_forecast_adv')

def eligible_only(values,atoms):
    # Condition on coordinates that appear in at least eight distinct histories.
    counts=np.bincount(atoms.ravel(),minlength=1024)
    use=counts[atoms]>=8
    rr,cc=np.nonzero(use)
    unique,assignment=np.unique(atoms[rr,cc],return_inverse=True)
    n=len(atoms);m=len(unique)
    cell=rr*m+assignment
    obs=np.bincount(cell,minlength=n*m).reshape(n,m)
    rng=np.random.default_rng(22112475)
    row=rng.multinomial(n,np.full(n,1/n),2500)
    col=rng.multinomial(m,np.full(m,1/m),2500)
    denom=np.einsum('bi,bi->b',row@obs,col);assert denom.min()>0
    result={}
    for name,z in values.items():
        mat=np.bincount(cell,weights=z[rr,cc],minlength=n*m).reshape(n,m)
        samples=np.einsum('bi,bi->b',row@mat,col)/denom
        result[name]=dict(mean=float(z[rr,cc].mean()),history_atom_95=np.quantile(samples,[.025,.975]).tolist(),
                          equal_atom=float(np.mean(mat.sum(0)/obs.sum(0))),eligible_atoms=m,events=len(rr))
    return result

def run(method,seed):
    f=dict(np.load(OUT/f'{method}_{seed}_directions.npz'));output={}
    for domain in np.unique(f['domains']):
        rows=f['domains']==domain;atom=f['atom'][rows]
        vals={key:f[key][rows] for key in KEYS}
        assert all(np.isfinite(x).all() for x in vals.values())
        all_ids=crossed(vals,atom,None,False)
        restricted=eligible_only(vals,atom)
        output[str(domain)]=dict(all_observed_coordinates=all_ids,minimum_eight_histories=restricted)
    (OUT/f'{method}_{seed}_clustered.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);args=p.parse_args()
    for seed in SEEDS:run(args.method,seed)
