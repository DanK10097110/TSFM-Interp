"""Compare fixed training objectives at the history level without equating IDs across dictionaries."""
import json
import numpy as np
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import OUT,METHODS,SEEDS
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import RNG

KEYS=['patch32_paired','patch32_other','patch32_dense','patch32_adv','bins4_adv','level_adv','forecast16_paired','forecast16_other','forecast16_dense','forecast16_adv','patch_minus_forecast_adv']

def describe(row):
    w=np.random.default_rng(RNG).multinomial(len(row),np.full(len(row),1/len(row)),3000)
    return dict(mean=float(row.mean()),history95=np.quantile(w@row/len(row),[.025,.975]).tolist())

def main():
    arms={}
    for m in METHODS:
        for s in SEEDS:
            f=dict(np.load(OUT/f'{m}_{s}_directions.npz'))
            arms[m,s]={k:np.nanmean(f[k],axis=1) for k in KEYS}
    result={}
    for dom in np.unique(f['domains']):
        use=f['domains']==dom;means={}
        for m in METHODS:
            means[m]={k:np.mean([arms[m,s][k][use] for s in SEEDS],axis=0) for k in KEYS}
        out={m:{k:describe(z) for k,z in arm.items()} for m,arm in means.items()}
        out['mask_matched_pairing_control']={k:describe(means['masked'][k]-means['shuffled'][k]) for k in KEYS}
        result[str(dom)]=out
    (OUT/'methods.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
