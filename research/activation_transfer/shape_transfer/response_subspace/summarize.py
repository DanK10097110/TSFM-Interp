"""Equal-atom functional equivalence with paired series and two-way atom/series intervals."""
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import ci,metrics,resampling_counts,terms
from research.activation_transfer.shape_transfer.response_subspace.run import HERE


def intervals(mask,y,truth,series_counts,feature_counts):
    t=terms(mask,y,truth)
    p=metrics(t.sum(axis=0))["skill"]
    sums=(series_counts@t.reshape(len(t),-1)).reshape(len(series_counts),*t.shape[1:])
    segment=metrics(sums)["skill"]
    valid=np.isfinite(p)
    if not valid.any():
        raise ValueError("No measurable active features for endpoint")
    draw=segment[:,valid]
    weights=feature_counts[:,valid]*np.isfinite(draw)
    two=np.divide(np.nansum(weights*draw,axis=1),weights.sum(axis=1),
                  out=np.full(len(weights),np.nan),where=weights.sum(axis=1)>0)
    return np.nanmean(p),np.nanmean(draw,axis=1),two,valid


def run(out):
    result={}
    for source in ("bolt","timesfm"):
        by_seed={}
        for seed in range(3):
            with np.load(out/f"{source}_seed{seed}.npz") as f:
                data=dict(f)
            methods=[k for k in data if k+"_norm_fraction" in data and k!="native"]
            domains={}
            for domain in np.unique(data["split"]):
                use=data["split"]==domain
                active=data["active"][use]
                cols=active.sum(axis=0)>=8
                n_features=int(cols.sum())
                if n_features==0:
                    domains[str(domain)]=dict(frequent=0)
                    continue
                mask=active[:,cols]
                draws=resampling_counts(int(use.sum()),draws=2500)
                fdraws=np.random.default_rng(82013).multinomial(n_features,np.full(n_features,1/n_features),size=len(draws))
                d=dict(frequent=n_features,active_cases=int(mask.sum()),atoms=data["atoms"][cols].tolist(),
                       median_norm={k:float(np.median(data[k+"_norm_fraction"][cols])) for k in methods},
                       median_hidden_cosine={k:float(np.median(data[k+"_hidden_cosine"][cols])) for k in methods},endpoints={})
                for endpoint in ("early_median","early_centered_median","late_centered_median","early_width","late_width"):
                    section=slice(8,16) if endpoint.startswith("late") else slice(0,8)
                    def transform(arr):
                        v=arr[use][:,cols,section].astype(float)
                        y=(v[:,:,:,2]-v[:,:,:,0])/2 if endpoint.endswith("width") else v[:,:,:,1]
                        return y-y.mean(axis=-1,keepdims=True) if "centered" in endpoint else y
                    truth=transform(data["native"])
                    estimates={k:intervals(mask,transform(data[k]),truth,draws,fdraws) for k in methods}
                    e={k:dict(value=float(p),ci95=np.nanquantile(bs,[.025,.975]).tolist(),
                              ci95_features_and_histories=np.nanquantile(two,[.025,.975]).tolist(),measurable=int(valid.sum()))
                       for k,(p,bs,two,valid) in estimates.items()}
                    for left,right in [("foil32","foil8"),("project32","pca32"),("project32","foil32")]:
                        p,b,c,_=estimates[left];q,s,t,_=estimates[right]
                        e[left+"_minus_"+right]=dict(**ci(p-q,b-s),
                                                     ci95_features_and_histories=np.nanquantile(c-t,[.025,.975]).tolist())
                    d["endpoints"][endpoint]=e
                domains[str(domain)]=d
            by_seed[str(seed)]=domains
        result[source]=by_seed
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    run(HERE/"results")
