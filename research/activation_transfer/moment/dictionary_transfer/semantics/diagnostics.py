"""Station-conditioned discrimination and individual-versus-mean change for the frozen two features."""
import json

import numpy as np

from research.activation_transfer.moment.dictionary_transfer.semantics.core import NAMES, OUT, auc, measurements
from research.activation_transfer.moment.dictionary_transfer.semantics.evaluate import describe, indices, ratio, weather


def run():
    station_ids,raw,atoms,natural,cl=weather()
    stats=measurements(raw)[0]
    nom=json.loads((OUT/"nomination.json").read_text())["winners"]
    ix=indices("NOAA",cl)
    station_draw=ix[:,::4]//4
    out={"natural":{},"changes":{}}
    with np.load(OUT/"edited_codes.npz") as f:
        for c,kind in enumerate(NAMES):
            fit=nom[kind];j=fit["column"];assert atoms[j]==fit["id"]
            z=stats[:,c]*fit["polarity"];y=natural[:,j]>0
            den,num=[],[]
            for s in range(len(station_ids)):
                chosen=cl==s
                yy,zz=y[chosen],z[chosen]
                pair=zz[yy,None]-zz[None,~yy]
                den.append(pair.size);num.append((pair>0).sum()+.5*(pair==0).sum())
            den,num=np.asarray(den),np.asarray(num)
            adm=den>0
            out["natural"][kind]=dict(informative_stations=int(adm.sum()),within_station_pairs=int(den.sum()),
                within_station_auc=describe(ratio(num[station_draw].sum(1),den[station_draw].sum(1))),
                by_origin=[None if not np.isfinite(a:=auc(y[k::4],z[k::4])) else float(a) for k in range(4)])
            for edit in ("recent","ramp"):
                actual=f[edit+"_code"][:,j]-f["base_code"][:,j]
                pred=f[edit+"_"+kind+"_predicted_delta"]
                a,b=actual[ix],pred[ix]
                active=np.abs(a)>1e-6
                pc=b-b.mean(1,keepdims=True)
                ac=a-a.mean(1,keepdims=True)
                out["changes"][edit+"_"+kind]=dict(
                    centered_skill=describe(1-ratio(((pc-ac)**2).sum(1),(ac**2).sum(1))),
                    true_unchanged=int(np.sum(~active[0])),
                    predicted_change_when_unchanged=int(np.sum((~active[0])&(np.abs(b[0])>1e-6))),
                    closed_energy_frac=describe(ratio(((~active)*b**2).sum(1),(a*a).sum(1))))
    (OUT/"diagnostics.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
