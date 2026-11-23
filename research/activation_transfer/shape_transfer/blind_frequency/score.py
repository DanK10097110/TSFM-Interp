"""Crossed feature/series and conditional-series uncertainty on held-out spectral bands."""
import argparse
import json

import numpy as np
from research.activation_transfer.atoms.scores import ci,resampling_counts
from research.activation_transfer.shape_transfer.blind_frequency.data import HERE
from research.activation_transfer.shape_transfer.blind_frequency.geometry import bands

ARMS=("minimum","paired","shuffled","blind_high","independent","donor_regression","donor_static")
DIFF=(("paired","minimum"),("paired","shuffled"),("paired","independent"),
      ("paired","blind_high"),("blind_high","minimum"),("independent","minimum"),
      ("donor_regression","donor_static"),("independent","donor_regression"))


def collapse(score,wt):
    good=np.isfinite(score)
    denom=(wt*good).sum(axis=-1)
    return np.divide((wt*np.where(good,score,0.)).sum(axis=-1),denom,
                     out=np.full(denom.shape,np.nan,dtype=float),where=denom>0)


def scores(effect,reference,mask,series,feature):
    error=mask*np.sum((effect-reference)**2,axis=-1)
    energy=mask*np.sum(reference**2,axis=-1)
    n=series@mask.astype(float)
    den=series@energy
    valid=(n>0)&(den>1e-10*n*reference.shape[-1])
    boot=1-np.divide(series@error,den,out=np.full_like(den,np.nan),where=valid)
    valid_point=energy.sum(0)>1e-10*mask.sum(0)*reference.shape[-1]
    point=1-np.divide(error.sum(0),energy.sum(0),out=np.full(mask.shape[1],np.nan),where=valid_point)
    boot[:,~valid_point]=np.nan
    return np.nanmean(point),collapse(boot,np.ones_like(feature)),collapse(boot,feature),point


def analyze(source,cohort,follow=False):
    with np.load(HERE/"results"/(source+"_"+cohort+".npz")) as f:
        active=f["active"]; labels=f["split"]; atoms=f["atoms"]
        parts={k:bands(f[k].astype(float)) for k in ("source",*ARMS)}
        methods=ARMS
        comparisons=DIFF
        if follow:
            with np.load(HERE/"results"/(source+"_"+cohort+"_static.npz")) as extra:
                for k in ("ids","atoms","active"):
                    np.testing.assert_array_equal(f[k],extra[k])
                parts["static"]=bands(extra["static"].astype(float))
                static_cap=extra["clipped"]
            methods=("static","independent","paired")
            comparisons=(("independent","static"),("static","paired"))
        result={}
        for domain in np.unique(labels):
            i=labels==domain
            eligible=active[i].sum(0)>=8
            m=active[i][:,eligible]
            s=resampling_counts(len(m),draws=8000)
            rng=np.random.default_rng(20261121075)
            k=m.shape[1]
            if not k:
                result[domain]=dict(n=len(m),atoms=[],n_active=0);continue
            features=rng.multinomial(k,np.repeat(1/k,k),size=len(s))
            out=dict(n=len(m),atoms=atoms[eligible].tolist(),n_active=int(m.sum()),firing=m.sum(0).tolist(),
                     cap_counts={key:int((f[key+"_norm"][i][:,eligible][m]>.2).sum()) for key in ARMS[:5]})
            if follow:
                out["cap_counts"]["static"]=int(static_cap[i][:,eligible][m].sum())
            donor=parts["source"]
            tot=(m*np.sum(donor["late"][i][:,eligible]**2,axis=-1)).sum(0)
            fast=(m*np.sum(donor["fast"][i][:,eligible]**2,axis=-1)).sum(0)
            out["fast_energy_share_equal_atom"]=float(np.mean(fast/tot))
            for key in ("early","late","slow","fast"):
                ref=donor[key][i][:,eligible]
                table={}
                raw={}
                for method in methods:
                    raw[method]=scores(parts[method][key][i][:,eligible],ref,m,s,features)
                    point,sboot,cboot,each=raw[method]
                    table[method]=dict(value=float(point),series95=ci(point,sboot)["ci95"],crossed95=ci(point,cboot)["ci95"],
                                       by_atom=[float(v) if np.isfinite(v) else None for v in each])
                for a,b in comparisons:
                    p=raw[a][0]-raw[b][0]
                    sb=raw[a][1]-raw[b][1]
                    cb=raw[a][2]-raw[b][2]
                    row=dict(value=float(p),series95=ci(p,sb)["ci95"],crossed95=ci(p,cb)["ci95"])
                    if cohort=="fresh" and key=="late" and a=="paired" and b in ("shuffled","independent"):
                        row["family_lower"]=float(np.nanquantile(cb,.00625))
                    if follow and cohort=="fresh" and key=="late" and a=="independent" and b=="static":
                        row["family_lower"]=float(np.nanquantile(cb,.0125))
                    table[a+"_minus_"+b]=row
                out[key]=table
            result[domain]=out
    return result


def main(cohort,follow=False):
    report={k:analyze(k,cohort,follow) for k in ("bolt","timesfm")}
    path=HERE/"results"/(cohort+("_followup" if follow else "")+"_scores.json")
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cohort",choices=["old","fresh"])
    p.add_argument("--follow",action="store_true");a=p.parse_args();main(a.cohort,a.follow)
