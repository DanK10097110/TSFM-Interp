"""Aggregate same- and cross-encoder physical swaps with all methods paired on firing events."""
import argparse
import json
import numpy as np
from research.activation_transfer.moment.dictionary_transfer.match import OUT,SIZES
from research.activation_transfer.moment.dictionary_transfer.intervene import ARMS,SCRATCH
from research.activation_transfer.moment.token_paths.geometry import CACHE
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.head_test.data import OUT as HEADS
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,entry,difference
from research.activation_transfer.atoms.scores import metrics,terms


def alternate(size,state,sd):
    with np.load(HEADS/f"{size}_0_heads.npz") as f:w=f["interior_hidden_weight"]
    return (state.astype(float)@w*sd[:,None,None]).reshape(len(sd),16)


def run(source,target,seed):
    stem=f"{source}_{seed}_to_{target}"
    folder=SCRATCH/stem
    with np.load(folder/"events.npz") as f:d=dict(f)
    with np.load(OUT/(stem+"_matches.npz")) as f:matches=dict(f)
    with np.load(CACHE/source/f"ablation_{seed}.npz") as f:
        for k in ("ids","row","col","atoms"):np.testing.assert_array_equal(d[k],f[k])
        np.testing.assert_array_equal(f["codes"]>0,d["z"]>0)
        np.testing.assert_allclose(f["codes"],d["z"],atol=2e-5,rtol=2e-6)
        truth={"native":f["physical"],"interior":alternate(source,f["dstate"],f["sd"])}
    records={};vecs={};floors={}
    for arm in ARMS:
        with np.load(folder/(arm+".npz")) as f:
            vecs[arm]=f["norm"]
            native,state=f["native"],f["dstate"]
            zero=vecs[arm]==0
            floors[arm]=float(np.max(np.abs(native[zero]))) if zero.any() else 0.
            # A structural zero (closed gate) can differ from an archived ordinary
            # prediction by roundoff across batch shapes. Verify it before using exact zero.
            assert floors[arm]<2e-5
            native[zero]=0
            state[zero]=0
            records[arm]={"native":native,"interior":alternate(target,state,d["sd"][d["row"]])}
    if source==target:
        # Identical encoder, original hidden vector, independent batch geometry.
        np.testing.assert_allclose(records["dense"]["native"],truth["native"],atol=2e-5,rtol=.005)
    result={}
    compact={**d,**{k:matches[k] for k in ("corr_id","geom_id","random_id","corr_best","geom_best","cos_at_corr","corr_at_geom")}}
    for head in truth:
        compact["truth_"+head]=truth[head]
        for method in ARMS:compact[head+"_"+method]=records[method][head]
    for kind,v in vecs.items():compact[kind+"_length"]=v
    np.savez_compressed(OUT/(stem+"_forecast.npz"),**compact)
    for dom in np.unique(d["split"]):
        p=Panel(d["z"]>0,d["row"],d["col"],d["split"]==dom)
        r,c=d["row"][p.events],d["col"][p.events]
        cell=dict(features=int(p.keep.sum()),events=int(p.events.sum()),zero_write_roundoff=floors,atoms=d["atoms"][p.keep].tolist(),
                  naturally_matched_dose_present=float(d["target_active"][r,c].mean()),
                  distinct_corr_matches=int(len(set(matches["corr_id"][p.keep]))),
                  median_training_match_correlation=float(np.median(matches["corr_best"][p.keep])),
                  median_decoder_cosine_at_corr_match=float(np.median(matches["cos_at_corr"][p.keep])),
                  median_geometry_match_cosine=float(np.median(matches["geom_best"][p.keep])),
                  response_clip_fraction=float(np.mean(d["response_scale"][p.events]<1)),
                  median_norm={k:float(np.median(v[p.events])) for k,v in vecs.items()})
        for head,ref in truth.items():
            for side,sl in (("early",slice(8)),("late",slice(8,16))):
                for kind in ("shape","full"):
                    y=component(ref[:,sl],kind); estimates={}; per={}; amplitude={}
                    for method,v in records.items():
                        series=component(v[head][:,sl],kind)
                        estimates[method]=p.estimate(series,y)
                        values=metrics(terms(p.mask,p.cube(series),p.cube(y)).sum(0))["skill"]
                        per[method]=[float(v) if np.isfinite(v) else None for v in values]
                        amplitude[method]=float(np.sqrt(np.mean(series[p.events]**2)))
                    key="_".join((head,side,kind))
                    cell[key]={k:entry(v) for k,v in estimates.items()}
                    cell[key]["rms_source"]=float(np.sqrt(np.mean(y[p.events]**2)))
                    cell[key]["rms_target"]=amplitude
                    for a,b in (("corr_forced","corr_own"),("geom_forced","corr_forced"),("dense","geom_forced"),("response","dense")):
                        cell[key][a+"_minus_"+b]=difference(estimates[a],estimates[b])
                    cell[key+"_per_atom"]=per
        result[str(dom)]=cell
    (OUT/(stem+"_forecast.json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=SIZES);p.add_argument("target",choices=SIZES)
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    a=p.parse_args();run(a.source,a.target,a.seed)
