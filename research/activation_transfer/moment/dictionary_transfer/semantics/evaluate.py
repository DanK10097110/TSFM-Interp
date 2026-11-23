"""Locked-coefficient external validation, including nonlinear effects of public weather input edits."""
import argparse
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
import torch

from research.activation_transfer.moment.dictionary_transfer.match import dictionary, encode
from research.activation_transfer.moment.dictionary_transfer.semantics.core import (
    NAMES, OUT, SEED, TRANSFER, materials, measurements, predicted,
)
from research.activation_transfer.moment.forecast_probe.capture import WEATHER
from research.activation_transfer.moment.input_paths.run import modify


def indices(key, clusters):
    groups = [np.flatnonzero(clusters == s) for s in np.unique(clusters)]
    assert len({len(g) for g in groups}) == 1
    seed = int(hashlib.sha256(("moment-scalar|" + key).encode()).hexdigest()[:12], 16)
    draw = np.random.default_rng(seed).integers(len(groups), size=(2000, len(groups)))
    boot = np.stack(groups)[draw].reshape(2000, -1)
    return np.vstack((np.arange(len(clusters)), boot))


def describe(draw):
    draw = np.asarray(draw)
    finite = draw[1:][np.isfinite(draw[1:])]
    return {"value":float(draw[0]) if np.isfinite(draw[0]) else None,
            "ci95":np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None, None],
            "valid_bootstrap":len(finite)}


def ratio(a,b):
    return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)


def auc_many(y, score, ix):
    firing = y[ix] > 0
    n = firing.sum(1)
    ranks = rankdata(score[ix], axis=1)
    return ratio((ranks*firing).sum(1)-n*(n+1)/2,n*(firing.shape[1]-n))


def correlation(a, b):
    x = a - a.mean(1,keepdims=True)
    y = b - b.mean(1,keepdims=True)
    return ratio((x*y).sum(1), np.sqrt((x*x).sum(1)*(y*y).sum(1)))


def quantile(x):
    return np.quantile(x,[.1,.5,.9]).tolist() if len(x) else None


def clean_group(name, raw, y, atoms, clusters, nom, fit_mean):
    z, phys, patch = measurements(raw)
    ix = indices(name, clusters)
    out = dict(n=len(raw), n_clusters=len(np.unique(clusters)),
               stat_pearson=float(np.corrcoef(z.T)[0,1]), selected={})
    for c, kind in enumerate(NAMES):
        fit = nom["winners"][kind]
        j = int(np.flatnonzero(atoms==fit["id"])[0])
        firing = y[:,j] > 0
        other = NAMES[1-c]
        foil = next(f for f in nom["all_candidates"][other] if f["id"]==fit["id"])
        preds = predicted(z[:,c], fit)
        loss = np.square(y[:,j]-preds)[ix].sum(1)
        energy = np.square(y[:,j]-fit_mean[j])[ix].sum(1)
        vals = dict(id=fit["id"], cases=int(firing.sum()), firing_clusters=len(np.unique(clusters[firing])),
                    prevalence=describe(firing[ix].mean(1)), auc=describe(auc_many(y[:,j], z[:,c]*fit["polarity"], ix)),
                    competing_stat_auc=describe(auc_many(y[:,j], z[:,1-c]*foil["polarity"], ix)),
                    amplitude_skill_vs_fit_mean=describe(1-ratio(loss,energy)))
        for tag, mask in (("firing", firing), ("silent", ~firing)):
            vals[tag] = {"stat_raw_q10_median_q90":quantile(phys[mask,c]),
                         "stat_sd_q10_median_q90":quantile(z[mask,c]),
                         "patch_mean_raw_q10_median_q90":quantile(patch[mask])}
        out["selected"][kind] = vals
    cs = [np.flatnonzero(atoms==nom["winners"][s]["id"])[0] for s in NAMES]
    a,b = (y[:,j]>0 for j in cs)
    out["selected_pair_cofiring"] = dict(both=int((a&b).sum()), union=int((a|b).sum()),
                                          amplitude_pearson=float(np.corrcoef(y[:,cs].T)[0,1]) if a.any() and b.any() else None)
    return out


def weather():
    with np.load(WEATHER / "panel.npz") as f:
        station_ids, raw = f["ids"], f["context"]
    with np.load(TRANSFER / "weather/results/22112201.npz") as f:
        atoms, codes, station, origin = f["atoms"], f["z"].astype(float), f["station"], f["origin"]
        assert int(f["pos"])==61
        assert f["ids"][0].startswith(str(station_ids[0]))
    np.testing.assert_array_equal(station,np.repeat(np.arange(46),4))
    np.testing.assert_array_equal(origin,np.tile(np.arange(4),46))
    return station_ids, raw.reshape(-1,480), atoms, codes, station


def clean(nom):
    ids, raw, y, atoms, _ = materials(True)
    fit_ix = np.isin(ids,nom["training_ids"])
    mean = y[fit_ix].mean(0)
    out = {}
    for label, keep in (("fit",fit_ix), ("validation",~fit_ix)):
        out[label] = clean_group(label,raw[keep],y[keep],atoms,np.arange(keep.sum()),nom,mean)
    # Code overlap with the training runner-up checks redundancy without replacing a selected atom.
    near = {}
    for desc in NAMES:
        first = nom["winners"][desc]["column"]
        choices = [c for c in nom["all_candidates"][desc] if c["eligible"] and c["column"]!=first][:2]
        near[desc] = []
        for alt in choices:
            alt_col=alt["column"]
            a,b=y[fit_ix,first],y[fit_ix,alt_col]
            near[desc].append(dict(id=alt["id"], validation_auc=alt["validation"]["auc"],
                train_amplitude_pearson=float(np.corrcoef(a,b)[0,1]),
                train_support_jaccard=float(np.sum((a>0)&(b>0))/np.sum((a>0)|(b>0)))))
    ext, x, ey, evatoms, domain = materials(False)
    assert not set(ext)&set(ids)
    np.testing.assert_array_equal(evatoms,atoms)
    for name in np.unique(domain):
        keep=domain==name
        out[str(name)]=clean_group(str(name),x[keep],ey[keep],atoms,np.arange(keep.sum()),nom,mean)
    _, x, watoms, wy, station=weather()
    np.testing.assert_array_equal(watoms,atoms)
    out["NOAA"]=clean_group("NOAA",x,wy,atoms,station,nom,mean)
    (OUT/"clean.json").write_text(json.dumps(dict(domains=out, alternatives=near),indent=2,allow_nan=False)+"\n")


def edited(nom):
    station_ids, raw, atoms, natural, clusters = weather()
    base, raw0, _ = measurements(raw)
    ix=indices("NOAA",clusters)
    sae=dictionary("large",SEED)
    summary, arrays = {}, {"station_ids":station_ids,"atom_ids":atoms}
    with np.load(TRANSFER/"input_deltas/results/large_native.npz") as f:
        np.testing.assert_array_equal(station_ids, f["ids"])
        basecode=encode(sae,f["base_state"])[:,atoms].astype(float)
        err=float(abs(basecode-natural).max())
        assert err < 1e-3 and np.array_equal(basecode>0,natural>0),err
        arrays["base_code"]=basecode
        for edit in ("recent","ramp"):
            new, physical, _=measurements(modify(raw,edit))
            after=encode(sae,f[edit+"_state"])[:,atoms].astype(float)
            arrays[edit+"_code"]=after
            summary[edit]={}
            for c,name in enumerate(NAMES):
                fit=nom["winners"][name]
                j=fit["column"]
                actual=after[:,j]-basecode[:,j]
                pred=predicted(new[:,c],fit)-predicted(base[:,c],fit)
                arrays[edit+"_"+name+"_predicted_delta"]=pred
                a,b=actual[ix],pred[ix]
                alive=abs(a)>1e-6
                truth_order=rankdata(a,axis=1)
                pred_order=rankdata(b,axis=1)
                norm=np.square(a).sum(1)
                summary[edit][name]=dict(id=fit["id"],raw_stat_delta_range=[float(v) for v in (np.min(physical[:,c]-raw0[:,c]),np.max(physical[:,c]-raw0[:,c]))],
                    normalized_stat_delta_q10_median_q90=quantile(new[:,c]-base[:,c]),
                    actual_nonzero=int(alive[0].sum()), predicted_nonzero=int(np.sum(abs(pred)>1e-6)),
                    gained=int(np.sum((after[:,j]>0)&(basecode[:,j]==0))),lost=int(np.sum((after[:,j]==0)&(basecode[:,j]>0))),
                    zero_change_skill=describe(1-ratio(np.square(a-b).sum(1),norm)),
                    actual_mean=describe(a.mean(1)),predicted_mean=describe(b.mean(1)),
                    actual_abs=describe(abs(a).mean(1)),predicted_abs=describe(abs(b).mean(1)),
                    sign_accuracy_when_true_changes=describe(ratio(((np.sign(a)==np.sign(b))&alive).sum(1),alive.sum(1))),
                    spearman=describe(correlation(truth_order,pred_order)))
    (OUT/"edits.json").write_text(json.dumps(dict(clean_code_max_abs_against_independent_panel=err,
                                               n_stations=len(station_ids),arms=summary),indent=2,allow_nan=False)+"\n")
    np.savez_compressed(OUT/"edited_codes.npz",**arrays)


if __name__ == "__main__":
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["clean","edited"])
    (clean if p.parse_args().stage=="clean" else edited)(json.loads((OUT/"nomination.json").read_text()))
