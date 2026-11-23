"""Read physically measured forecast displacements only for the two independently nominated IDs."""
import json

import numpy as np

from research.activation_transfer.moment.dictionary_transfer.semantics.core import OUT, SEED, TRANSFER
from research.activation_transfer.moment.dictionary_transfer.semantics.evaluate import describe, indices, ratio


def waves(source, estimates, which, event_rows, clusters, min_clusters=8, noise=None):
    n=len(clusters)
    pick=np.flatnonzero(which)
    used_rows=event_rows[pick]
    if len(np.unique(used_rows)) != len(used_rows):
        raise ValueError("Expected one event per atom per history")
    nsites=len(np.unique(clusters[used_rows]))
    result=dict(events=len(used_rows),clusters=nsites,eligible=nsites>=min_clusters)
    if nsites<min_clusters:
        return result
    ix=indices("physical"+str(len(np.unique(clusters))),clusters)
    result["centered_leads"]={}
    for label,chunk in (("1–8",slice(0,8)),("9–16",slice(8,16))):
        truth=source[pick,chunk].astype(float)
        truth-=truth.mean(1,keepdims=True)
        energy=np.zeros(n)
        energy[used_rows]=(truth**2).sum(1)
        floor=None
        if noise is not None:
            nulls=noise[pick,chunk].astype(float)
            nulls-=nulls.mean(1,keepdims=True)
            floor=float(np.sqrt((nulls**2).mean()))
        rms=float(np.sqrt((truth**2).mean()))
        cell=dict(donor_rms_history_sd=rms,matched_noop_rms_history_sd=floor,
                  above_10x_noise=bool(rms>10*max(floor,1e-12)) if floor is not None else None,
                  physical_skill={})
        for arm, effects in estimates.items():
            v=effects[pick,chunk].astype(float)
            v-=v.mean(1,keepdims=True)
            error=np.zeros(n)
            error[used_rows]=((v-truth)**2).sum(1)
            cell["physical_skill"][arm]=describe(1-ratio(error[ix].sum(1),energy[ix].sum(1)))
        result["centered_leads"][label]=cell
    return result


def run():
    nom=json.loads((OUT/"nomination.json").read_text())["winners"]
    result={}
    for target in ("large","small"):
        stem=f"large_{SEED}_to_{target}"
        with np.load(TRANSFER/"results"/(stem+"_forecast.npz")) as f, np.load(TRANSFER/"results"/(stem+"_group_waves.npz")) as g:
            for a in ("ids","row","col"):
                np.testing.assert_array_equal(f[a],g[a])
            atoms,dom,rows,cols = f["atoms"],f["split"],f["row"],f["col"]
            for desc, fit in nom.items():
                j=fit["column"]
                assert int(atoms[j])==fit["id"]
                out={}
                for domain in np.unique(dom):
                    use=dom==domain
                    row_map=np.cumsum(use)-1
                    which=(cols==j)&use[rows]
                    assert np.sum(which)==np.sum(f["z"][use,j]>0)
                    arms={k:f["native_"+k] for k in ("corr_own","corr_forced","random")}
                    arms["group16"]=g["sixteen_native"]
                    out[str(domain)]=waves(f["truth_native"],arms,which,row_map[rows],np.arange(use.sum()))
                result[f"{desc}_{fit['id']}_large_to_{target}"]=dict(matched_target_id=int(f["corr_id"][j]),
                    source_seed=SEED,target_seed=22112202,panels=out)
    with np.load(TRANSFER/"weather/results/22112201.npz") as f:
        atoms, rows,cols,station = f["atoms"],f["rows"],f["cols"],f["station"]
        sd=f["input_sd"][rows,None]
        for desc,fit in nom.items():
            j=fit["column"]
            assert atoms[j]==fit["id"]
            which=cols==j
            assert which.sum()==np.count_nonzero(f["z"][:,j])
            truth=(f["donor"]-f["noop"])/sd
            arms={k:(f[k]-f["noop"])/sd for k in ("corr_own","corr_forced","random","group16")}
            cell=waves(truth,arms,which,rows,station,noise=f["noop"]/sd)
            result[f"{desc}_{fit['id']}_large_to_large"]["panels"]["NOAA"]=cell
    (OUT/"physical.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
