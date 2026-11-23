"""Describe old study item reuse without modifying the frozen forecast-quality panel."""
import json
import numpy as np
from research.activation_transfer.moment.forecast_quality.data import OUT, ROOT

SUITE = ROOT/"research/activation_transfer"


def load(path):
    return json.loads((SUITE/path).read_text())


def canonical(value):
    domain,item=str(value).split("/",1)
    domain={"daily":"m4_daily","weekly":"m4_weekly","traffic":"monash_traffic"}.get(domain,domain)
    if domain in {"monash_traffic","m4_hourly","monash_electricity_hourly"}:
        item=item.removeprefix("T").lstrip("0") or "0"
    return domain+"/"+item


def run():
    reg=load("moment/forecast_quality/results/manifest.json")
    with np.load(OUT/"panel.npz") as arch:
        idx=arch["split"]=="holdout"
        current=list(map(canonical,arch["id"][idx]))
    studies={
        "original_SAE_atom_intervention":list(load("atoms/results/manifest.json")["split"]),
        "matched_window_steering":load("atoms/matched_windows/results/manifest.json")["ids"],
        "paired_dense_transport":list(load("transport/results/manifest.json")["items"]),
        "external_input_control":[r["item"] for r in load("controls/items.json")["rows"]],
        "fresh_shape_transfer":load("shape_transfer/fresh_cohort/results/manifest.json")["item_ids"],
        "rolling_origins":[r["series"] for r in load("shape_transfer/rolling/results/manifest.json")["series"]],
        "cross_dictionary_confirmation":load("shape_transfer/dictionary_prior/refit/results/confirm_manifest.json")["ids"],
        "parallel_small_readout_fit":["m4_daily/"+i for i in load("moment/head_test/results/manifest.json")["ids"]],
        "reserved_support_480_and_256":["m4_daily/"+i for i in reg["reserved_m4_daily_support"]],
    }
    with np.load(SUITE/"replication/results/weekly_pair.npz") as f:
        studies["weekly_dictionary_pairs"]=["weekly/"+str(i) for i in f["ids"]]
    cases={k:sorted(set(map(canonical,v)).intersection(current)) for k,v in studies.items()}
    assert not cases["parallel_small_readout_fit"] and not cases["reserved_support_480_and_256"]
    out={}
    for d in sorted({v.split("/")[0] for v in current}):
        matches={v for entries in cases.values() for v in entries if v.startswith(d+"/")}
        out[d]=dict(registered_holdout=96, used_by_any_checked_earlier_study=len(matches),
                    absent_from_checked_studies=96-len(matches), reused_series=sorted(matches))
    (OUT/"item_overlap.json").write_text(json.dumps(dict(checked_study_item_overlaps=cases,domains=out,
        scope="Selected actual item rosters; different timestamps, interventions or outcomes can share a raw series. Earlier training exclusion lists are not counted as prior observations."),indent=2)+"\n")


if __name__=="__main__":
    run()
