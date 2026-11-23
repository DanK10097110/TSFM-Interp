"""Select new daily and hourly item identities without loading any model or SAE coding a history."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from research.activation_transfer.atoms.materials import CACHE as OLD_ATOMS, contexts
from research.activation_transfer.replication.data import CACHE as OLD_REPLICATION, HUB_REVISION

HERE=Path(__file__).resolve().parent
SUITE=HERE.parents[1]
ROOT=SUITE.parents[1]
OUT=HERE/"results"
CACHE=Path("/tmp/tsfm_fresh_full_space_v1")
PUBLIC="autogluon___chronos_datasets"
VERSION="0.0.0/eeecad0b82a8c237e212ce6f8d1abecb513e2cec/chronos_datasets-train.arrow"


def nested(o):
    if isinstance(o,dict):
        yield o
        for v in o.values():
            yield from nested(v)
    elif isinstance(o,list):
        for v in o:
            yield from nested(v)


def exclusions():
    sets={"daily":set(),"m4_hourly":set(),"monash_traffic":set()}
    detail={}
    def add(path,domain,values):
        before=len(sets[domain])
        for v in values:
            key=str(v)
            if domain=="daily" and "/" in key:
                prefix,key=key.split("/",1)
                if prefix not in ("daily","m4_daily"):
                    continue
            sets[domain].add(key)
        detail.setdefault(str(path),{})[domain]=len(sets[domain])-before
    def read(p):
        return json.loads((SUITE/p).read_text())
    for path,fn in [
        ("atoms/results/manifest.json",lambda d:d["split"].keys()),
        ("replication/results/manifest.json",lambda d:d["sources"]["daily"]["items"]),
        ("atoms/matched_windows/results/manifest.json",lambda d:d["ids"]+d["map_train_ids"]),
        ("forecast_requests/results/bolt.json",lambda d:sum((d[k] for k in ["clean_fit_ids","source_fit_ids","calibration_ids","test_ids"]),[])),
        ("steering/results/results.json",lambda d:sum((d["series_ids"][k] for k in ("clean_fit","cue_fit","daily")),[])),
        ("steering/amplify/results/manifest.json",lambda d:d["items"].keys()),
        ("controls/items.json",lambda d:[v["item"] for v in d["rows"]]),
        ("transport/results/manifest.json",lambda d:d["items"].keys()),
        ("transport/results/natural_manifest.json",lambda d:d["m4_daily"]["items"]),
        ("transport/results/axes_manifest.json",lambda d:d["m4_daily"]["ids"]),
        ("boundary/m4.json",lambda d:sum(d["ids"].values(),[])),
        ("boundary/fixed.json",lambda d:d["datasets"]["reserve"]["ids"]),
        ("boundary/reserve_excluded.json",lambda d:d),
        ("shape_transfer/tangents/results/manifest.json",lambda d:list(d["ids"])+d["calibration"]),
    ]:
        add(path,"daily",fn(read(path)))
    for p in [ROOT/"tsfm_lens/runs/real_background_poc492306/metadata.json",
              ROOT/"tsfm_lens/runs/steering_e08_timesfm_real/results.json"]:
        values=list(nested(json.loads(p.read_text())))
        for domain in ("m4_hourly","monash_traffic"):
            selected=[v.get("source_id",v.get("id")) for v in values if v.get("dataset")==domain]
            add(p.relative_to(ROOT),domain,[v for v in selected if v is not None])
    for p in sorted(ROOT.glob("benchmark_*/public_dev/corpus.jsonl")):
        selected={"m4_hourly":set(),"monash_traffic":set()}
        with p.open() as f:
            for line in f:
                if not any(k in line for k in selected):
                    continue
                for record in nested(json.loads(line)):
                    for domain,found in selected.items():
                        if str(record.get("corpus","")).endswith("/"+domain+"/train"):
                            item=str(record.get("item_id","")).split(":")[0]
                            if item.isdigit():
                                found.add(f"T{int(item):06d}")
                            elif re.fullmatch("T[0-9]{6}",item):
                                found.add(item)
        for domain,ids in selected.items():
            add(p.relative_to(ROOT),domain,ids)
    return sets,detail


def select(cache,out):
    sets,detail=exclusions()
    seen=set()
    for old in (OLD_ATOMS,OLD_REPLICATION):
        with np.load(old/"data.npz") as archive:
            seen.update(hashlib.sha256(x.tobytes()).digest() for x in archive["x"])
    fs=Path.home()/".cache/huggingface"
    daily=fs/"hub/datasets--Salesforce--GiftEval/snapshots"/HUB_REVISION/"m4_daily/data-00000-of-00001.arrow"
    items,x=contexts(daily,128,2026112601,seen,sets["daily"])
    hourly_root=fs/"datasets"/PUBLIC
    for domain in ("m4_hourly","monash_traffic"):
        path=hourly_root/domain/VERSION
        if not path.exists():
            continue
        temp=set(seen)
        try:
            new,z=contexts(path,128,2026112602,temp,sets[domain])
            break
        except ValueError as exc:
            if not re.fullmatch(r"Only [0-9]+ eligible in .+", str(exc)):
                raise
            continue
    else:
        raise ValueError("insufficient eligible hourly histories")
    ids=np.array(["daily/"+v for v in items]+[domain+"/"+v for v in new])
    x=np.vstack([x,z])
    split=np.array(["daily"]*128+[domain]*128)
    for p,i in [(daily,"daily"),(path,domain)]:
        with p.open("rb") as f:detail.setdefault("selected_corpora",{})[i]={"file":str(p),"sha256":hashlib.file_digest(f,"sha256").hexdigest()}
    cache.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache/"data.npz",x=x,ids=ids,split=split)
    meta=dict(item_ids=ids.tolist(),domains=["daily",domain],excluded_counts={k:len(v) for k,v in sets.items()},
              cumulative_new_exclusions_by_source=detail,seed={"daily":2026112601,domain:2026112602},
              split_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(meta["domains"],meta["excluded_counts"],meta["split_sha256"],ids[:3],ids[128:131],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    select(a.cache,a.out)
