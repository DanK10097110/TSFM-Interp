"""Freeze four disjoint series sets before evaluating any model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.steering.study import DATA_ROOT, HORIZON, REVISION

STUDY = Path(__file__).resolve().parent
SUITE = STUDY.parents[1]
CACHE = Path("/tmp/tsfm_amplify_v1")
OUT = STUDY / "results"


def excluded_items():
    def read(path):
        return json.loads((SUITE / path).read_text())

    other = read("steering/results/results.json")["series_ids"]
    excluded = {f"m4_{'weekly' if key == 'weekly' else 'daily'}/{i}" for key, items in other.items() for i in items}
    excluded.update(read("transport/results/manifest.json")["items"])
    excluded.update(i for cohort in read("transport/results/natural_manifest.json").values() for i in cohort["items"])
    excluded.update(i for cohort in read("transport/results/axes_manifest.json").values() for i in cohort["ids"])
    excluded.update(row["item"] for row in read("controls/items.json")["rows"])
    excluded.update("m4_" + i for i in read("atoms/results/manifest.json")["split"])
    rep = read("replication/results/manifest.json")["sources"]["daily"]
    excluded.update("m4_" + i for i in rep["items"])
    return excluded


def select(data_root, cache, out):
    excluded = excluded_items()
    manifests, rows, series, split = {}, [], [], []
    seen = set()
    for domain, counts, seed in [("daily", [("clean",320),("cue",40),("daily",80)], 2026112201),
                                  ("weekly", [("weekly",32)], 2026112202)]:
        path = data_root / ("m4_" + domain) / "data-00000-of-00001.arrow"
        with pa.memory_map(str(path)) as source:
            table = pa.ipc.open_stream(source).read_all()
        labels = [s for s,n in counts for _ in range(n)]
        chosen = 0
        for index in np.random.default_rng(seed).permutation(len(table)):
            item = "m4_" + domain + "/" + str(table["item_id"][int(index)].as_py())
            if item in excluded:
                continue
            raw = np.asarray(table["target"][int(index)].as_py(),dtype=float)[-(64+HORIZON):-HORIZON]
            if raw.shape != (64,) or not np.isfinite(raw).all() or raw.std() < 1e-7:
                continue
            x = ((raw - raw.mean()) / raw.std()).astype("f4")
            digest = sha256(x.tobytes()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            excluded.add(item)
            series.append(item)
            rows.append(x)
            split.append(labels[chosen])
            chosen += 1
            if chosen == len(labels):
                break
        if chosen != len(labels):
            raise ValueError(f"Only {chosen} eligible items for {domain}")
        with path.open("rb") as fp:
            from hashlib import file_digest
            checksum = file_digest(fp,"sha256").hexdigest()
        manifests[domain] = dict(seed=seed,sha256=checksum)
    arr = np.stack(rows)
    cache.mkdir(exist_ok=True,parents=True)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(cache/"inputs.npz",x=arr,ids=series,split=split)
    manifest = dict(revision=REVISION,data=manifests,input_sha256=sha256(arr.tobytes()).hexdigest(),
                    items=dict(zip(series,split)))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=DATA_ROOT)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    select(a.data_root,a.cache,a.out)
