"""Select disjoint series, preserve absolute raw offsets and build a five-origin panel."""
from hashlib import sha256, file_digest
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.atoms.materials import CACHE as ATOMS
from research.activation_transfer.replication.data import CACHE as REP, HUB_REVISION
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH, PUBLIC, VERSION, exclusions
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = Path("/tmp/tsfm_rolling_20261121")
LAGS = (0, 1, 4, 16, 32)
N = 96


def windows(values):
    """Input index stop is the first unavailable observation at that origin."""
    initial_start = len(values) - 8 - LAGS[-1] - 480
    starts = np.array(LAGS) + initial_start
    raw = np.stack([values[i:i+480] for i in starts]).astype(float)
    if not np.isfinite(raw).all() or np.min(raw.std(1)) < 1e-8:
        return None
    mu, sd = float(raw[0].mean()), float(raw[0].std())
    fixed = ((raw - mu) / sd).astype("f4")
    normalized = ((raw - raw.mean(1,keepdims=True)) / raw.std(1,keepdims=True)).astype("f4")
    return starts, mu, sd, fixed, normalized


def prepare():
    sets, details = exclusions()
    seen = set()
    for archived in (ATOMS, REP, FRESH, BLIND):
        with np.load(archived / "data.npz") as f:
            for name, x in zip(f["ids"], f["x"]):
                domain, _, item = str(name).partition("/")
                if domain in sets:
                    sets[domain].add(item)
                seen.add(sha256(x.tobytes()).digest())
    home = Path.home()/".cache/huggingface"
    paths = (home/"hub/datasets--Salesforce--GiftEval/snapshots"/HUB_REVISION/"m4_daily/data-00000-of-00001.arrow",
             home/"datasets"/PUBLIC/"monash_traffic"/VERSION)
    records, inputs, sources = [], [], {}
    for domain,seed,path in zip(("daily","monash_traffic"),(20261121331,20261121332),paths):
        with pa.memory_map(str(path)) as stream:
            table = pa.ipc.open_stream(stream).read_all()
        field = "item_id" if "item_id" in table.column_names else "id"
        count = 0
        for i in np.random.default_rng(seed).permutation(len(table)):
            item = str(table[field][int(i)].as_py())
            if item in sets[domain]:
                continue
            vals = table["target"][int(i)].values
            if len(vals)<520:
                continue
            res = windows(vals.to_numpy(zero_copy_only=False))
            if res is None:
                continue
            starts, mu, sd, fixed, normalized = res
            hashes = [sha256(x.tobytes()).digest() for x in normalized]
            if set(hashes) & seen:
                continue
            key = domain + "/" + item
            inputs.append(fixed)
            records.append(dict(series=key, domain=domain, raw_length=len(vals), start=starts.tolist(),
                                stop=(starts+480).tolist(), anchor_mean=mu, anchor_sd=sd))
            seen.update(hashes)
            sets[domain].add(item)
            count += 1
            if count == N:
                break
        if count != N:
            raise ValueError(f"Only {count} eligible {domain} series")
        with path.open("rb") as f:
            digest=file_digest(f,"sha256").hexdigest()
        sources[domain]=dict(seed=seed, file=str(path), sha256=digest, prior_excluded=len(sets[domain])-N)
    ids = np.array([f"{lag}/{r['series']}" for lag in LAGS for r in records])
    x = np.stack(inputs).transpose(1,0,2).reshape(len(ids),480).copy()
    split = np.array([r["domain"] for _ in LAGS for r in records])
    manifest = dict(lags=LAGS,series=records, source=sources, original_exclusion_sources=details,
                    other_archives=["original atoms","replication atoms","fresh_cohort","blind_frequency"],
                    id_hash=sha256("\n".join(ids).encode()).hexdigest(), input_hash=sha256(x.tobytes()).hexdigest())
    CACHE.mkdir(exist_ok=True,parents=True)
    OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(CACHE/"data.npz",x=x,ids=ids,split=split)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(x.shape,manifest["id_hash"],manifest["input_hash"],flush=True)


if __name__ == "__main__":
    prepare()
