"""Lock the new daily and virtual-machine cohorts and capture their clean study-window tokens."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

HERE = Path(__file__).resolve().parent
from research.activation_transfer.atoms.materials import CACHE as ORIGINAL, HUB, capture

CACHE = Path("/tmp/tsfm_atom_replication_20261120")
HUB_REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"


def prepare(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(ORIGINAL / "data.npz") as primary:
        keep = np.isin(primary["split"], ["train", "cal"])
        xs, ids, splits = list(primary["x"][keep]), list(primary["ids"][keep]), list(primary["split"][keep])
        excluded = set(primary["ids"])
        seen = {hashlib.sha256(x.tobytes()).digest() for x in primary["x"]}
    detail = {}
    for name, location, n, seed in [("daily", "m4_daily", 192, 2026112068),
                                     ("machines", "bitbrains_fast_storage/H", 128, 2026112069)]:
        path = HUB / "datasets--Salesforce--GiftEval/snapshots" / HUB_REVISION / location / "data-00000-of-00001.arrow"
        with pa.memory_map(str(path)) as stream:
            table = pa.ipc.open_stream(stream).read_all()
        selected = []
        for i in np.random.default_rng(seed).permutation(len(table)):
            item = f"{name}/{table['item_id'][int(i)].as_py()}"
            if item in excluded:
                continue
            row = table["target"][int(i)].values
            if name == "machines":
                row = row[0].values
            if len(row) < 488:
                continue
            x = row.slice(len(row)-488, 480).to_numpy(zero_copy_only=False).astype(float)
            if not np.isfinite(x).all() or x.std() < 1e-8:
                continue
            x = ((x-x.mean()) / x.std()).astype("f4")
            digest = hashlib.sha256(x.tobytes()).digest()
            if digest in seen:
                continue
            selected.append(item)
            xs.append(x)
            ids.append(item)
            splits.append(name)
            seen.add(digest)
            excluded.add(item)
            if len(selected) == n:
                break
        if len(selected) != n:
            raise ValueError(f"Only {len(selected)} eligible {name} histories")
        with path.open("rb") as reader:
            file_hash = hashlib.file_digest(reader, "sha256").hexdigest()
        detail[name] = dict(source=location, sha256=file_hash, items=selected, seed=seed)
    xs = np.stack(xs)
    provenance = dict(sources=detail, inherited="atoms/results/manifest.json",
                      input_sha256=hashlib.sha256(xs.tobytes()).hexdigest())
    np.savez_compressed(cache / "data.npz", x=xs, ids=ids, split=splits)
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["data", "bolt", "timesfm"])
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=HERE / "results")
    args = p.parse_args()
    torch.set_num_threads(3)
    if args.stage == "data":
        prepare(args.cache, args.out)
    else:
        capture(args.stage, args.cache)
