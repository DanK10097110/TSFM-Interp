"""Select disjoint diagnostic and evaluation series and materialize two real rolling test origins."""
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ROOT = HERE.parents[3]
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
REV = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
CHRONOS_REV = "0.0.0/eeecad0b82a8c237e212ce6f8d1abecb513e2cec/chronos_datasets-train.arrow"
GIFT = HF / "hub/datasets--Salesforce--GiftEval/snapshots" / REV
CHRONOS = Path(os.environ.get("TSFM_QUALITY_CHRONOS", HF / "datasets/autogluon___chronos_datasets"))
SPECS = {
    "m4_daily": (GIFT / "m4_daily/data-00000-of-00001.arrow", (7,), 56),
    "m4_weekly": (GIFT / "m4_weekly/data-00000-of-00001.arrow", (52,), 13),
    "m4_hourly": (CHRONOS / "m4_hourly" / CHRONOS_REV, (24, 168), 168),
    "monash_traffic": (CHRONOS / "monash_traffic" / CHRONOS_REV, (24, 168), 168),
    "monash_electricity_hourly": (CHRONOS / "monash_electricity_hourly" / CHRONOS_REV, (24, 168), 168),
    "bitbrains_rnd": (GIFT / "bitbrains_rnd/H/data-00000-of-00001.arrow", (24, 168), 168),
    "bitbrains_fast_storage": (GIFT / "bitbrains_fast_storage/H/data-00000-of-00001.arrow", (24, 168), 168),
}
CONTEXT = 480
HORIZON = 16
DIAGNOSTIC = 24
HOLDOUT = 96
SALT = "moment-quality-20261122-v1"


def hash_array(arr):
    return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def number_id(item):
    digits = item[1:] if item.startswith("T") else item
    return str(int(digits)) if digits.isdigit() else item


def benchmark_items():
    out = {domain: set() for domain in SPECS}
    for line in (ROOT / "benchmark_v3/external_real/corpus.jsonl").read_text().splitlines():
        params = json.loads(line)["provenance"]["generator_params"]
        if params["dataset"] in out:
            out[params["dataset"]].add(number_id(str(params["item_id"])))
    return out


def label_hash(tag):
    return sha256((SALT + "|" + tag).encode()).hexdigest()


def origins(n, domain, item):
    """Data-independent origin placement. Earliest possible window is [0,480)."""
    slack = n - CONTEXT - 2 * HORIZON
    if slack < 0:
        return None
    span = SPECS[domain][2]
    rng = np.random.default_rng(int(label_hash("origin|" + domain + "|" + item)[:16], 16))
    late_offset = int(rng.integers(min(span, slack) + 1))
    gap = HORIZON + int(rng.integers(min(span, slack - late_offset) + 1))
    latest = n - HORIZON - late_offset
    return np.array([latest - gap, latest], dtype=int)


def select():
    exclusions = benchmark_items()
    support = ROOT / "research/activation_transfer/shape_transfer/support_confirmation/results"
    reserved = {str(item) for name in ("manifest.json", "short_manifest.json")
                for item in json.loads((support / name).read_text())["items"]}
    known = set()
    items, domains, splits, cuts, histories, truths = [], [], [], [], [], []
    sources = {}
    for domain, (path, periods, _) in SPECS.items():
        with path.open("rb") as handle:
            digest = file_digest(handle, "sha256").hexdigest()
        with pa.memory_map(str(path)) as handle:
            table = pa.ipc.open_stream(handle).read_all()
        id_field = "item_id" if "item_id" in table.column_names else "id"
        raw_ids = [str(v) for v in table[id_field].to_pylist()]
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError(f"Ambiguous raw item IDs in {domain}")
        order = sorted(range(len(raw_ids)), key=lambda i: label_hash("item|" + domain + "|" + raw_ids[i]))
        records = []
        skipped = {"external_benchmark": 0, "support_reserved": 0, "short": 0, "invalid": 0, "constant": 0, "duplicate": 0}
        for i in order:
            item = raw_ids[i]
            if number_id(item) in exclusions[domain]:
                skipped["external_benchmark"] += 1
                continue
            if domain == "m4_daily" and item in reserved:
                skipped["support_reserved"] += 1
                continue
            values = table["target"][i].values
            if domain.startswith("bitbrains_"):
                values = values[0].values
            raw = values.to_numpy(zero_copy_only=False).astype(float)
            idx = origins(len(raw), domain, item)
            if idx is None:
                skipped["short"] += 1
                continue
            x = np.stack([raw[o-CONTEXT:o] for o in idx])
            y = np.stack([raw[o:o+HORIZON] for o in idx])
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                skipped["invalid"] += 1
                continue
            if np.any(x.std(1) < 1e-8) or np.any(np.abs(np.diff(x, axis=1)).mean(1) < 1e-12):
                skipped["constant"] += 1
                continue
            fingerprints = [hash_array(a) for a in x]
            if len(set(fingerprints)) < 2 or known.intersection(fingerprints):
                skipped["duplicate"] += 1
                continue
            known.update(fingerprints)
            split = "diagnostic" if len(records) < DIAGNOSTIC else "holdout"
            records.append(dict(item=item, split=split, raw_length=len(raw), origin=idx.tolist(),
                                context_sha256=fingerprints, future_sha256=[hash_array(a) for a in y]))
            items.append(domain + "/" + item)
            domains.append(domain)
            splits.append(split)
            cuts.append(idx)
            histories.append(x)
            truths.append(y)
            if len(records) == DIAGNOSTIC + HOLDOUT:
                break
        sources[domain] = dict(path=str(path), sha256=digest, periods=periods, rows=len(table),
                               excluded_items=sorted(exclusions[domain]), rejected_before_quota=skipped,
                               items=records)
        print(domain, len(records), "series", flush=True)
    x, y = np.stack(histories), np.stack(truths)
    manifest = dict(reserved_m4_daily_support=sorted(reserved), salt=SALT, history_length=CONTEXT, horizon=HORIZON, sources=sources,
                    input_sha256=hash_array(x), target_sha256=hash_array(y), ids=items)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "manifest.json"
    if target.exists():
        expected = json.loads(target.read_text())
        regenerated = json.loads(json.dumps(manifest))
        for records in (expected["sources"], regenerated["sources"]):
            for source in records.values():
                source.pop("path")
        if expected != regenerated:
            raise ValueError("Regeneration changed the frozen dataset")
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(OUT / "panel.npz", id=items, domain=domains, split=splits,
                        origin=np.stack(cuts), context=x, future=y)


if __name__ == "__main__":
    select()
