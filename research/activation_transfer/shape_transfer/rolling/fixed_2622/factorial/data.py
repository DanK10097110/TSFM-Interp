"""Reconstruct the same 96 traffic sensors at the four prespecified schedules."""
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import N, OUT as ORIGINAL, pair

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = Path("/tmp/tsfm_rolling_2622_factorial_20261121")
OFFSETS = (0, 12, 168, 180)


def prepare():
    original = json.loads((ORIGINAL / "manifest.json").read_text())
    path = Path(original["source"])
    if sha256(path.read_bytes()).hexdigest() != original["source_sha256"]:
        raise ValueError("Traffic input differs from the original data")
    with pa.memory_map(str(path)) as stream:
        table = pa.ipc.open_stream(stream).read_all()
    field = "item_id" if "item_id" in table.column_names else "id"
    row_by_item = {"monash_traffic/" + str(key): i for i, key in enumerate(table[field].to_pylist())}
    contexts = []
    rows = []
    ids = []
    for offset in OFFSETS:
        pairs = []
        for record in original["series"]:
            series = record["series"]
            raw = table["target"][row_by_item[series]].values.to_numpy(zero_copy_only=False)
            result = pair(raw, offset)
            if result is None:
                raise ValueError(f"Invalid fixed {offset}-hour window for {series}")
            windows, _, info = result
            if offset == 0 and info != record["tail"]:
                raise ValueError("Tail provenance changed")
            timestamps = table["timestamp"][row_by_item[series]].values.to_numpy(zero_copy_only=False)
            observed_times = timestamps[info["start"][0]:info["stop"][1]]
            if not np.all(np.diff(observed_times) == np.timedelta64(1, "h")):
                raise ValueError("Calendar offset cannot be inferred from this hourly window")
            stops = [str(timestamps[index - 1]) for index in info["stop"]]
            rows.append(dict(series=series, last_observed_timestamp=stops, **info))
            pairs.append(windows)
        contexts.append(np.stack(pairs).transpose(1, 0, 2))
        ids.extend(f"{offset}/{advance}/{r['series']}" for advance in (0, 32) for r in original["series"])
    inputs = np.stack(contexts).reshape(-1, 480)
    split = np.repeat(OFFSETS, 2 * N).astype(str)
    if len(set(ids)) != len(inputs) or len(inputs) != 4 * 2 * N:
        raise ValueError("Unexpected duplicate or missing origin")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "data.npz", x=inputs, ids=ids, split=split)
    manifest = dict(
        source_sha256=original["source_sha256"], histories=rows, input_sha256=sha256(inputs.tobytes()).hexdigest(),
        id_sha256=sha256("\n".join(ids).encode()).hexdigest(),
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(inputs)} windows frozen", manifest["input_sha256"], flush=True)


if __name__ == "__main__":
    prepare()
