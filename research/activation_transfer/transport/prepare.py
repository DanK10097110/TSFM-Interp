"""Choose series before running models and persist a reproducible, chronological split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa


CONTEXT = 128
HORIZON = 16
CONCEPTS = ("ramp", "sine8", "recent_step", "pulse")
DOSE = 0.35
REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
DEFAULT_CACHE = Path("/tmp/tsfm_activation_transport_v1")


def patterns():
    """Four signed, local input contrasts in clean-context standard deviations."""
    i = np.arange(32)
    result = np.zeros((len(CONCEPTS), CONTEXT), np.float32)
    result[:, -32:] = np.stack([np.linspace(-1, 1, 32), np.sin(2 * np.pi * i / 8),
                                i >= 24, np.maximum(0, 1 - np.abs(i - 16) / 4)])
    return result


def eligible(table, count, seed, used_contexts):
    """Return distinct item IDs and histories, rejecting exact standardized duplicates."""
    identities = set()
    rows = []
    for index in np.random.default_rng(seed).permutation(len(table)):
        item = str(table["item_id"][int(index)].as_py())
        if item in identities:
            continue
        target = table["target"][int(index)].values
        if len(target) < CONTEXT + HORIZON:
            continue
        window = target.slice(len(target) - CONTEXT - HORIZON, CONTEXT).to_numpy(zero_copy_only=False)
        if not np.isfinite(window).all():
            continue
        window = window.astype(np.float64)
        if window.std() < 1e-8:
            continue
        standardized = ((window - window.mean()) / window.std()).astype(np.float32)
        fingerprint = hashlib.sha256(standardized.tobytes()).hexdigest()
        if fingerprint in used_contexts:
            continue
        rows.append((item, standardized))
        identities.add(item)
        used_contexts.add(fingerprint)
        if len(rows) == count:
            return rows
    raise ValueError(f"requested {count} eligible items, found {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=Path.home() / ".cache/huggingface/hub" /
                        "datasets--Salesforce--GiftEval/snapshots" / REVISION)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    x, ids, split = [], [], []
    details = {}
    used_contexts = set()
    for group, count, seed in [("m4_daily", 512, 2026112001), ("m4_weekly", 128, 2026112002)]:
        path = args.data_root / group / "data-00000-of-00001.arrow"
        with pa.memory_map(str(path)) as source:
            rows = eligible(pa.ipc.open_stream(source).read_all(), count, seed, used_contexts)
        ids.extend(group + "/" + i for i, _ in rows)
        x.extend(row for _, row in rows)
        split.extend(["train"] * 384 + ["daily"] * 128 if group == "m4_daily" else ["weekly"] * 128)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        details[group] = dict(sha256=digest, seed=seed)
    np.savez_compressed(args.cache / "data.npz", x=np.stack(x), ids=ids, split=split,
                        patterns=patterns(), concepts=CONCEPTS, dose=np.float32(DOSE))
    manifest = dict(revision=REVISION, data=details, items=dict(zip(ids, split)))
    (args.cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
