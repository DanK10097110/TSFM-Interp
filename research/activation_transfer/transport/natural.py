"""Transfer observed-window hidden contrasts to fresh daily, weekly and VM histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.transport.analyze import fit_bridge, metrics, sufficient, zscore
from research.activation_transfer.transport.capture import REVISIONS, load_adapter, run_pass, signed_response
from research.activation_transfer.transport.context_controls import bootstrap_summary, donors_and_clusters
from research.activation_transfer.transport.prepare import CONTEXT, DEFAULT_CACHE, HORIZON, REVISION

SOURCES = [("m4_daily", "m4_daily", 160, 2026112061), ("m4_weekly", "m4_weekly", 70, 2026112062),
           ("bitbrains_fast_storage", "bitbrains_fast_storage/H", 128, 2026112063)]
SHIFTS = [1, 4]
STUDY = Path(__file__).resolve().parent


def prepare(cache, out, data_root):
    """Reserve fresh item IDs across experiments, extracting real shifted histories by indexing."""
    originals = dict(np.load(cache / "data.npz"))
    excluded = set(originals["ids"])
    used = {hashlib.sha256(row.tobytes()).digest() for row in originals["x"]}
    controls = STUDY.parent / "controls"
    for path, key in [(controls / "items.json", "rows"), (controls / "bitbrains_items.json", "items")]:
        excluded.update(item["item"] for item in json.loads(path.read_text())[key])
    ids, groups, segments, info = [], [], [], {}
    for name, location, count, seed in SOURCES:
        path = data_root / location / "data-00000-of-00001.arrow"
        with pa.memory_map(str(path)) as source:
            table = pa.ipc.open_stream(source).read_all()
        chosen = []
        size = CONTEXT + 2 * max(SHIFTS)
        for i in np.random.default_rng(seed).permutation(len(table)):
            item_id = name + "/" + str(table["item_id"][int(i)].as_py())
            if item_id in excluded:
                continue
            values = table["target"][int(i)].values
            if name == "bitbrains_fast_storage":
                values = values[0].values
            if len(values) < HORIZON + size:
                continue
            raw = np.asarray(values.slice(len(values) - HORIZON - size, size).to_numpy(zero_copy_only=False),
                             dtype=float)
            center = raw[4:-4]
            if not np.isfinite(raw).all() or center.std() < 1e-8:
                continue
            normalized = ((raw - center.mean()) / center.std()).astype(np.float32)
            digest = hashlib.sha256(normalized[4:-4].tobytes()).digest()
            if digest in used:
                continue
            excluded.add(item_id)
            used.add(digest)
            segments.append(normalized)
            groups.append(name)
            chosen.append(item_id)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"{name} had only {len(chosen)} eligible series")
        with path.open("rb") as f:
            file_hash = hashlib.file_digest(f, "sha256").hexdigest()
        info[name] = dict(sha256=file_hash, seed=seed, items=chosen)
        ids.extend(chosen)
    out.mkdir(parents=True, exist_ok=True)
    (out / "natural_manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    np.savez_compressed(cache / "natural_data.npz", segment=segments, ids=ids, group=groups)
    print({k: len(s["items"]) for k, s in info.items()}, flush=True)


def contexts(segment):
    """Return an actual window centered at t, and windows with endpoints t±1 and t±4."""
    k = max(SHIFTS)
    center = segment[:, k:k + CONTEXT]
    positive = np.stack([segment[:, k+s:k+s+CONTEXT] for s in SHIFTS], axis=1)
    negative = np.stack([segment[:, k-s:k-s+CONTEXT] for s in SHIFTS], axis=1)
    return center, positive, negative


def capture(name, cache):
    data = dict(np.load(cache / "natural_data.npz"))
    x, plus, minus = contexts(data["segment"])
    adapter = load_adapter(name)
    clean, baseline = run_pass(adapter, x)
    high, f_hi = run_pass(adapter, plus.reshape(-1, CONTEXT))
    low, f_lo = run_pass(adapter, minus.reshape(-1, CONTEXT))
    delta = ((high - low) / 2).reshape(len(x), len(SHIFTS), -1)
    input_effect = ((f_hi - f_lo) / 2).reshape(len(x), len(SHIFTS), HORIZON)
    native = signed_response(adapter, x, delta)
    np.savez_compressed(cache / ("natural_" + name + ".npz"), clean=clean, delta=delta,
                        input_effect=input_effect, native=native, forecast=baseline)
    print(name, "natural capture complete", flush=True)


def analyze(target_name, cache, out):
    source_name = next(name for name in REVISIONS if name != target_name)
    fixed = dict(np.load(cache / "data.npz"))
    train = np.flatnonzero(fixed["split"] == "train")
    tr_source = np.load(cache / (source_name + ".npz"))["clean"]
    tr_target = np.load(cache / (target_name + ".npz"))["clean"]
    data = dict(np.load(cache / "natural_data.npz"))
    source = dict(np.load(cache / ("natural_" + source_name + ".npz")))
    target = dict(np.load(cache / ("natural_" + target_name + ".npz")))
    x, hi, lo = contexts(data["segment"])
    aligned, _ = fit_bridge(tr_source[train], tr_target[train])
    raw, _ = fit_bridge(fixed["x"][train], tr_target[train])
    perm = np.random.default_rng(2026112003).permutation(train)
    shuffled, _ = fit_bridge(tr_source[train], tr_target[perm])
    donor_idx = np.empty(len(x), dtype=int)
    pairs_by_group = {}
    for name, _, _, seed in SOURCES:
        idx = np.flatnonzero(data["group"] == name)
        donor, pairs = donors_and_clusters(len(idx), seed + 151)
        donor_idx[idx] = idx[donor]
        pairs_by_group[name] = pairs
    transported = aligned.transport(source["delta"])
    vectors = dict(bridge=transported, shuffle=shuffled.transport(source["delta"]),
                   raw=raw.transport((zscore(hi) - zscore(lo)) / 2), bridge_donor=transported[donor_idx])
    adapter = load_adapter(target_name)
    effects = {k: signed_response(adapter, x, v) for k, v in vectors.items()}
    effects.update(source_native=source["native"], input_edit=target["input_effect"])
    stem = source_name + "_to_" + target_name + "_natural"
    np.savez_compressed(out / (stem + ".npz"), ids=data["ids"], group=data["group"],
                        donor_ids=data["ids"][donor_idx], shifts=SHIFTS, native=target["native"], **effects)
    results = {}
    for name, _, _, _ in SOURCES:
        f = data["group"] == name
        group_effects = {k: v[f] for k, v in effects.items()}
        ref = target["native"][f]
        summaries = bootstrap_summary(group_effects, ref, pairs_by_group[name])
        summaries["native_rms"] = np.sqrt(np.mean(ref**2, axis=(0, 2))).tolist()
        summaries["n"] = int(f.sum())
        e = group_effects["bridge"] - ref
        energy = (ref**2).sum(axis=(1, 2))
        relative_errors = (e**2).sum(axis=(1, 2)) / np.maximum(energy, 1e-20)
        summaries["median_item_recovery"] = float(np.median(1 - relative_errors))
        summaries["strongest_item_share_of_energy"] = float(energy.max() / energy.sum())
        summaries["hidden"] = {k: {key: float(q) for key, q in
                              metrics(sufficient(v[f], target["delta"][f]).sum(axis=0)).items()}
                               for k, v in vectors.items()}
        summaries["versus_source"] = bootstrap_summary(
            {"bridge": group_effects["bridge"], "native_target": ref, "shuffle": group_effects["shuffle"]},
            group_effects["source_native"], pairs_by_group[name])
        summaries["by_shift"] = {str(s): bootstrap_summary(
            {k: v[:, i] for k, v in group_effects.items()}, ref[:, i], pairs_by_group[name])
            for i, s in enumerate(SHIFTS)}
        results[name] = summaries
        print(target_name, name, {k: round(v["recovery"]["value"], 3)
                                  for k, v in summaries["full"]["methods"].items()}, flush=True)
    (out / (stem + ".json")).write_text(json.dumps(results, separators=(",", ":"), allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "capture", "analyze"])
    p.add_argument("model", nargs="?", choices=REVISIONS)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--data-root", type=Path, default=Path.home() / ".cache/huggingface/hub" /
                   "datasets--Salesforce--GiftEval/snapshots" / REVISION)
    p.add_argument("--out", type=Path, default=STUDY / "results")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.cache, args.out, args.data_root)
    elif args.stage == "capture":
        capture(args.model, args.cache)
    else:
        analyze(args.model, args.cache, args.out)


if __name__ == "__main__":
    main()
