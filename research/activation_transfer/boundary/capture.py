"""Collect paired final-window states for the prospectively specified boundary study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.extraction.alignment import pooling_matrix
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.models import build_adapter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HUB = Path(os.getenv("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
DATA_REV = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
L = 480
H = 8
FAMILIES = ["pulse", "sine"]
DOSES = [.01, .03, .1]
MODELS = {
    "bolt": ("Chronos-Bolt", "chronos_bolt", "amazon/chronos-bolt-small",
             "772f3d25d38aec6d914c8949dab4462e2d46f5d8", "encoder.block.3"),
    "timesfm": ("TimesFM", "timesfm", "google/timesfm-2.5-200m-pytorch",
                "1d952420fba87f3c6dee4f240de0f1a0fbc790e3", "stacked_xf.10"),
}
ALL_MODELS = {
    **MODELS,
    "chronos2": ("Chronos-2", "chronos2", "amazon/chronos-2",
                 "29ec3766d36d6f73f0696f85560a422f50e8498c", "encoder.block.5"),
}
DATASETS = {"m4": "m4_daily", "machines": "bitbrains_rnd/H",
            "fast": "bitbrains_fast_storage/H", "weekly": "m4_weekly", "reserve": "m4_daily"}
DEFAULT_SCRATCH = Path("/tmp/tsfm_boundary_20261120")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hashkey(label):
    return hashlib.sha256(label.encode()).digest()


def dictionary_path(model):
    name, _, _, _, layer = ALL_MODELS[model]
    return ROOT / "tsfm_lens/runs/panel7_v3_ablcheck/sae" / name / (layer.replace(".", "_") + ".pt")


def excluded_ids(dataset):
    ids = set()
    with (ROOT / "benchmark_v3/external_real/corpus.jsonl").open() as f:
        for line in f:
            params = json.loads(line)["provenance"]["generator_params"]
            if params["dataset"] == DATASETS[dataset].split("/")[0]:
                ids.add(str(params["item_id"]))
    return ids


def cohort(dataset, count=None):
    """Choose items by ID only, then take one finite, nonconstant history per item."""
    name = DATASETS[dataset]
    data = HUB / "datasets--Salesforce--GiftEval/snapshots" / DATA_REV / name / "data-00000-of-00001.arrow"
    with pa.memory_map(str(data), "r") as stream:
        table = ipc.open_stream(stream).read_all()
    excluded = excluded_ids(dataset)
    if dataset == "reserve":
        excluded.update(str(i) for i in json.loads((HERE/"reserve_excluded.json").read_text()))
    ids = table["item_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("ambiguous item identity")
    salt = f"boundary-{dataset}-20261120"
    order = sorted(range(len(ids)), key=lambda i: hashkey(f"{salt}|{ids[i]}"))
    kept, values, ends = [], [], []
    duplicates = set()
    count = count or (256 if dataset == "m4" else 80)
    for i in order:
        item = ids[i]
        if item in excluded or (dataset == "m4" and int(item) < 2000):
            continue
        if dataset == "reserve" and not 1000 <= int(item) < 2000:
            continue
        x = np.asarray(table["target"][i].as_py(), dtype=np.float64)
        if dataset in {"machines", "fast"}:
            x = x[0]  # CPU-usage channel only, one item per machine.
        if x.size < L:
            continue
        end = x.size
        if dataset in {"machines", "fast"}:
            # Latest fully observed 480-point window, with no imputation.
            cumulative_missing = np.r_[0, np.cumsum(~np.isfinite(x))]
            eligible = np.flatnonzero((cumulative_missing[L:] - cumulative_missing[:-L]) == 0)
            if eligible.size == 0:
                continue
            end = int(eligible[-1] + L)
        raw = x[end-L:end]
        if not np.isfinite(raw).all() or raw.std() <= 1e-8:
            continue
        normalized = np.asarray((raw - raw.mean()) / raw.std(), dtype=np.float32)
        fingerprint = hashlib.sha256(normalized.tobytes()).digest()
        if fingerprint in duplicates:
            continue
        duplicates.add(fingerprint)
        kept.append(item)
        ends.append(end)
        values.append(normalized)
        if len(kept) >= count:
            break
    if len(kept) < count:
        raise ValueError(f"only {len(kept)} eligible {dataset} items, requested {count}")
    return np.asarray(kept), np.stack(values), np.asarray(ends), data


def shapes(dataset, ids):
    """Per-item randomized, Hann-tapered shape with unit RMS over 64 input steps."""
    t = np.arange(64, dtype=float)
    out = np.empty((len(ids), 2, 64), dtype=np.float32)
    for i, item in enumerate(ids):
        rng = np.random.default_rng(int.from_bytes(hashkey(f"cue|{dataset}|{item}")[:8], "big"))
        center, width = rng.uniform(22, 46), rng.uniform(4, 12)
        pulse = rng.choice([-1., 1.]) * np.exp(-.5 * ((t - center) / width)**2)
        sine = np.sin(2 * np.pi * t / rng.uniform(8, 32) + rng.uniform(0, 2*np.pi))
        for j, curve in enumerate((pulse, sine)):
            curve = curve * np.hanning(t.size)
            out[i, j] = curve / np.sqrt(np.mean(curve**2))
    return out


def adapter_for(model):
    name, adapter, repo, revision, _ = ALL_MODELS[model]
    path = HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(path)
    spec = ModelConfig(name=name, adapter=adapter, checkpoint=str(path), batch_size=16)
    # Chronos-2 context states depend on the full set of forecast placeholder tokens.
    horizon = 64 if model == "chronos2" else H
    model = build_adapter(spec, DataConfig(context_len=L, horizon=horizon),
                          torch.device("cpu"), torch.float32)
    model.ensure_loaded()
    return model


def capture(model, dataset, scratch, threads, batch_size=8):
    torch.set_num_threads(threads)
    scratch.mkdir(exist_ok=True, parents=True)
    ids, xs, ends, data = cohort(dataset)
    cues = shapes(dataset, ids)
    enc = adapter_for(model)
    layer = ALL_MODELS[model][-1]
    window_weights = pooling_matrix(enc.token_time_spans(), L, 32)[-1].numpy()
    positions = np.flatnonzero(window_weights)
    w = window_weights[positions]
    assert np.allclose(w, 1/len(w)), w
    chunks = []
    for start in range(0, len(xs), batch_size):
        b = xs[start:start + batch_size]
        def final_tokens(inputs):
            return capture_raw_tokens(enc, inputs, [layer])[layer].numpy()[:, positions].copy()
        clean = final_tokens(b)
        repeat = final_tokens(b)
        variants = []
        for family in range(len(FAMILIES)):
            group = []
            for amp in DOSES:
                inputs = b.copy()
                inputs[:, -64:] += amp * cues[start:start+len(b), family]
                group.append(final_tokens(inputs).mean(axis=1))
            variants.append(np.stack(group, axis=1))
        chunks.append((clean, repeat.mean(axis=1), np.stack(variants, axis=1)))
        print(f"{dataset}/{model} {start + len(b)}/{len(xs)}", flush=True)
    clean, repeat, variants = (np.concatenate(part) for part in zip(*chunks))
    path = scratch / f"{dataset}_{model}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=ids, raw=xs, ends=ends, cues=cues, tokens=clean, repeat=repeat,
             states=variants, positions=positions, source_sha256=digest(data),
             sae_sha256=digest(dictionary_path(model)))
    tmp.replace(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--dataset", choices=DATASETS, default="m4")
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(capture(args.model, args.dataset, args.scratch, args.threads, args.batch_size))
