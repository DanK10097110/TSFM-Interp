"""Collect external-intervention responses and split identities using production forecasts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REV = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
CONTEXT = 128
HORIZON = 16
DOSE = 0.35
EDITS = ("ramp", "sine8", "recent_step", "pulse", "translate", "fixed_moments")
MODELS = {
    "bolt": ("chronos_bolt", "amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8"),
    "timesfm": ("timesfm", "google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"),
}


def source_exclusions(transport_manifest):
    exclusions = set(json.loads(transport_manifest.read_text())["items"])
    with (REPO / "benchmark_v3/external_real/corpus.jsonl").open() as f:
        for line in f:
            src = json.loads(line)["provenance"]["generator_params"]
            exclusions.add(f"{src['dataset']}/{src['item_id']}")
    published = REPO / "examples/steering/validation/m4_daily.json"
    for group in json.loads(published.read_text())["groups"]:
        exclusions.update("m4_daily/" + case["item"] for case in group["cases"])
    return exclusions


def read_histories(table, count, seed, group, exclusions, used_hashes, variate=None):
    """Choose one eligible, nonduplicated, independently standardized history per ID."""
    rows = []
    for idx in np.random.default_rng(seed).permutation(len(table)):
        item = group + "/" + str(table["item_id"][int(idx)].as_py())
        if item in exclusions:
            continue
        sequence = table["target"][int(idx)].values
        if variate is not None:
            sequence = sequence[variate].values
        if len(sequence) < CONTEXT + HORIZON:
            continue
        history = sequence.slice(len(sequence)-CONTEXT-HORIZON, CONTEXT).to_numpy(zero_copy_only=False)
        if not np.isfinite(history).all():
            continue
        history = history.astype(np.float64)
        if history.std() < 1e-8:
            continue
        history = ((history-history.mean())/history.std()).astype(np.float32)
        digest = hashlib.sha256(history.tobytes()).hexdigest()
        if digest in used_hashes:
            continue
        exclusions.add(item)
        used_hashes.add(digest)
        rows.append((item, digest, history))
        if len(rows) == count:
            return rows
    raise ValueError(f"{group} has only {len(rows)} eligible, unused series under the protocol")


def manifest_and_histories(args):
    groups = {"m4_daily": (320, 975510), "m4_weekly": (96, 975511), "LOOP_SEATTLE/H": (160, 975512)}
    exclusions = source_exclusions(args.transport_manifest)
    hashes = set()
    output = {"data_revision": REV, "sources": {}, "rows": []}
    xs = []
    for key, (count, seed) in groups.items():
        group = key.split("/")[0]
        path = args.data_root / key / "data-00000-of-00001.arrow"
        with pa.memory_map(str(path)) as f:
            table = pa.ipc.open_stream(f).read_all()
            rows = read_histories(table, count, seed, group, exclusions, hashes)
        with path.open("rb") as f:
            output["sources"][key] = {"seed": seed, "sha256": hashlib.file_digest(f, "sha256").hexdigest()}
        for i, (item, digest, x) in enumerate(rows):
            split = "train" if group == "m4_daily" and i < 128 else group
            output["rows"].append({"item": item, "sha256": digest, "split": split})
            xs.append(x)
    return output, np.stack(xs)


def edit_histories(x):
    """Return signed raw stimuli, per-series RMS dose, and ten clean-input predictors."""
    n = len(x)
    i = np.arange(32)
    edits = np.zeros((5, CONTEXT), dtype=np.float64)
    edits[:4, -32:] = np.stack([np.linspace(-1, 1, 32), np.sin(2*np.pi*i/8),
                                i >= 24, np.maximum(0, 1-np.abs(i-16)/4)])
    edits[4] = 1.0
    base = np.asarray(x, dtype=np.float64)[:, None, None, :]
    signs = np.array([-1., 1.])[None, None, :, None]
    pairs = np.broadcast_to(base, (n, len(EDITS), 2, CONTEXT)).copy()
    pairs[:, :5] = base + signs * DOSE * edits[None, :, None, :]
    interior = x[:, -31:-1].astype(np.float64)
    mean = interior.mean(axis=1, keepdims=True)
    u = interior - mean
    length = np.linalg.norm(u, axis=1, keepdims=True)
    unit = np.divide(u, length, out=np.zeros_like(u), where=length>1e-12)
    template = (np.maximum(0, 1-np.abs(i-8)/4) - np.maximum(0, 1-np.abs(i-22)/4))[1:-1]
    template = template - template.mean()
    direction = template - (unit @ template)[:, None] * unit
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    v = np.divide(direction, norm, out=np.zeros_like(direction), where=norm>1e-12)
    angle = np.arctan(DOSE)
    plus = mean + np.cos(angle)*u + np.sin(angle)*length*v
    minus = mean + np.cos(angle)*u - np.sin(angle)*length*v
    for j, edited in enumerate([minus, plus]):
        pairs[:, 5, j, -31:-1] = np.where((norm>1e-12) & (length>1e-12), edited, interior)
    rms = np.sqrt(np.mean(((pairs[:, :, 1]-pairs[:, :, 0]) / 2)**2, axis=-1))
    pairs = pairs.astype(np.float32)
    return pairs, rms, input_summaries(x)


def input_summaries(x):
    x = x.astype(np.float64)
    def slope(width):
        time = np.arange(width) - (width-1)/2
        return x[:, -width:] @ time / (time @ time)
    def correlation(lag):
        past, future = x[:, :-lag].copy(), x[:, lag:].copy()
        past -= past.mean(axis=1,keepdims=True)
        future -= future.mean(axis=1,keepdims=True)
        denom = np.linalg.norm(past, axis=1)*np.linalg.norm(future, axis=1)
        return np.divide(np.sum(past*future, axis=1), denom, out=np.zeros(len(x)), where=denom>1e-12)
    return np.column_stack([x[:, -1], x[:, -1]-x[:, -2], slope(8), slope(32), x[:, -32:].std(axis=1),
                            correlation(1), correlation(8), correlation(24), np.abs(np.diff(x)).mean(axis=1),
                            np.abs(np.diff(x, n=2)).mean(axis=1)])


def forecast_responses(model_name, pairs, args):
    """Evaluate central differences with no activation hooks."""
    import torch
    from tsfm_lens.config import ModelConfig, DataConfig
    from tsfm_lens.models import build_adapter
    adapter, repo, revision = MODELS[model_name]
    path = args.hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"Missing pinned checkpoint: {path}")
    m = build_adapter(ModelConfig(name=model_name, adapter=adapter, checkpoint=str(path)),
                      DataConfig(context_len=CONTEXT, horizon=HORIZON), torch.device("cpu"), torch.float32)
    m.ensure_loaded()
    subsequences = pairs.reshape(-1, CONTEXT)
    ys = []
    for start in range(0, len(subsequences), args.batch_size):
        y = m.predict(subsequences[start:start+args.batch_size], HORIZON, [.5])["quantiles"][:, :, 0]
        if not np.isfinite(y).all():
            raise ArithmeticError(f"Nonfinite output for {model_name} at {start}")
        ys.append(y)
    y = np.concatenate(ys).reshape(*pairs.shape[:-1], HORIZON)
    # Re-evaluate a fixed sample, measuring a different batch partition's error.
    for index in [0, len(pairs)//2, len(pairs)-1]:
        check = m.predict(pairs[index, 0, :], HORIZON, [.5])["quantiles"][:, :, 0]
        np.testing.assert_allclose(check, y[index, 0], rtol=3e-4, atol=4e-5)
    m.unload()
    return ((y[:, :, 1]-y[:, :, 0])/2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=Path.home()/".cache/huggingface/hub")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--transport-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--allow-write-new-manifest", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        args.data_root = args.hub/"datasets--Salesforce--GiftEval/snapshots"/REV
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, xs = manifest_and_histories(args)
    manifest_path = args.out/"items.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen split differs from input files; refusing to mix results")
    elif args.allow_write_new_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    else:
        parser.error("No split manifest; use --allow-write-new-manifest before a new run")
    print(f"Frozen {len(xs)} items", flush=True)
    pairs, doses, summaries = edit_histories(xs)
    output = dict(rms=doses, summaries=summaries, edits=np.array(EDITS),
                  item=np.array([r["item"] for r in manifest["rows"]]),
                  split=np.array([r["split"] for r in manifest["rows"]]))
    for model_name in MODELS:
        print(f"Collecting {model_name}", flush=True)
        output[model_name] = forecast_responses(model_name, pairs, args)
    final = args.out/"responses.npz"
    np.savez_compressed(final, **output)
    print(f"Wrote {final}", flush=True)


if __name__ == "__main__":
    main()
