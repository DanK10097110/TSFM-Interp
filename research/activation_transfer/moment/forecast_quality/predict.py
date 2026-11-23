"""Run the official frozen checkpoint adapters using only the prospective observed contexts."""
import argparse
import importlib.metadata
import json
import os
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import numpy as np
import torch

from research.activation_transfer.boundary.capture import MODELS as EXISTING, HUB
from research.activation_transfer.moment.forecast_quality.data import CONTEXT, OUT, hash_array
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter

NAMES = ("moment_small", "moment_base", "moment_large", "bolt", "timesfm")
MOMENT_ROOT = Path(os.getenv("TSFM_MOMENT_NATIVE", "/tmp/tsfm_moment_native"))


def checkpoint(model):
    if model.startswith("moment_"):
        variant = model.split("_")[-1]
        return "moment", f"MOMENT-1-{variant}", MOMENT_ROOT / variant
    name, kind, repo, revision, _ = EXISTING[model]
    return kind, name, HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def adapter(model):
    kind, name, path = checkpoint(model)
    obj = build_adapter(ModelConfig(name=name, adapter=kind, checkpoint=str(path), batch_size=16),
                        DataConfig(context_len=CONTEXT, horizon=16), torch.device("cpu"), torch.float32)
    obj.ensure_loaded()
    return obj


def calls(enc, x, horizon, batch):
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            obs = np.ascontiguousarray(x[start:start+batch], dtype=np.float32)
            result = enc.predict(obs, horizon, [.5])
            pred = result["quantiles"][..., 0]
            if pred.shape != (len(obs), horizon) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid direct {horizon}-lead forecast")
            values.append(pred)
    return np.concatenate(values).astype(np.float64)


def available_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def checkpoint_metadata(model):
    _, _, where = checkpoint(model)
    hashes = {}
    for file in sorted(where.iterdir()):
        if file.is_file() and (file.suffix in (".safetensors", ".bin") or file.name == "config.json"):
            with file.open("rb") as reader:
                hashes[file.name] = file_digest(reader, "sha256").hexdigest()
    versions = {name: available_version(name) for name in ("momentfm", "torch", "transformers", "chronos-forecasting", "timesfm")}
    return str(where), hashes, versions


def save_npz(path, **arrays):
    """Publish a whole pass atomically, so interruption never makes a completed-looking partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as temp:
        name = temp.name
        np.savez_compressed(temp, **arrays)
    os.chmod(name, 0o640)
    os.replace(name, path)


def cached_call(enc, values, horizon, batch, cache, signature):
    if cache.exists():
        with np.load(cache) as f:
            if str(f["signature"]) == signature and f["pred"].shape == (len(values), horizon):
                return f["pred"], float(f["seconds"]), True
    start = monotonic()
    pred = calls(enc, values, horizon, batch)
    elapsed = monotonic() - start
    save_npz(cache, pred=pred, seconds=elapsed, signature=signature)
    return pred, elapsed, False


def run(model, batch):
    enc = adapter(model)
    with np.load(OUT / "panel.npz") as archive:
        ids = archive["id"]
        contexts = archive["context"]
    n, origins, length = contexts.shape
    x = contexts.reshape(-1, length)
    mean = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    z = (x - mean) / sd
    fingerprint = hash_array(contexts)
    results = dict(ids=ids, context_sha256=fingerprint)
    where, hashes, versions = checkpoint_metadata(model)
    signature = json.dumps([model, fingerprint, hashes, versions, batch, torch.get_num_threads(), "raw-f32-v1"], sort_keys=True)
    times, resumed = {}, []
    for horizon in (8, 16):
        for kind, inputs in (("raw", x), ("standardized", z)):
            key = f"{kind}_{horizon}"
            cache = OUT / ".passes" / f"{model}_{key}.npz"
            value, times[key], hit = cached_call(enc, inputs, horizon, batch, cache, signature)
            if hit:
                resumed.append(key)
            if kind == "standardized":
                value = value * sd + mean
            results[key] = value.reshape(n, origins, horizon)
            print(model, key, f"{times[key]:.1f}s", "from cache" if hit else "", flush=True)
    info = dict(model=model, path=where, hashes=hashes, versions=versions, seconds=times, resumed_passes=resumed,
                batch=batch, torch_threads=torch.get_num_threads(), device="cpu-float32")
    save_npz(OUT / (model + ".npz"), **results)
    (OUT / (model + ".json")).write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=NAMES)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.model, args.batch)
