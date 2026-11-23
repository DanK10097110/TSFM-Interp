"""Freeze one direction per historical daily holdout before measuring its response."""
import hashlib
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.replication.data import CACHE as ORIGINAL
from research.activation_transfer.shape_transfer.sparse_writes.run import HERE as SPARSE

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SCRATCH = Path("/tmp/tsfm_sparse_stability_20261120")
SEED = 2026112701
DOSES = (.001, .02)
N = 96


def prepare():
    with np.load(SPARSE / "results/bolt.npz") as old:
        ids = old["ids"][old["split"] == "daily"][:N]
    with np.load(ORIGINAL / "data.npz") as raw:
        lookup = {key: i for i, key in enumerate(raw["ids"])}
        x = raw["x"][[lookup[key] for key in ids]]
    noise = np.random.default_rng(SEED).normal(size=x.shape)
    noise -= noise.mean(axis=1, keepdims=True)
    noise *= x.std(1, keepdims=True) / np.sqrt(np.square(noise).mean(1, keepdims=True))
    perturbed = np.concatenate([(x + a * noise).astype("f4") for a in DOSES])
    names = np.array([f"{a}/{key}" for a in DOSES for key in ids])
    SCRATCH.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(SCRATCH / "data.npz", ids=names, x=perturbed, split=np.repeat(DOSES, N))
    sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
    record = dict(ids=ids.tolist(), original_sha256=sha(x), perturbed_sha256=sha(perturbed),
                  noise_sha256=sha(noise), seed=SEED, relative_rms=list(DOSES))
    (OUT / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
