"""Run raw-unit frozen-adapter USCRN forecasts at both preselected horizons."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.post_release.data import OUT
from research.activation_transfer.moment.forecast_quality.predict import NAMES, adapter, cached_call, checkpoint_metadata, save_npz
from research.activation_transfer.moment.forecast_quality.data import hash_array


def main(model):
    with np.load(OUT / "panel.npz") as f:
        x = f["context"]
        ids = f["ids"]
    values = dict(ids=ids, context_sha256=hash_array(x))
    enc = adapter(model)
    _, hashes, versions = checkpoint_metadata(model)
    signature = json.dumps([model, values["context_sha256"], hashes, versions, 16, torch.get_num_threads(), "raw-f32-v1"], sort_keys=True)
    for h in (8,16):
        cache = OUT / ".passes" / f"{model}_raw_{h}.npz"
        values[f"raw_{h}"] = cached_call(enc, x.reshape(-1,480), h, 16, cache, signature)[0].reshape(*x.shape[:2], h)
    save_npz(OUT / (model+".npz"), **values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=NAMES)
    torch.set_num_threads(2)
    main(parser.parse_args().model)
