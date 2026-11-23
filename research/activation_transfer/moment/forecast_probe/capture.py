"""Capture the exact all-position input of the released forecast-task linear head."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MOMENT = HERE.parent
QUALITY = MOMENT / "forecast_quality" / "results"
WEATHER = MOMENT / "forecast_quality" / "post_release" / "results"
HEADS = Path("/tmp/tsfm_moment_nov22_heads")
MODEL = Path("/tmp/tsfm_moment_native/large")
CACHE = Path("/tmp/tsfm_moment_forecast_probe_20261122")
OUT = HERE / "results"


def terminal_split(context):
    """The last sixteen recorded values are withheld labels throughout fitting."""
    return context[:, :-16], context[:, -16:]


def rows(group):
    if group == "train":
        with np.load(HEADS / "train.npz") as f:
            return f["ids"], *terminal_split(f["context"]), f["split"]
    if group == "test":
        with np.load(QUALITY / "panel.npz") as f:
            keep = f["split"] == "holdout"
            return f["id"][keep], f["context"][keep], f["future"][keep], f["domain"][keep]
    with np.load(WEATHER / "panel.npz") as f:
        return f["ids"], f["context"], f["future"], f["region"]


def native(x, kind="edge"):
    lag = np.ascontiguousarray(x, dtype=np.float32)
    assert lag.ndim == 2 and lag.shape[1] <= 512 and np.isfinite(lag).all()
    pad = 512 - lag.shape[1]
    array = np.pad(lag, ((0, 0), (pad, 0)), mode=kind)
    mask = np.ones(array.shape, dtype=np.float32)
    mask[:, :pad] = 0
    return dict(x_enc=torch.from_numpy(array[:, None]), input_mask=torch.from_numpy(mask))


def model():
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained(str(MODEL), strict=True,
        model_kwargs={"task_name": "forecasting", "forecast_horizon": 16})
    m.init()
    m.eval().requires_grad_(False)
    assert type(m.head).__name__ == "ForecastingHead" and tuple(m.head.linear.weight.shape) == (16, 65536)
    m.head.linear.weight.zero_()
    m.head.linear.bias.zero_()
    return m


def checksum(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def key(ids, x):
    return checksum(ids) + checksum(x)


def run(m, group):
    ids, array, _, _ = rows(group)
    flat = array.reshape(-1, array.shape[-1])
    identities = np.repeat(ids, len(flat) // len(ids))
    target = CACHE / group
    target.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(flat), 32):
        stop = min(len(flat), start + 32)
        file = target / f"{start:04}_{stop:04}.npz"
        sign = key(identities[start:stop], flat[start:stop])
        if file.exists():
            with np.load(file) as old:
                assert str(old["key"]) == sign
            continue
        states, means, sds = [], [], []
        def record(_module, args):
            states.append(args[0][:, 0].detach().numpy().copy())
        hook = m.head.linear.register_forward_pre_hook(record)
        try:
            for chunk in np.array_split(flat[start:stop], int(np.ceil((stop - start) / 8))):
                with torch.inference_mode():
                    raw = m(**native(chunk)).forecast[:, 0].numpy()
                mu = m.normalizer.mean.reshape(-1).numpy().copy()
                sd = m.normalizer.stdev.reshape(-1).numpy().copy()
                ref = chunk.std(axis=1) + 1e-5
                assert np.max(np.abs(mu - chunk.mean(1)) / ref) < 2e-4
                assert np.max(np.abs(sd - ref) / ref) < 2e-5
                np.testing.assert_array_equal(raw, np.broadcast_to(mu[:, None], raw.shape))
                means.extend(mu)
                sds.extend(sd)
        finally:
            hook.remove()
        tmp = file.with_suffix(".running")
        with tmp.open("wb") as f:
            np.savez(f, h=np.concatenate(states), mu=means, sd=sds, key=sign)
        os.replace(tmp, file)
        if start % 128 == 0 or stop == len(flat):
            print(group, stop, len(flat), flush=True)


def features(group):
    ids, array, _, _ = rows(group)
    flat = array.reshape(-1, array.shape[-1])
    identities = np.repeat(ids, len(flat) // len(ids))
    h = np.empty((len(flat), 65536), np.float32)
    mu, sd = np.empty(len(flat)), np.empty(len(flat))
    for start in range(0, len(flat), 32):
        stop = min(len(flat), start + 32)
        with np.load(CACHE / group / f"{start:04}_{stop:04}.npz") as f:
            assert str(f["key"]) == key(identities[start:stop], flat[start:stop])
            h[start:stop], mu[start:stop], sd[start:stop] = f["h"], f["mu"], f["sd"]
    return h, mu, sd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="+", choices=["train", "test", "weather"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    engine = model()
    for name in args.group:
        run(engine, name)
